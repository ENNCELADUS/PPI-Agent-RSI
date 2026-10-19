"""Build the agent-benchmark train/test split from raw TMP pairs + public DB tables.

Run: uv run python build_split.py
- Pool: all raw positives/negatives (pretrain, finetune, GT test), deduplicated as
  unordered pairs; drops NA, self loops and pairs labelled both ways.
- A pair is "public" if its gene-level pair appears in any public_db/*.tsv.gz.
- test.csv  = every non-public positive + non-public negatives at TEST_NEG_RATIO:1.
- train.csv = everything else (the agent's reference KB), all negatives kept.
"""
import json

import duckdb

from common import OUT, PUBLIC, RAW_FILES, load_idmap, node_of, raw_universe

TEST_NEG_RATIO = 38  # matches the original GT test prior (183,909 / 4,788)
SEED = 0.42
STRING_LEVELS = (150, 400, 700, 900)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")

    acc2gene = load_idmap()[0]
    con.execute("CREATE TABLE node_map(acc VARCHAR, node VARCHAR)")
    con.executemany("INSERT INTO node_map VALUES (?, ?)", [(a, node_of(a, acc2gene)) for a in raw_universe()])

    reads = []
    for name, (path, sep) in RAW_FILES.items():
        if name == "test":
            label = "CASE truth_de WHEN 'positive' THEN 1 ELSE 0 END"
            cols = "p1 AS a, p2 AS b"
        else:
            label = "1" if name.endswith("_pos") else "0"
            cols = '"protein.id.A" AS a, "protein.id.B" AS b'
        reads.append(f"SELECT {cols}, {label} AS y, '{name}' AS src FROM read_csv('{path}', delim='{sep}', header=true, all_varchar=true)")
    con.execute(f"""
        CREATE TABLE raw AS
        SELECT least(a, b) AS a, greatest(a, b) AS b, y, src FROM ({' UNION ALL '.join(reads)})
        WHERE a <> 'NA' AND b <> 'NA' AND a <> b
    """)
    con.execute("""
        CREATE TABLE pairs AS
        SELECT a, b, max(y) AS y, string_agg(DISTINCT src, ';' ORDER BY src) AS sources
        FROM raw GROUP BY a, b HAVING count(DISTINCT y) = 1
    """)
    conflicts = con.execute("SELECT count(*) FROM (SELECT a, b FROM raw GROUP BY a, b HAVING count(DISTINCT y) > 1)").fetchone()[0]
    con.execute("""
        CREATE TABLE pairs_n AS
        SELECT p.*, least(ma.node, mb.node) AS na, greatest(ma.node, mb.node) AS nb
        FROM pairs p JOIN node_map ma ON p.a = ma.acc JOIN node_map mb ON p.b = mb.acc
    """)

    dbs = sorted(p.name.removesuffix(".tsv.gz") for p in PUBLIC.glob("*.tsv.gz") if p.name != "uniprot_idmap.tsv.gz")
    joins, flags = [], []
    for db in dbs:
        con.execute(f"""
            CREATE TABLE db_{db} AS
            SELECT a, b, max(TRY_CAST(score AS DOUBLE)) AS score
            FROM read_csv('{PUBLIC / (db + '.tsv.gz')}', delim='\t', header=true, all_varchar=true, quote='')
            GROUP BY a, b
        """)
        joins.append(f"LEFT JOIN db_{db} ON p.na = db_{db}.a AND p.nb = db_{db}.b")
        flags.append(f"db_{db}.a IS NOT NULL AS in_{db}, db_{db}.score AS {db}_score")
    con.execute(f"CREATE TABLE ann AS SELECT p.*, {', '.join(flags)} FROM pairs_n p {' '.join(joins)}")
    any_db = " OR ".join(f"in_{db}" for db in dbs)
    con.execute(f"ALTER TABLE ann ADD COLUMN public BOOLEAN; UPDATE ann SET public = ({any_db})")

    n_test_pos = con.execute("SELECT count(*) FROM ann WHERE y = 1 AND NOT public").fetchone()[0]
    con.execute(f"""
        CREATE TABLE test AS
        SELECT a, b, y FROM ann WHERE y = 1 AND NOT public
        UNION ALL
        SELECT a, b, y FROM (
            SELECT a, b, y FROM ann WHERE y = 0 AND NOT public ORDER BY hash(a || '_' || b || '{SEED}')
            LIMIT {n_test_pos * TEST_NEG_RATIO})
    """)
    con.execute("CREATE TABLE train AS SELECT a, b, y FROM ann ANTI JOIN test USING (a, b)")

    for name in ("train", "test"):
        con.execute(f"COPY (SELECT a AS uniprotID_A, b AS uniprotID_B, y AS isInteraction FROM {name} ORDER BY a, b) TO '{OUT / (name + '.csv')}' (HEADER)")
    con.execute(f"COPY (SELECT * EXCLUDE (na, nb) FROM ann) TO '{OUT / 'pair_annotations.parquet'}' (FORMAT parquet)")

    stats = {"conflicting_pairs_dropped": conflicts, "public_dbs": dbs}
    for name in ("train", "test"):
        pos, neg = con.execute(f"SELECT sum(y), count(*) - sum(y) FROM {name}").fetchone()
        prots = con.execute(f"SELECT count(*) FROM (SELECT a FROM {name} UNION SELECT b FROM {name})").fetchone()[0]
        stats[name] = {"pos": int(pos), "neg": int(neg), "proteins": prots}
    stats["positives_in_db"] = {db: con.execute(f"SELECT count(*) FROM ann WHERE y = 1 AND in_{db}").fetchone()[0] for db in dbs}
    stats["negatives_in_db"] = {db: con.execute(f"SELECT count(*) FROM ann WHERE y = 0 AND in_{db}").fetchone()[0] for db in dbs}
    if "string" in dbs:
        other = " OR ".join(f"in_{db}" for db in dbs if db != "string") or "false"
        stats["nonpublic_pos_if_string_threshold"] = {
            t: con.execute(f"SELECT count(*) FROM ann WHERE y = 1 AND NOT ({other}) AND coalesce(string_score, 0) < {t}").fetchone()[0]
            for t in STRING_LEVELS
        }
    stats["test_by_original_source"] = dict(con.execute(
        "SELECT sources, count(*) FROM ann JOIN test USING (a, b) WHERE test.y = 1 GROUP BY sources ORDER BY 2 DESC").fetchall())
    (OUT / "split_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
