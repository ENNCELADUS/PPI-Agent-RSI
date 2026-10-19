"""Build the agent-benchmark train/test split from raw TMP positives + public DB tables.

Run from scripts/agent_split/: uv run python build_split.py
- Positives: all raw TMP positives (pretrain, finetune, GT test) as unordered pairs;
  drops NA, self loops and pairs the raw data also labels negative. Raw negatives
  are not used otherwise.
- A positive is "public" if any gene-level node pair of it is in any public_db/*.tsv.gz.
- test.csv  = every non-public positive + an equal number of PRING-sampled negatives
  that have no public evidence at all (any DB, STRING at any score, tested negatives).
- train.csv = every public positive + clean experimentally tested negatives
  (Negatome, IntAct negative) + PRING-sampled negatives up to 1:1. This file is the
  agent's reference KB; `evidence` says where each row comes from.

PRING negative sampling (Zheng et al. 2025, data_process/graph_gen/negative_sample.py,
after Bernett et al. 2024): both proteins are drawn from the multiset of the split's
positive endpoints (so negatives keep the positives' degree distribution), and the
second protein is redrawn until the pair is not a self pair, not a known PPI and not
already drawn. Train treats TMP positives, curated DBs and STRING > 700 as known
(as PRING does); test rejects any public evidence so test stays non-public.
"""
import csv
import gzip
import itertools
import json
import random

import duckdb

from common import OUT, PUBLIC, PUBLIC_NEG, RAW_FILES, IdMap, raw_universe

SEED = 0
CURATED = ("biogrid", "bioplex", "corum", "hippie", "intact")
DBS = CURATED + ("string",)
STRING_HIGH = 700  # PRING counts STRING combined score > 0.7 as a known PPI


class PairKeys:
    """Unordered gene-level node pairs of an accession pair, as integer keys."""

    def __init__(self, idmap):
        self.idmap, self.index = idmap, {}

    def node(self, name):
        return self.index.setdefault(name, len(self.index))

    def key(self, x, y):
        i, j = sorted((self.node(x), self.node(y)))
        return i << 32 | j

    def of(self, a, b):
        """Keys over all distinct-node combinations of the two proteins."""
        na, nb = self.idmap.acc_nodes(a), self.idmap.acc_nodes(b)
        return [self.key(x, y) for x, y in itertools.product(na, nb) if x != y]

    def same_gene(self, a, b):
        return bool(self.idmap.acc_nodes(a) & self.idmap.acc_nodes(b))


def raw_positives():
    reads = []
    for name, (path, sep) in RAW_FILES.items():
        if name == "test":
            label, cols = "CASE truth_de WHEN 'positive' THEN 1 ELSE 0 END", "p1 AS a, p2 AS b"
        else:
            label, cols = ("1" if name.endswith("_pos") else "0"), '"protein.id.A" AS a, "protein.id.B" AS b'
        reads.append(f"SELECT {cols}, {label} AS y FROM read_csv('{path}', delim='{sep}', header=true, all_varchar=true)")
    con = duckdb.connect()
    con.execute(f"""
        CREATE TABLE raw AS SELECT least(a, b) AS a, greatest(a, b) AS b, y FROM ({' UNION ALL '.join(reads)})
        WHERE a <> 'NA' AND b <> 'NA' AND a <> b
    """)
    rows = con.execute("SELECT a, b, min(y) = 0 FROM raw GROUP BY a, b HAVING max(y) = 1 ORDER BY a, b").fetchall()
    return [(a, b) for a, b, conflict in rows if not conflict], sum(conflict for *_, conflict in rows)


def load_db(name, keys):
    scores = {}
    with gzip.open(PUBLIC / f"{name}.tsv.gz", "rt") as f:
        next(f)
        for line in f:
            a, b, score, _ = line.split("\t", 3)
            k = keys.key(a, b)
            score = float(score) if score else None
            if k not in scores or (score is not None and (scores[k] is None or score > scores[k])):
                scores[k] = score
    return scores


def tested_negatives(universe, keys):
    """Experimentally tested negatives mapped onto TMP accessions, with their sources."""
    node2acc = {}
    for acc in sorted(universe):
        for node in keys.idmap.acc_nodes(acc):
            node2acc.setdefault(node, acc)
    pairs = {}
    for path in sorted(PUBLIC_NEG.glob("*.tsv.gz")):
        source = path.name.removesuffix(".tsv.gz")
        with gzip.open(path, "rt") as f:
            next(f)
            for line in f:
                ends = []
                for acc in line.split():
                    if acc in universe:
                        ends.append(acc)
                    else:
                        hits = sorted(node2acc[n] for n in keys.idmap.acc_nodes(acc) if n in node2acc)
                        ends.append(hits[0] if hits else None)
                if None in ends or ends[0] == ends[1]:
                    continue
                pairs.setdefault(tuple(sorted(ends)), set()).add(source)
    return {p: ";".join(sorted(s)) for p, s in sorted(pairs.items())}


def pring_negatives(positives, n, is_known, rng, keys):
    candidates = [acc for pair in positives for acc in pair]
    drawn = {}
    while len(drawn) < n:
        a = rng.choice(candidates)
        while True:
            b = rng.choice(candidates)
            pair = (a, b) if a < b else (b, a)
            if a != b and not keys.same_gene(a, b) and pair not in drawn and not is_known(keys.of(*pair)):
                break
        drawn[pair] = None
    return list(drawn)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    keys = PairKeys(IdMap())
    universe = raw_universe()

    positives, conflicts = raw_positives()
    db = {name: load_db(name, keys) for name in DBS}
    string_high = {k for k, s in db["string"].items() if s is not None and s > STRING_HIGH}
    curated = set().union(*(db[name].keys() for name in CURATED))
    public_any = curated | db["string"].keys()

    pos_keys = {p: keys.of(*p) for p in positives}
    tmp_pos = {k for ks in pos_keys.values() for k in ks}
    public = {p for p, ks in pos_keys.items() if any(k in public_any for k in ks)}
    test_pos = [p for p in positives if p not in public]
    train_pos = [p for p in positives if p in public]

    # Clean tested negatives: not a TMP positive, not curated, STRING below the PRING cut.
    tested = tested_negatives(universe, keys)
    tested_keys = {k for p in tested for k in keys.of(*p)}
    funnel = {"in_tmp_universe": len(tested)}
    clean = {}
    for p, source in tested.items():
        ks = keys.of(*p)
        if any(k in tmp_pos for k in ks):
            continue
        funnel["after_tmp_positive_filter"] = funnel.get("after_tmp_positive_filter", 0) + 1
        if any(k in curated for k in ks):
            continue
        funnel["after_curated_filter"] = funnel.get("after_curated_filter", 0) + 1
        if max((db["string"].get(k) or 0 for k in ks), default=0) >= STRING_HIGH:
            continue
        clean[p] = source
    funnel["clean"] = len(clean)

    # Test first, so train sampling can exclude every test pair.
    test_neg = pring_negatives(
        test_pos, len(test_pos),
        lambda ks: any(k in tmp_pos or k in public_any or k in tested_keys for k in ks), rng, keys)
    test_neg_keys = {k for p in test_neg for k in keys.of(*p)}
    train_neg = pring_negatives(
        train_pos, len(train_pos) - len(clean),
        lambda ks: any(k in tmp_pos or k in curated or k in string_high or k in tested_keys or k in test_neg_keys
                       for k in ks), rng, keys)

    train = [(p, 1, "public_db") for p in train_pos] + [(p, 0, s) for p, s in clean.items()] + \
            [(p, 0, "pring_sampled") for p in train_neg]
    test = [(p, 1) for p in test_pos] + [(p, 0) for p in test_neg]
    train.sort()
    test.sort()
    with open(OUT / "train.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uniprotID_A", "uniprotID_B", "isInteraction", "evidence"])
        w.writerows((a, b, y, ev) for (a, b), y, ev in train)
    with open(OUT / "test.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uniprotID_A", "uniprotID_B", "isInteraction"])
        w.writerows((a, b, y) for (a, b), y in test)

    # Per-pair DB evidence for every output row.
    ann_path = OUT / "pair_annotations.tsv.gz"
    with gzip.open(ann_path, "wt") as f:
        f.write("\t".join(["a", "b", "y", "split", "evidence"] + [c for n in DBS for c in (f"in_{n}", f"{n}_score")] + ["public"]) + "\n")
        rows = [(p, y, "train", ev) for p, y, ev in train] + [(p, y, "test", "") for p, y in test]
        for (a, b), y, split, ev in rows:
            ks = keys.of(a, b)
            cells, any_db = [], False
            for name in DBS:
                hit = [k for k in ks if k in db[name]]
                scores = [db[name][k] for k in hit if db[name][k] is not None]
                cells += [str(bool(hit)).lower(), str(max(scores)) if scores else ""]
                any_db |= bool(hit)
            f.write("\t".join([a, b, str(y), split, ev] + cells + [str(any_db).lower()]) + "\n")
    duckdb.execute(f"COPY (SELECT * FROM read_csv('{ann_path}', delim='\t', header=true)) "
                   f"TO '{OUT / 'pair_annotations.parquet'}' (FORMAT parquet)")
    ann_path.unlink()

    stats = {
        "conflicting_pairs_dropped": conflicts,
        "public_dbs": list(DBS),
        "train": {"pos": len(train_pos), "neg_tested": len(clean), "neg_pring": len(train_neg)},
        "test": {"pos": len(test_pos), "neg_pring": len(test_neg)},
        "tested_negative_funnel": funnel,
        "tested_negative_sources": dict(sorted(
            {s: sum(1 for v in clean.values() if v == s) for s in set(clean.values())}.items())),
    }
    (OUT / "split_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
