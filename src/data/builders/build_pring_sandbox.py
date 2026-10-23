"""Build the PRING dev sandbox: a closed-world, internal benchmark for debugging the agent.

Run: uv run python -m src.data.builders.build_pring_sandbox
Inputs in data/PRING/raw/: PRING human BFS split as packaged by the motif-graph project
(PRING commit edbdc01, node ids rewritten to node_*), node_sequences.tsv (node -> UniProt,
sequence; from H20) and one bulk all-human UniProt name table. Fetch them with:

    M=~/Documents/motif-graph-prompt-tuning-for-inductive-protein-protein-interaction-prediction/data/benchmark_2025_neurips
    mkdir -p data/PRING/raw/breadth_first && cp $M/positive_edges.txt data/PRING/raw/
    cp $M/breadth_first/{train,test}_edges.txt data/PRING/raw/breadth_first/
    H20=/2023533015/topology-conditioned-inductive-edge-prediction/data/official_sequence_baselines
    scp -P 30838 root@10.15.171.204:$H20/node_sequences.tsv data/PRING/raw/
    curl -C - --retry 5 -o data/PRING/raw/uniprot_human_names.tsv \
      'https://rest.uniprot.org/uniprotkb/stream?query=organism_id:9606&fields=accession,gene_primary,protein_name&format=tsv'

Sandbox numbers are never reported: every label here is public, so a web-enabled agent would
look it up.

- kb.csv = PRING BFS train pairs without self pairs, minus the transductive held-out positives.
- pool_inductive = 500 positives + 500 negatives sampled from PRING's BFS test pairs. Test
  proteins never appear in the KB (PRING node split, <= 40% identity to train).
- pool_transductive = 500 KB positives held out so that both endpoints keep another KB partner
  (mirrors the private split, where test proteins have public partners) + 500 PRING-sampled
  negatives over their endpoints that are neither PRING positives nor KB pairs.
- Self pairs are dropped everywhere, as in the private split: PRING keeps them as positives,
  so a self pair in a pool would be a guaranteed positive.
- Columns follow the private split (uniprotID_A, uniprotID_B, isInteraction). Labels, permuted
  labels and the anonymised-id map go to hidden/, never to the agent; pools are shuffled.
- anon/ repeats the agent files with random opaque ids and sequences only (memorisation arm).
"""
import csv
import hashlib
import random
from collections import Counter
from datetime import date

from src.data.negatives import IdentityKeys, pring_negatives
from src.data.paths import HIDDEN, PRING, SANDBOX
from src.data.paths import PRING_RAW as RAW

INPUTS = ("positive_edges.txt", "breadth_first/train_edges.txt", "breadth_first/test_edges.txt",
          "node_sequences.tsv", "uniprot_human_names.tsv")
SEED = 0
N = 500  # positives, and negatives, per pool
PAIR = ["uniprotID_A", "uniprotID_B"]


def edges(path):
    """Unordered non-self node pairs -> label (files without a label column are all positive)."""
    out = {}
    for row in csv.reader(open(path), delimiter="\t"):
        if row[0] == row[1]:
            continue
        pair, label = tuple(sorted(row[:2])), int(row[2]) if len(row) > 2 else 1
        assert out.setdefault(pair, label) == label, f"conflicting labels for {pair}"
    return out


def hold_out(train, rng):
    """N train positives whose endpoints each keep at least one other KB positive."""
    pos = sorted(p for p, y in train.items() if y)
    rng.shuffle(pos)
    deg = Counter(x for p in pos for x in p)
    held = []
    for a, b in pos:
        if deg[a] > 1 and deg[b] > 1:
            held.append((a, b))
            deg[a] -= 1
            deg[b] -= 1
            if len(held) == N:
                return held
    raise AssertionError("not enough holdable positives")


def write(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t" if path.suffix == ".tsv" else ",")
        w.writerow(header)
        w.writerows(rows)


def main():
    rng = random.Random(SEED)
    node2acc, seq = {}, {}
    for r in csv.DictReader(open(RAW / "node_sequences.tsv"), delimiter="\t"):
        node2acc[r["node_id"]], seq[r["node_id"]] = r["uniprot_id"], r["sequence"]
    assert len(set(node2acc.values())) == len(node2acc), "two nodes share an accession"
    names = {r[0]: r[1:] for r in csv.reader(open(RAW / "uniprot_human_names.tsv"), delimiter="\t")}

    known = set(edges(RAW / "positive_edges.txt"))  # every PRING positive, any split
    train = edges(RAW / "breadth_first/train_edges.txt")
    test = edges(RAW / "breadth_first/test_edges.txt")

    test_pos = sorted(p for p, y in test.items() if y)
    test_neg = sorted(p for p, y in test.items() if not y)
    inductive = {p: 1 for p in rng.sample(test_pos, N)} | {p: 0 for p in rng.sample(test_neg, N)}

    held = hold_out(train, rng)
    held_set = set(held)
    kb = {p: y for p, y in train.items() if p not in held_set}
    negs = pring_negatives(held, N, lambda ks: any(k in known or k in kb for k in ks), rng, IdentityKeys())
    transductive = {p: 1 for p in held} | {p: 0 for p in negs}
    pools = {"inductive": inductive, "transductive": transductive}

    kb_proteins = {x for p in kb for x in p}
    kb_partners = Counter(x for p, y in kb.items() if y for x in p)
    for pool in pools.values():
        assert len(pool) == 2 * N and sum(pool.values()) == N
        assert not pool.keys() & kb.keys(), "pool pair in KB"
        assert not any(p in known for p, y in pool.items() if not y), "negative is a PRING positive"
    assert not {x for p in inductive for x in p} & kb_proteins, "inductive protein in KB"
    assert all(kb_partners[x] for p in held for x in p), "transductive endpoint without KB partner"

    proteins = sorted(kb_proteins | {x for pool in pools.values() for p in pool for x in p})
    opaque = proteins[:]
    rng.shuffle(opaque)
    anon = {x: f"P{i:05d}" for i, x in enumerate(opaque, 1)}
    ids = {"": node2acc, "anon": anon}

    for sub, name in ids.items():
        out, hidden = SANDBOX / sub, HIDDEN / sub
        if sub:
            write(out / "proteins.tsv", ["id", "sequence"], sorted((name[x], seq[x]) for x in proteins))
        else:
            write(out / "proteins.tsv", ["accession", "gene", "protein_name", "sequence"],
                  [(name[x], *names.get(name[x], ("", "")), seq[x]) for x in proteins])
        write(out / "kb.csv", PAIR + ["isInteraction", "evidence"],
              sorted((name[a], name[b], y, "pring") for (a, b), y in kb.items()))
        for pool_name, pool in pools.items():
            order = sorted(pool)
            random.Random(f"{SEED}-{pool_name}").shuffle(order)  # same order in both id spaces
            labels = [pool[p] for p in order]
            permuted = labels[:]
            random.Random(f"{SEED}-{pool_name}-permuted").shuffle(permuted)
            rows = [(name[a], name[b]) for a, b in order]
            write(out / f"pool_{pool_name}.csv", PAIR, rows)
            write(hidden / f"{pool_name}.csv", PAIR + ["isInteraction"], [(*r, y) for r, y in zip(rows, labels)])
            write(hidden / f"{pool_name}_permuted.csv", PAIR + ["isInteraction"],
                  [(*r, y) for r, y in zip(rows, permuted)])
    write(HIDDEN / "anon_map.tsv", ["id", "accession"], sorted((anon[x], node2acc[x]) for x in proteins))

    unnamed = sum(node2acc[x] not in names for x in proteins)
    stats = {
        "kb_pos": sum(kb.values()), "kb_neg": len(kb) - sum(kb.values()), "kb_proteins": len(kb_proteins),
        "proteins": len(proteins), "proteins_without_uniprot_entry": unnamed,
        "self_pairs_dropped_from_train": sum(1 for r in csv.reader(open(RAW / "breadth_first/train_edges.txt"),
                                                                    delimiter="\t") if r[0] == r[1]),
    }
    checksums = {f: hashlib.sha256((RAW / f).read_bytes()).hexdigest()[:16] for f in INPUTS}
    readme = [
        "# PRING dev sandbox (generated by src/data/builders/build_pring_sandbox.py)", "",
        f"Built {date.today()}, seed {SEED}. Internal dev sandbox; never report these numbers.",
        "Give the agent `sandbox/` (named arm) or `sandbox/anon/` (anonymised arm), never `hidden/`.", "",
        "| Item | Value |", "|---|---|",
        *(f"| {k} | {v:,} |" for k, v in stats.items()),
        *(f"| pool_{k} | {N} pos + {N} neg |" for k in pools), "",
        "Inputs (sha256 prefix): PRING human BFS split at commit edbdc01 via the motif-graph package;",
        "node_sequences.tsv from H20; UniProt names from rest.uniprot.org organism_id:9606 stream.", "",
        *(f"- `raw/{f}` {c}" for f, c in checksums.items()), "",
    ]
    (PRING / "README.md").write_text("\n".join(readme))
    print(stats)


if __name__ == "__main__":
    main()
