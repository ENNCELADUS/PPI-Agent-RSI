"""Active-learning B0 arm of the memory agent's final test: B0 retrained on the knowledge base plus the round labels.

    uv run python -m src.eval.al_predictor export <name>            # pair files for H20 (node ids)
    uv run python -m src.eval.al_predictor summary <name> <scores>  # after training and scoring on H20

The arm uses the same revealed data as the agent arms: every case of data/PRING/runs/<name>/memory/round10/
cases.jsonl (`hit` = 1, `not detected` = 0). It is the round-10 model of the usual pool-based active-learning
loop that retrains from scratch on all labels after each round; the picks are the agent's, so no earlier round's
model changes them. export writes data/PRING/predictor/al/<name>/revealed_pairs.tsv (node_a, node_b, label) for
the B0 trainer's data.extra_training_pairs and final_pairs.tsv (node_a, node_b, label) of hidden/final_test.csv
for src.predictor.score. Training (motif-graph repo, branch al-revealed: configs/split_seed42/b0_v31.yaml with
the extra pairs, same seed) and scoring run on H20; summary reads the merged scores (node_a, node_b, logit,
label) and writes data/PRING/runs/<name>/test_al_b0/summary.json: the retrained B0 ("al_b0", raw sigmoid as for
the frozen B0) and the frozen B0 on the same pairs, with the paired bootstrap of al_b0 - b0.
"""

import argparse
import csv
import json
import math

from src.data.builders.build_pring_memory import FINAL_TEST
from src.data.ids import pair
from src.data.paths import PRING, PRING_RAW
from src.eval.pring_test import classification, paired_bootstrap, predictor_p

RUNS = PRING / "runs"


def nodes():
    rows = csv.DictReader(open(PRING_RAW / "node_sequences.tsv"), delimiter="\t")
    return {r["uniprot_id"]: r["node_id"] for r in rows}


def final_test():
    """[(a, b, label)] of hidden/final_test.csv as canonical accession pairs."""
    rows = csv.DictReader(open(FINAL_TEST))
    return [(*pair(r["uniprotID_A"], r["uniprotID_B"]), int(r["isInteraction"])) for r in rows]


def write_tsv(path, rows):
    with open(path, "w", newline="") as f:
        csv.writer(f, delimiter="\t", lineterminator="\n").writerows(rows)


def export(name):
    node = nodes()
    out = PRING / "predictor" / "al" / name
    out.mkdir(parents=True, exist_ok=True)
    cases = [json.loads(line) for line in open(RUNS / name / "memory" / "round10" / "cases.jsonl")]
    revealed = {pair(node[c["a"]], node[c["b"]]): int(c["result"] == "hit") for c in cases}
    assert len(revealed) == len(cases), "duplicate cases"
    test = final_test()
    assert not {pair(node[a], node[b]) for a, b, _ in test} & revealed.keys(), "a revealed pair is a final-test pair"
    write_tsv(out / "revealed_pairs.tsv", [(a, b, y) for (a, b), y in sorted(revealed.items())])
    write_tsv(out / "final_pairs.tsv", [(node[a], node[b], y) for a, b, y in test])
    print(f"{out}: {len(revealed):,} revealed pairs ({sum(revealed.values()):,} hits), {len(test):,} final-test pairs")


def summary(name, scores):
    node = nodes()
    logit = {pair(r["node_a"], r["node_b"]): float(r["logit"]) for r in csv.DictReader(open(scores), delimiter="\t")}
    test = final_test()
    y = [label for _, _, label in test]
    al = [1 / (1 + math.exp(-logit[pair(node[a], node[b])])) for a, b, _ in test]
    frozen = predictor_p("b0", [(a, b) for a, b, _ in test])
    b0 = [frozen[(a, b)] for a, b, _ in test]
    out = RUNS / name / "test_al_b0"
    out.mkdir(exist_ok=True)
    result = {
        "pairs": len(test),
        "positives": sum(y),
        "scores": str(scores),
        "al_b0": classification(y, al),
        "b0": classification(y, b0),
        "al_b0_minus_b0": paired_bootstrap(y, al, b0),
    }
    (out / "summary.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stage", choices=("export", "summary"))
    parser.add_argument("name")
    parser.add_argument("scores", nargs="?", help="summary: merged src.predictor.score output")
    args = parser.parse_args()
    if args.stage == "export":
        export(args.name)
    else:
        assert args.scores, "summary needs the merged scores file"
        summary(args.name, args.scores)


if __name__ == "__main__":
    main()
