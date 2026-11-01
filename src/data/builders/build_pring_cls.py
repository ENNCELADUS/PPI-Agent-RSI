"""Build the classification-test sandbox: the PRING sandbox with the KB cut to train and no predictor.

Run: uv run python -m src.data.builders.build_pring_cls   (after build_pring_sandbox and build_pring_tools)
Then: uv run python -m src.eval.pring_test <name>

Why: PRING val leaves the agent's KB for the RSI part (user decision 2026-11-01), so the fixed agent that
predicts PRING's test pairs knows only train. The first arm is the LLM alone (no DL predictor).

- data/PRING/sandbox_cls/: kb.csv = the sandbox KB rows that are in hidden/train.csv; proteins.tsv and
  annotations.tsv keep train-KB and test proteins (val-only proteins go); homologs.tsv keeps rows whose
  target is a train-KB protein. No scores.tsv.gz. hidden/ is shared; the test pairs are hidden/test.csv.
"""

import csv

from src.data.paths import HIDDEN, PRING, SANDBOX

OUT = PRING / "sandbox_cls"


def rows(path, delimiter):
    with open(path, newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        return next(reader), list(reader)


def write(path, header, body, delimiter):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter=delimiter)
        w.writerow(header)
        w.writerows(body)


def main():
    _, train = rows(HIDDEN / "train.csv", ",")
    train_pairs = {(a, b) for a, b, _ in train}
    head, kb = rows(SANDBOX / "kb.csv", ",")
    kb = [r for r in kb if (r[0], r[1]) in train_pairs]
    assert len(kb) == len(train_pairs), "train pair missing from the sandbox KB"
    kb_proteins = {x for r in kb for x in r[:2]}

    p_head, proteins = rows(SANDBOX / "proteins.tsv", "\t")
    candidates = {r[0] for r in proteins if r[3] == "1"}
    assert not kb_proteins & candidates, "test protein in KB"
    keep = kb_proteins | candidates
    OUT.mkdir(exist_ok=True)
    write(OUT / "kb.csv", head, kb, ",")
    write(OUT / "proteins.tsv", p_head, [r for r in proteins if r[0] in keep], "\t")
    a_head, ann = rows(SANDBOX / "annotations.tsv", "\t")
    write(OUT / "annotations.tsv", a_head, [r for r in ann if r[0] in keep], "\t")
    h_head, hom = rows(SANDBOX / "homologs.tsv", "\t")
    hom = [r for r in hom if r[0] in keep and r[1] in kb_proteins]
    write(OUT / "homologs.tsv", h_head, hom, "\t")
    stats = {
        "kb_pairs": len(kb),
        "kb_pos": sum(r[2] == "1" for r in kb),
        "kb_proteins": len(kb_proteins),
        "test_proteins": len(candidates),
        "test_proteins_with_homolog": len({r[0] for r in hom} & candidates),
    }
    print(stats)


if __name__ == "__main__":
    main()
