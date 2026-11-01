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

from src.data.builders.build_pring_noribo import drop_rows
from src.data.paths import HIDDEN, PRING, SANDBOX

OUT = PRING / "sandbox_cls"


def main():
    train = {(r["uniprotID_A"], r["uniprotID_B"]) for r in csv.DictReader(open(HIDDEN / "train.csv"))}
    kb = {x for p in train for x in p}
    rows = csv.DictReader(open(SANDBOX / "proteins.tsv"), delimiter="\t")
    candidates = {r["accession"] for r in rows if r["candidate"] == "1"}
    assert not kb & candidates, "test protein in KB"
    keep = kb | candidates
    OUT.mkdir(exist_ok=True)
    dropped = {
        "kb.csv": drop_rows("kb.csv", lambda f: tuple(f[:2]) in train, OUT),
        "proteins.tsv": drop_rows("proteins.tsv", lambda f: f[0] in keep, OUT),
        "annotations.tsv": drop_rows("annotations.tsv", lambda f: f[0] in keep, OUT),
        "homologs.tsv": drop_rows("homologs.tsv", lambda f: f[0] in keep and f[1] in kb, OUT),
    }
    assert sum(1 for _ in open(OUT / "kb.csv")) - 1 == len(train), "train pair missing from the sandbox KB"
    print(f"{OUT.relative_to(PRING)}: KB {len(train):,} pairs, {len(kb):,} proteins; rows dropped {dropped}")


if __name__ == "__main__":
    main()
