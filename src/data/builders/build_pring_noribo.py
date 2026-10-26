"""Build the ribosome-ablation sandbox: the PRING sandbox with the cytosolic ribosome removed from the candidates.

Run: uv run python -m src.data.builders.build_pring_noribo   (after build_pring_tools)
Then: uv run python -m src.eval.rounds <name> <R> --sandbox sandbox_noribo

Why: in the true-feedback baseline 87% of the agent's hits were ribosome-ribosome pairs (71 proteins whose
pairs are 98% hits). Removing those proteins asks what the agent finds without that shortcut (user decision
2026-10-26).

- Ribosome: candidate proteins whose gene starts with RPL or RPS, except kinases (RPS6K*), the muscle paralog
  RPL3L and the ribosome-binding RPS19BP1, plus the 40S subunit protein RACK1: 71 proteins.
- data/PRING/sandbox_noribo/ is sandbox/ as if the screen never contained them: their rows leave
  proteins.tsv, annotations.tsv and homologs.tsv (as query), every pair with one of them leaves scores.tsv.gz
  (scores unchanged, ranks close up), kb.csv is copied (it holds no candidate). hidden/ is shared; the
  evaluator's validity check makes a removed protein an unknown id.
"""

import csv
import gzip
import shutil

from src.data.paths import PRING, SANDBOX

OUT = PRING / "sandbox_noribo"
RIBOSOME_SIZE = 71


def is_ribosomal(gene):
    if gene == "RACK1":
        return True
    return gene.startswith(("RPL", "RPS")) and not gene.startswith(("RPS6K", "RPL3L", "RPS19BP1"))


def ribosome(proteins_tsv=SANDBOX / "proteins.tsv"):
    """Accessions of the candidate ribosomal proteins."""
    rows = csv.DictReader(open(proteins_tsv), delimiter="\t")
    return {r["accession"] for r in rows if r["candidate"] == "1" and is_ribosomal(r["gene"])}


def drop_rows(name, keep, opener=open):
    """Copy sandbox/<name> to OUT keeping the header and the rows where keep(fields) holds; rows dropped."""
    dropped = 0
    with opener(SANDBOX / name, "rt") as src, opener(OUT / name, "wt") as out:
        out.write(next(src))
        for line in src:
            if keep(line.rstrip("\n").split("\t")):
                out.write(line)
            else:
                dropped += 1
    return dropped


def main():
    drop = ribosome()
    assert len(drop) == RIBOSOME_SIZE, len(drop)
    OUT.mkdir(exist_ok=True)
    shutil.copy(SANDBOX / "kb.csv", OUT / "kb.csv")
    counts = {
        "proteins.tsv": drop_rows("proteins.tsv", lambda f: f[0] not in drop),
        "annotations.tsv": drop_rows("annotations.tsv", lambda f: f[0] not in drop),
        "homologs.tsv": drop_rows("homologs.tsv", lambda f: f[0] not in drop),
        "scores.tsv.gz": drop_rows("scores.tsv.gz", lambda f: f[0] not in drop and f[1] not in drop, gzip.open),
    }
    n = sum(r["candidate"] == "1" for r in csv.DictReader(open(OUT / "proteins.tsv"), delimiter="\t"))
    assert counts["scores.tsv.gz"] == 2_035_153 - n * (n - 1) // 2, counts  # every removed pair, nothing else
    print(f"{OUT.relative_to(PRING)}: {n:,} candidates, rows dropped {counts}")


if __name__ == "__main__":
    main()
