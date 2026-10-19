"""Shared paths and ID normalization for the agent-benchmark split.

Every protein is mapped to a gene-level node: Entrez GeneID when UniProt
knows one, otherwise the bare UniProt accession. Pairs are compared as
unordered node pairs, so isoforms and TrEMBL entries of the same gene match.
"""
import csv
import gzip
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TMP = ROOT / "data" / "TMP"
RAW = TMP / "raw"
PUBLIC = TMP / "public_db"
OUT = TMP / "agent_split"
IDMAP = PUBLIC / "uniprot_idmap.tsv.gz"

RAW_FILES = {
    "pretrain_pos": (RAW / "homo_positive_train.txt", "\t"),
    "pretrain_neg": (RAW / "homo_negative_train.csv", ","),
    "finetune_pos": (RAW / "homo_positive_finetune.txt", "\t"),
    "finetune_neg": (RAW / "homo_negative_pair_fintune.csv", ","),
    "test": (RAW / "depedent_test.csv", ","),
}


def strip_isoform(acc):
    return acc.split("-", 1)[0].strip()


def load_idmap():
    """Return (acc->gene, string_id->acc, ensg->[acc])."""
    acc2gene, string2acc, ensg2acc = {}, {}, {}
    with gzip.open(IDMAP, "rt") as f:
        for acc, kind, value in csv.reader(f, delimiter="\t"):
            if kind == "GeneID":
                # Keep the smallest GeneID so the mapping is deterministic.
                if acc not in acc2gene or int(value) < int(acc2gene[acc]):
                    acc2gene[acc] = value
            elif kind == "STRING":
                string2acc[value] = acc
            elif kind == "Ensembl":
                ensg2acc.setdefault(value.split(".", 1)[0], []).append(acc)
    return acc2gene, string2acc, ensg2acc


def node_of(acc, acc2gene):
    acc = strip_isoform(acc)
    gene = acc2gene.get(acc)
    return f"g{gene}" if gene else acc


def raw_universe():
    """All UniProt accessions used anywhere in the raw TMP files."""
    accs = set()
    for path, sep in RAW_FILES.values():
        with open(path) as f:
            rows = csv.reader(f, delimiter=sep)
            next(rows)
            for row in rows:
                accs.add(row[0])
                accs.add(row[1])
    accs.discard("NA")
    return accs
