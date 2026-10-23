"""TMP protein identity: accessions as gene-level nodes, and the raw TMP universe.

Every protein expands to the set of gene-level nodes it may stand for:
Entrez GeneIDs from UniProt, plus the Entrez ID of its STRING protein (which
also resolves TrEMBL/secondary accessions), else its STRING protein ID, else
the bare accession. A pair is
matched against a database if any node combination of the two proteins is in
it, so multi-gene proteins (e.g. histone H4) and isoforms are not missed.
"""
import csv
import gzip

from src.data.paths import IDMAP, RAW_FILES, STRING_ALIAS


def strip_isoform(acc):
    return acc.split("-", 1)[0].strip()


class IdMap:
    """acc -> gene nodes, ENSP -> gene nodes, ENSG -> accs."""

    def __init__(self):
        self.acc2genes, self.ensg2accs = {}, {}
        string2accs = {}
        with gzip.open(IDMAP, "rt") as f:
            for acc, kind, value in csv.reader(f, delimiter="\t"):
                if kind == "GeneID":
                    self.acc2genes.setdefault(acc, set()).add(f"g{value}")
                elif kind == "STRING":
                    string2accs.setdefault(value, set()).add(acc)
                elif kind == "Ensembl":
                    self.ensg2accs.setdefault(value.split(".", 1)[0], []).append(acc)
        self.ensp2genes = {}
        with gzip.open(STRING_ALIAS, "rt") as f:
            for ensp, kind, value in csv.reader(f, delimiter="\t"):
                if kind == "entrez":
                    self.ensp2genes.setdefault(ensp, set()).add(f"g{value}")
                else:
                    string2accs.setdefault(ensp, set()).add(value)
        # Fold both directions together: an accession gains its STRING gene,
        # a STRING protein gains the genes of all its accessions. A STRING
        # protein with no Entrez gene stands for itself.
        for ensp, accs in string2accs.items():
            genes = set(self.ensp2genes.get(ensp, ()))
            for acc in accs:
                genes |= self.acc2genes.get(acc, set())
            genes = genes or {ensp}
            self.ensp2genes[ensp] = genes
            for acc in accs:
                self.acc2genes.setdefault(acc, set()).update(genes)

    def acc_nodes(self, acc):
        acc = strip_isoform(acc)
        return self.acc2genes.get(acc) or {acc}

    def ensp_nodes(self, ensp):
        return self.ensp2genes.get(ensp) or {ensp}


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
