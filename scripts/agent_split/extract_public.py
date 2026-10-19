"""Stream one public PPI source from stdin and keep pairs among TMP proteins.

Usage:  curl -sL <url> | python extract_public.py <source>
Writes data/TMP/public_db/<source>.tsv.gz with columns: a, b, score, detail
(a < b, gene-level nodes; score is the source's confidence, empty if none).
The `idmap` source instead writes the compact UniProt mapping used by all others.
"""
import csv
import gzip
import re
import sys

from common import IDMAP, PUBLIC, load_idmap, node_of, raw_universe, strip_isoform

csv.field_size_limit(sys.maxsize)
UNIPROT = re.compile(r"uniprotkb:([A-Z0-9]+(?:-\d+)?)")
ENTREZ = re.compile(r"entrez gene(?:/locuslink)?:(\d+)")


def write_idmap(lines):
    keep = {"GeneID", "STRING", "Ensembl"}
    n = 0
    with gzip.open(IDMAP, "wt") as out:
        for line in lines:
            acc, kind, value = line.rstrip("\n").split("\t")
            if kind in keep:
                out.write(f"{acc}\t{kind}\t{value}\n")
                n += 1
    print(f"idmap rows kept: {n}", file=sys.stderr)


def string_pairs(lines, maps):
    _, string2acc, _ = maps
    next(lines)
    for line in lines:
        a, b, score = line.split()
        if a in string2acc and b in string2acc:
            yield string2acc[a], string2acc[b], int(score), ""


def biogrid_pairs(lines, maps):
    for row in csv.reader(lines, delimiter="\t"):
        if row[0].startswith("#") or row[15] != "9606" or row[16] != "9606":
            continue
        # Columns 1/2 are Entrez IDs; prefix marks them as gene nodes already.
        yield f"g{row[1]}", f"g{row[2]}", "", row[12]


def mitab_pairs(lines, score_key):
    """IntAct-style MITAB: UniProt in cols 0-3, taxids 9/10, scores col 14."""
    for row in csv.reader(lines, delimiter="\t"):
        if row[0].startswith("#") or "taxid:9606" not in row[9] or "taxid:9606" not in row[10]:
            continue
        ida = UNIPROT.search(row[0] + "|" + row[2])
        idb = UNIPROT.search(row[1] + "|" + row[3])
        if not (ida and idb):
            continue
        score = re.search(score_key + r":([\d.]+)", row[14])
        yield ida.group(1), idb.group(1), float(score.group(1)) if score else "", row[11]


def hippie_pairs(lines, maps):
    next(lines)
    for row in csv.reader(lines, delimiter="\t"):
        a, b = ENTREZ.search(row[0]), ENTREZ.search(row[1])
        if a and b:
            yield f"g{a.group(1)}", f"g{b.group(1)}", float(row[14]), row[12]


def huri_pairs(lines, maps):
    _, _, ensg2acc = maps
    for line in lines:
        a, b = line.split()[:2]
        for x in ensg2acc.get(a, []):
            for y in ensg2acc.get(b, []):
                yield x, y, "", ""


def bioplex_pairs(lines, maps):
    next(lines)
    for row in csv.reader(lines, delimiter="\t"):
        # GeneA, GeneB, UniprotA, UniprotB, SymbolA, SymbolB, pW, pNI, pInt
        yield f"g{row[0]}", f"g{row[1]}", float(row[8]), ""


def corum_pairs(lines, maps):
    rows = csv.DictReader(lines, delimiter="\t")
    entrez_col = next(c for c in rows.fieldnames if "entrez" in c.lower())
    for row in rows:
        genes = sorted({g.strip() for g in row[entrez_col].split(";") if g.strip().isdigit()})
        for i, a in enumerate(genes):
            for b in genes[i + 1:]:
                yield f"g{a}", f"g{b}", "", row.get("complex_name", "")


SOURCES = {
    "string": string_pairs,
    "biogrid": biogrid_pairs,
    "intact": lambda lines, maps: mitab_pairs(lines, "intact-miscore"),
    "hippie": hippie_pairs,
    "huri": huri_pairs,
    "bioplex": bioplex_pairs,
    "corum": corum_pairs,
}


def to_node(x, acc2gene):
    return x if x.startswith("g") else node_of(strip_isoform(x), acc2gene)


def main(source):
    lines = (line for line in sys.stdin)
    if source == "idmap":
        return write_idmap(lines)
    maps = load_idmap()
    acc2gene = maps[0]
    universe = {node_of(a, acc2gene) for a in raw_universe()}
    best = {}
    seen = 0
    for a, b, score, detail in SOURCES[source](lines, maps):
        seen += 1
        a, b = to_node(a, acc2gene), to_node(b, acc2gene)
        if a == b or a not in universe or b not in universe:
            continue
        key = (a, b) if a < b else (b, a)
        old = best.get(key)
        if old is None or (score != "" and (old[0] == "" or score > old[0])):
            best[key] = (score, detail)
    with gzip.open(PUBLIC / f"{source}.tsv.gz", "wt") as out:
        out.write("a\tb\tscore\tdetail\n")
        for (a, b), (score, detail) in best.items():
            out.write(f"{a}\t{b}\t{score}\t{detail}\n")
    print(f"{source}: {seen} source rows -> {len(best)} pairs within TMP universe", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1])
