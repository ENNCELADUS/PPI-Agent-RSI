"""Filter one public PPI source (text on stdin) to pairs among TMP proteins.

Usage:  <decoded source> | python extract_public.py <source>
Writes data/TMP/public_db/<source>.tsv.gz with columns: a, b, score, detail
(a < b, gene-level nodes; score is the source's confidence, empty if none).
The `idmap` and `string_alias` sources write the mapping tables (*.map.gz)
that every other source relies on, so run them first. The `negatome` and
`intact_negative` sources write experimentally tested non-interactions to
public_db/negative/<source>.tsv.gz as raw accession pairs.

Sources yield raw IDs: "g<entrez>" for genes, "9606.ENSP…" for STRING
proteins, anything else is a UniProt accession.
"""
import csv
import gzip
import itertools
import re
import sys

from common import IDMAP, PUBLIC, PUBLIC_NEG, STRING_ALIAS, IdMap, raw_universe, strip_isoform

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


def write_string_alias(lines):
    kinds = {"Ensembl_HGNC_entrez_id": "entrez", "UniProt_AC": "acc", "Ensembl_UniProt": "acc"}
    n = 0
    with gzip.open(STRING_ALIAS, "wt") as out:
        for line in lines:
            ensp, alias, source = line.rstrip("\n").split("\t")
            if source in kinds:
                out.write(f"{ensp}\t{kinds[source]}\t{alias}\n")
                n += 1
    print(f"string alias rows kept: {n}", file=sys.stderr)


def string_pairs(lines, idmap):
    next(lines)
    for line in lines:
        a, b, score = line.split()
        yield a, b, int(score), ""


def biogrid_pairs(lines, idmap):
    for row in csv.reader(lines, delimiter="\t"):
        if row[0].startswith("#") or row[15] != "9606" or row[16] != "9606":
            continue
        yield f"g{row[1]}", f"g{row[2]}", "", row[12]


def intact_pairs(lines, idmap):
    """MITAB: UniProt in cols 0-3, taxids 9/10, intact-miscore in col 14."""
    for row in csv.reader(lines, delimiter="\t"):
        if row[0].startswith("#") or "taxid:9606" not in row[9] or "taxid:9606" not in row[10]:
            continue
        ida = UNIPROT.search(row[0] + "|" + row[2])
        idb = UNIPROT.search(row[1] + "|" + row[3])
        if not (ida and idb):
            continue
        score = re.search(r"intact-miscore:([\d.]+)", row[14])
        yield ida.group(1), idb.group(1), float(score.group(1)) if score else "", row[11]


def hippie_pairs(lines, idmap):
    next(lines)
    for row in csv.reader(lines, delimiter="\t"):
        a, b = ENTREZ.search(row[0]), ENTREZ.search(row[1])
        if a and b:
            yield f"g{a.group(1)}", f"g{b.group(1)}", float(row[14]), row[12]


def huri_pairs(lines, idmap):
    for line in lines:
        a, b = line.split()[:2]
        for x in idmap.ensg2accs.get(a, []):
            for y in idmap.ensg2accs.get(b, []):
                yield x, y, "", ""


def bioplex_pairs(lines, idmap):
    next(lines)
    for row in csv.reader(lines, delimiter="\t"):
        # GeneA, GeneB, UniprotA, UniprotB, SymbolA, SymbolB, pW, pNI, pInt
        yield f"g{row[0]}", f"g{row[1]}", float(row[8]), ""


def corum_pairs(lines, idmap):
    for row in csv.DictReader(lines, delimiter="\t"):
        accs = sorted({a.strip() for a in row["subunits_uniprot_id"].split(";") if a.strip()})
        for i, a in enumerate(accs):
            for b in accs[i + 1:]:
                yield a, b, "", row["complex_name"]


def negatome_negatives(lines):
    # manual_stringent has 4 columns, combined_stringent 2; both start with two accessions.
    for line in lines:
        fields = line.rstrip("\n").split("\t")
        if len(fields) >= 2:
            yield fields[0].strip(), fields[1].strip()


def intact_negatives(lines):
    for row in csv.reader(lines, delimiter="\t"):
        if row[0].startswith("#") or row[35] != "true":
            continue
        if "taxid:9606" not in row[9] or "taxid:9606" not in row[10]:
            continue
        ida, idb = UNIPROT.search(row[0]), UNIPROT.search(row[1])
        if ida and idb:
            yield ida.group(1), idb.group(1)


NEG_SOURCES = {
    "negatome": negatome_negatives,
    "intact_negative": intact_negatives,
}


def write_negatives(source, lines):
    """Experimentally tested non-interactions, kept as raw accession pairs."""
    PUBLIC_NEG.mkdir(exist_ok=True)
    pairs = {tuple(sorted((strip_isoform(a), strip_isoform(b)))) for a, b in NEG_SOURCES[source](lines)}
    pairs = sorted(p for p in pairs if p[0] != p[1])
    with gzip.open(PUBLIC_NEG / f"{source}.tsv.gz", "wt") as out:
        out.write("a\tb\n")
        out.writelines(f"{a}\t{b}\n" for a, b in pairs)
    print(f"{source}: {len(pairs)} negative pairs", file=sys.stderr)


SOURCES = {
    "string": string_pairs,
    "biogrid": biogrid_pairs,
    "intact": intact_pairs,
    "hippie": hippie_pairs,
    "huri": huri_pairs,
    "bioplex": bioplex_pairs,
    "corum": corum_pairs,
}


def nodes_of(raw_id, idmap):
    if raw_id.startswith("g"):
        return {raw_id}
    if raw_id.startswith("9606."):
        return idmap.ensp_nodes(raw_id)
    return idmap.acc_nodes(raw_id)


def main(source):
    lines = (line for line in sys.stdin)
    if source == "idmap":
        return write_idmap(lines)
    if source == "string_alias":
        return write_string_alias(lines)
    if source in NEG_SOURCES:
        return write_negatives(source, lines)
    idmap = IdMap()
    universe = set().union(*(idmap.acc_nodes(a) for a in raw_universe()))
    cache = {}
    best = {}
    seen = 0
    for a, b, score, detail in SOURCES[source](lines, idmap):
        seen += 1
        for raw_id in (a, b):
            if raw_id not in cache:
                cache[raw_id] = nodes_of(raw_id, idmap) & universe
        for x, y in itertools.product(cache[a], cache[b]):
            if x == y:
                continue
            key = (x, y) if x < y else (y, x)
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
