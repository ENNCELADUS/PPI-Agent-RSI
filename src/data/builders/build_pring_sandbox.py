"""Build the PRING dev sandbox: a closed-world, internal reference KB and labelled splits for debugging the agent.

Run: uv run python -m src.data.builders.build_pring_sandbox
Inputs in data/PRING/raw/: PRING human BFS split as packaged by the motif-graph project
(PRING commit edbdc01, node ids rewritten to node_*), node_sequences.tsv (node -> UniProt,
sequence; from H20), bulk all-human UniProt name and annotation tables, and the GO ontology. Fetch:

    M=~/Documents/motif-graph-prompt-tuning-for-inductive-protein-protein-interaction-prediction/data/benchmark_2025_neurips
    mkdir -p data/PRING/raw/breadth_first && cp $M/positive_edges.txt data/PRING/raw/
    cp $M/breadth_first/{train,val,test}_edges.txt data/PRING/raw/breadth_first/
    H20=/2023533015/topology-conditioned-inductive-edge-prediction/data/official_sequence_baselines
    scp -P 30838 root@10.15.171.204:$H20/node_sequences.tsv data/PRING/raw/
    curl -C - --retry 5 -o data/PRING/raw/uniprot_human_names.tsv \
      'https://rest.uniprot.org/uniprotkb/stream?query=organism_id:9606&fields=accession,gene_primary,protein_name&format=tsv'
    curl -C - --retry 5 -o data/PRING/raw/uniprot_human_annotation.tsv.gz \
      'https://rest.uniprot.org/uniprotkb/stream?query=organism_id:9606&fields=accession,cc_function,cc_subcellular_location,cc_tissue_specificity,go_p,go_c,go_f,keyword,protein_families,xref_pfam,xref_interpro&format=tsv&compressed=true'
    curl -L -C - --retry 5 -o data/PRING/raw/go-basic.obo https://purl.obolibrary.org/obo/go/go-basic.obo

Tested negatives come from data/TMP/public_db/negative/ (all human Negatome 2.0 and IntAct negative
pairs as raw accessions; `fetch_public.sh negatome intact_negative`).

Sandbox numbers are never reported: every label here is public, so a web-enabled agent would
look it up.

- Self pairs are dropped everywhere (PRING keeps them as positives; the private split drops them).
- Tested negatives with both ends on the KB side of PRING's protein split join train; others and
  PRING positives are skipped. They are always kept; train and val are then filled to 1:1 with a
  seeded sample of PRING's own negatives.
- hidden/{train,val}.csv hold the balanced labelled splits (val is the predictor's validation set);
  kb.csv = train + val, `evidence` naming each row's source. Columns follow the private split
  (uniprotID_A, uniprotID_B, isInteraction).
- Wave discovery (docs/superpowers/specs/2026-10-23-pring-wave-discovery-design.md): the candidates
  are all pairs among the test proteins (`candidate` = 1 in proteins.tsv; none is a KB protein);
  hidden/test_positives.csv lists every PRING positive among them, the oracle's hits.
- annotations.tsv: everything UniProt says about each protein except its interactions (user
  decision 2026-10-23: all information but the label). Removed as the label in another form: any
  text sentence naming another human gene symbol, GO terms under protein binding or
  protein-containing complex. Interaction and subunit comments are never fetched.
"""

import csv
import gzip
import hashlib
import random
import re
from collections import defaultdict
from datetime import date

from src.data.paths import HIDDEN, PRING, PUBLIC_NEG, SANDBOX
from src.data.paths import PRING_RAW as RAW

SPLITS = ("train", "val", "test")
TESTED = ("negatome", "intact_negative")
INPUTS = (
    "positive_edges.txt",
    *(f"breadth_first/{s}_edges.txt" for s in SPLITS),
    "node_sequences.tsv",
    "uniprot_human_names.tsv",
    "uniprot_human_annotation.tsv.gz",
    "go-basic.obo",
)
SEED = 0
PAIR = ["uniprotID_A", "uniprotID_B"]
LEAKY_GO = ("GO:0005515", "GO:0032991")  # protein binding, protein-containing complex
TEXT = ("function", "subcellular_location", "tissue_specificity")
GO = ("go_process", "go_component", "go_function")
ANNOTATION = (*TEXT, *GO, "keywords", "families", "pfam", "interpro")
NOISE = re.compile(r"\s*\{ECO:[^}]*\}|\s*\(PubMed:[^)]*\)|(FUNCTION|SUBCELLULAR LOCATION|TISSUE SPECIFICITY): ")
SENTENCE = re.compile(r"(?<=[.;])\s+")
TOKEN = re.compile(r"[A-Za-z0-9-]+")
GO_TERM = re.compile(r"\[(GO:\d{7})\]")


def edges(path):
    """Unordered non-self node pairs -> label (files without a label column are all positive)."""
    out = {}
    for row in csv.reader(open(path), delimiter="\t"):
        if row[0] == row[1]:
            continue
        pair, label = tuple(sorted(row[:2])), int(row[2]) if len(row) > 2 else 1
        assert out.setdefault(pair, label) == label, f"conflicting labels for {pair}"
    return out


def balance(pairs, keep, rng):
    """All positives, every negative in `keep`, and a sample of the other negatives up to 1:1."""
    pos = sorted(p for p, y in pairs.items() if y)
    kept = sorted(p for p, y in pairs.items() if not y and p in keep)
    rest = sorted(p for p, y in pairs.items() if not y and p not in keep)
    assert len(kept) <= len(pos) <= len(kept) + len(rest), "cannot balance"
    return {p: 1 for p in pos} | {p: 0 for p in kept + rng.sample(rest, len(pos) - len(kept))}


def go_descendants(path, roots):
    """Every GO term under `roots` by is_a, roots included."""
    children, term = defaultdict(set), None
    for line in open(path):
        if line.startswith("["):
            term = None
        elif line.startswith("id: GO:"):
            term = line[4:].strip()
        elif line.startswith("is_a: GO:") and term:
            children[line[6:16]].add(term)
    out, todo = set(), list(roots)
    while todo:
        if (t := todo.pop()) not in out:
            out.add(t)
            todo.extend(children[t])
    return out


def scrub(text, symbols, own):
    """Text minus every sentence naming a gene symbol not in `own` (it can state a partner), and the drop count."""
    sentences = [x for x in SENTENCE.split(NOISE.sub("", text).strip()) if x]
    kept = [x for x in sentences if not (set(TOKEN.findall(x)) & symbols) - own]
    return " ".join(kept), len(sentences) - len(kept)


def write(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t" if path.suffix == ".tsv" else ",")
        w.writerow(header)
        w.writerows(rows)


def main():
    node2acc, seq = {}, {}
    for r in csv.DictReader(open(RAW / "node_sequences.tsv"), delimiter="\t"):
        node2acc[r["node_id"]], seq[r["node_id"]] = r["uniprot_id"], r["sequence"]
    assert len(set(node2acc.values())) == len(node2acc), "two nodes share an accession"
    acc2node = {a: n for n, a in node2acc.items()}
    names = {r[0]: r[1:] for r in csv.reader(open(RAW / "uniprot_human_names.tsv"), delimiter="\t")}

    known = set(edges(RAW / "positive_edges.txt"))  # every PRING positive, any split
    splits = {s: edges(RAW / f"breadth_first/{s}_edges.txt") for s in SPLITS}
    side = {s: {x for p in splits[s] for x in p} for s in SPLITS}
    kb_side = side["train"] | side["val"]
    assert not kb_side & side["test"], "test protein on the KB side"

    tested = {}  # node pair -> source
    for source in TESTED:
        for r in csv.DictReader(gzip.open(PUBLIC_NEG / f"{source}.tsv.gz", "rt"), delimiter="\t"):
            a, b = acc2node.get(r["a"]), acc2node.get(r["b"])
            if a and b and (p := tuple(sorted((a, b)))) not in known:
                tested.setdefault(p, source)
    for p in tested:
        if set(p) <= kb_side and p not in splits["val"]:
            splits["train"][p] = 0
    hits = sorted(p for p, y in splits.pop("test").items() if y)
    splits = {s: balance(pairs, tested, random.Random(f"{SEED}-{s}")) for s, pairs in splits.items()}
    for s, pairs in splits.items():
        assert 2 * sum(pairs.values()) == len(pairs), f"{s} not 1:1"
        assert not any(p in known for p, y in pairs.items() if not y), f"{s} negative is a PRING positive"
    assert not splits["train"].keys() & splits["val"].keys(), "pair in train and val"

    kb = splits["train"] | splits["val"]
    kb_proteins = {x for p in kb for x in p}
    candidates = side["test"]
    assert not kb_proteins & candidates, "test protein in KB"
    assert set(hits) == {p for p in known if set(p) <= candidates}, "PRING positive among test proteins missing"
    proteins = sorted(kb_proteins | candidates)

    leaky = go_descendants(RAW / "go-basic.obo", LEAKY_GO)
    genes = {a: set(TOKEN.findall(g)) for a, (g, _) in names.items()}
    symbols = {g for gs in genes.values() for g in gs if len(g) >= 3 and not g.islower()}
    wanted = {node2acc[x] for x in proteins}
    annotated, n_go_dropped = {}, 0
    for r in csv.reader(gzip.open(RAW / "uniprot_human_annotation.tsv.gz", "rt"), delimiter="\t"):
        if r[0] not in wanted:
            continue
        go = []
        for field in r[4:7]:
            terms = [t for t in field.split("; ") if t and GO_TERM.search(t)[1] not in leaky]
            n_go_dropped += len([t for t in field.split("; ") if t]) - len(terms)
            go.append("; ".join(terms))
        annotated[r[0]] = (r[1:4], go, r[7:])
    n_dropped = 0

    def annotation_row(acc):
        nonlocal n_dropped
        text, go, rest = annotated.get(acc, (("",) * 3, ("",) * 3, ("",) * 4))
        text = [scrub(t, symbols, genes.get(acc, set())) for t in text]
        n_dropped += sum(n for _, n in text)
        return (*(t for t, _ in text), *go, *rest)

    acc = node2acc
    write(
        SANDBOX / "proteins.tsv",
        ["accession", "gene", "protein_name", "candidate", "sequence"],
        [(acc[x], *names.get(acc[x], ("", "")), int(x in candidates), seq[x]) for x in proteins],
    )
    write(
        SANDBOX / "annotations.tsv",
        ["accession", *ANNOTATION],
        sorted((acc[x], *annotation_row(acc[x])) for x in proteins),
    )
    write(
        SANDBOX / "kb.csv",
        PAIR + ["isInteraction", "evidence"],
        sorted((acc[a], acc[b], y, tested.get((a, b), "pring")) for (a, b), y in kb.items()),
    )
    for s, pairs in splits.items():
        write(
            HIDDEN / f"{s}.csv",
            PAIR + ["isInteraction"],
            sorted((node2acc[a], node2acc[b], y) for (a, b), y in pairs.items()),
        )
    write(HIDDEN / "test_positives.csv", PAIR, sorted((node2acc[a], node2acc[b]) for a, b in hits))

    stats = {
        **{
            f"{s}_{k}": v
            for s, pairs in splits.items()
            for k, v in (
                ("pos", sum(pairs.values())),
                ("neg_tested", sum(p in tested for p, y in pairs.items() if not y)),
                ("neg_pring", sum(p not in tested for p, y in pairs.items() if not y)),
                ("proteins", len({x for p in pairs for x in p})),
            )
        },
        "kb_proteins": len(kb_proteins),
        "candidate_proteins": len(candidates),
        "candidate_pairs": len(candidates) * (len(candidates) - 1) // 2,
        "candidate_hits": len(hits),
        "proteins_without_uniprot_entry": sum(node2acc[x] not in names for x in proteins),
        "proteins_annotated": len(annotated),
        "annotation_sentences_dropped": n_dropped,
        "go_terms_dropped": n_go_dropped,
        "self_pairs_dropped": sum(
            1
            for s in SPLITS
            for r in csv.reader(open(RAW / f"breadth_first/{s}_edges.txt"), delimiter="\t")
            if r[0] == r[1]
        ),
    }
    checksums = {f"raw/{f}": hashlib.sha256((RAW / f).read_bytes()).hexdigest()[:16] for f in INPUTS}
    checksums |= {
        f"TMP/public_db/negative/{s}.tsv.gz": hashlib.sha256((PUBLIC_NEG / f"{s}.tsv.gz").read_bytes()).hexdigest()[:16]
        for s in TESTED
    }
    readme = [
        "# PRING dev sandbox (generated by src/data/builders/build_pring_sandbox.py)",
        "",
        f"Built {date.today()}, seed {SEED}. Internal dev sandbox; never report these numbers.",
        "Give the agent `sandbox/`, never `hidden/`.",
        "",
        "| Item | Value |",
        "|---|---|",
        *(f"| {k} | {v:,} |" for k, v in stats.items()),
        "",
        "Inputs (sha256 prefix): PRING human BFS split at commit edbdc01 via the motif-graph package;",
        "node_sequences.tsv from H20; UniProt names from rest.uniprot.org organism_id:9606 stream;",
        "tested negatives from Negatome 2.0 and IntAct negative.",
        "",
        *(f"- `{f}` {c}" for f, c in checksums.items()),
        "",
    ]
    (PRING / "README.md").write_text("\n".join(readme))
    print(stats)


if __name__ == "__main__":
    main()
