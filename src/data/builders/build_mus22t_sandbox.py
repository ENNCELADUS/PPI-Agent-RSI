"""Build the 22T agent sandbox: the agent-visible files for the 22T mouse membrane split, PRING-sandbox layout.

Run: uv run python -m src.data.builders.build_mus22t_sandbox   (after build_mus22t_agent_split; needs mmseqs)
Inputs: data/MUS22T/agent_split/ (hidden labels), data/MUS22T/raw/raw_22T_node_sequences.csv, mouse UniProt tables
and the GO ontology (data/PRING/raw/go-basic.obo, shared with the PRING sandbox). Fetch the UniProt tables:

    curl -C - --retry 5 -o data/MUS22T/raw/uniprot_mouse_names.tsv \
      'https://rest.uniprot.org/uniprotkb/stream?query=organism_id:10090&fields=accession,gene_primary,protein_name&format=tsv'
    curl -C - --retry 5 -o data/MUS22T/raw/uniprot_mouse_annotation.tsv.gz \
      'https://rest.uniprot.org/uniprotkb/stream?query=organism_id:10090&fields=accession,cc_function,cc_subcellular_location,cc_tissue_specificity,go_p,go_c,go_f,keyword,protein_families,xref_pfam,xref_interpro&format=tsv&compressed=true'

data/MUS22T/sandbox/ (what the agent may read; no web tools, user decision 2026-11-02):
- kb.csv: agent_split/train.csv; `evidence` = 22t (positive) or pring_sampled (negative). val and test stay hidden.
- proteins.tsv: every protein of a train or test pair; `candidate` = 1 if it is in a test pair (transductive: most
  candidates are KB proteins too). Val-only proteins are left out.
- annotations.tsv: as the PRING sandbox (all UniProt information but interactions): text sentences naming another
  mouse gene symbol are dropped, matched case-insensitively because mouse entries often name partners in human
  upper case; GO terms under protein binding or protein-containing complex are dropped.
- homologs.tsv: best 20 non-self mmseqs2 hits of every protein against the KB proteins (build_pring_tools.homologs).
No predictor scores. data/MUS22T/README.md records the counts and input checksums.
"""

import csv
import gzip
import hashlib
from datetime import date

from src.data.builders.build_mus22t_agent_split import OUT as SPLIT
from src.data.builders.build_pring_sandbox import (
    ANNOTATION,
    GO_TERM,
    LEAKY_GO,
    NOISE,
    PAIR,
    SENTENCE,
    TOKEN,
    go_descendants,
    write,
)
from src.data.builders.build_pring_tools import homologs
from src.data.paths import MUS22T as MUS
from src.data.paths import PRING_RAW

RAW = MUS / "raw"
SANDBOX = MUS / "sandbox"
GO_OBO = PRING_RAW / "go-basic.obo"


def read(split):
    return [(r["uniprotID_A"], r["uniprotID_B"], int(r["isInteraction"])) for r in csv.DictReader(open(SPLIT / split))]


def scrub(text, symbols, own):
    """Text minus every sentence naming (any case) a gene symbol not in `own`, and the drop count."""
    sentences = [x for x in SENTENCE.split(NOISE.sub("", text).strip()) if x]
    named = lambda x: {t.upper() for t in TOKEN.findall(x) if not t.islower()}  # noqa: E731
    kept = [x for x in sentences if not (named(x) & symbols) - own]
    return " ".join(kept), len(sentences) - len(kept)


def main():
    seq = {r["id"]: r["sequence"] for r in csv.DictReader(open(RAW / "raw_22T_node_sequences.csv"))}
    train, test = read("train.csv"), read("test.csv")
    kb_proteins = {x for a, b, _ in train for x in (a, b)}
    candidates = {x for a, b, _ in test for x in (a, b)}
    proteins = sorted(kb_proteins | candidates)
    assert set(proteins) <= seq.keys(), "protein without sequence"

    names = {r[0]: r[1:] for r in csv.reader(open(RAW / "uniprot_mouse_names.tsv"), delimiter="\t")}
    genes = {a: {g.upper() for g in TOKEN.findall(g)} for a, (g, _) in names.items()}
    symbols = {g.upper() for a, (gs, _) in names.items() for g in TOKEN.findall(gs) if len(g) >= 3 and not g.islower()}
    leaky = go_descendants(GO_OBO, LEAKY_GO)
    wanted, annotated, go_dropped, sentences_dropped = set(proteins), {}, 0, 0
    for r in csv.reader(gzip.open(RAW / "uniprot_mouse_annotation.tsv.gz", "rt"), delimiter="\t"):
        if r[0] not in wanted:
            continue
        go = []
        for field in r[4:7]:
            terms = [t for t in field.split("; ") if t]
            kept = [t for t in terms if GO_TERM.search(t)[1] not in leaky]
            go_dropped += len(terms) - len(kept)
            go.append("; ".join(kept))
        text = [scrub(t, symbols, genes.get(r[0], set())) for t in r[1:4]]
        sentences_dropped += sum(n for _, n in text)
        annotated[r[0]] = (*(t for t, _ in text), *go, *r[7:])
    empty = ("",) * len(ANNOTATION)

    write(
        SANDBOX / "proteins.tsv",
        ["accession", "gene", "protein_name", "candidate", "sequence"],
        [(x, *names.get(x, ("", "")), int(x in candidates), seq[x]) for x in proteins],
    )
    write(SANDBOX / "annotations.tsv", ["accession", *ANNOTATION], [(x, *annotated.get(x, empty)) for x in proteins])
    write(
        SANDBOX / "kb.csv",
        PAIR + ["isInteraction", "evidence"],
        sorted((a, b, y, "22t" if y else "pring_sampled") for a, b, y in train),
    )
    write(
        SANDBOX / "homologs.tsv",
        ["query", "target", "pident", "evalue", "bits"],
        homologs({x: seq[x] for x in proteins}, kb_proteins),
    )

    stats = {
        "kb_pairs": len(train),
        "kb_proteins": len(kb_proteins),
        "candidate_proteins": len(candidates),
        "candidates_also_in_kb": len(candidates & kb_proteins),
        "test_pairs": len(test),
        "proteins": len(proteins),
        "proteins_without_uniprot_entry": sum(x not in names for x in proteins),
        "proteins_annotated": len(annotated),
        "annotation_sentences_dropped": sentences_dropped,
        "go_terms_dropped": go_dropped,
    }
    inputs = [SPLIT / f"{s}.csv" for s in ("train", "val", "test")] + [
        RAW / "raw_22T_node_sequences.csv",
        RAW / "uniprot_mouse_names.tsv",
        RAW / "uniprot_mouse_annotation.tsv.gz",
        GO_OBO,
    ]
    readme = [
        "# 22T agent sandbox (generated by src/data/builders/build_mus22t_sandbox.py)",
        "",
        f"Built {date.today()}. Give the agent `sandbox/`, never `agent_split/` (val and test labels).",
        "Private mouse data: never commit, upload or send it anywhere except the configured LLM.",
        "",
        "| Item | Value |",
        "|---|---|",
        *(f"| {k} | {v:,} |" for k, v in stats.items()),
        "",
        "Inputs (sha256 prefix):",
        "",
        *(f"- `{p.relative_to(MUS.parent)}` {hashlib.sha256(p.read_bytes()).hexdigest()[:16]}" for p in inputs),
        "",
    ]
    (MUS / "README.md").write_text("\n".join(readme))
    print(stats)


if __name__ == "__main__":
    main()
