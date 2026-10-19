# PPI-Agent-RSI

LLM scientific agents for membrane protein–protein interaction (PPI) discovery, and recursive
self-improvement of the agent workflow under sparse experimental feedback. Current phase: measure a
fixed, expert-designed agent baseline before building any RSI.

## Layout

| Path | Contents |
| --- | --- |
| `docs/introduction.md` | Motivation and the three research directions |
| `docs/initial_baseline.md` | Baseline agent design and evaluation protocol |
| `docs/literature.md` | Related work, with links to local PDFs |
| `docs/research-reports/` | Dated deep-research reports; `research_notes/` holds the notes behind each |
| `literature/` | PDFs by topic (`ppi`, `science-agents`, `self-improvement`, `feedback-benchmarks`); index in `literature/README.md` |
| `scripts/agent_split/` | Builds the private benchmark split in `data/TMP/agent_split/` |
| `data/` | Private TMP data, git-ignored; never shared outside the configured LLM |

## Benchmarks

- **Private (built):** human transmembrane PPIs; test positives have no evidence in any public PPI
  database, with 1:1 degree-preserving negatives. See `CLAUDE.md` for the build commands.
- **Public (surveyed):** [2026-10-19 survey](docs/research-reports/2026-10-19-public-ppi-benchmarks.md).
  The shortlist is PRING, BioDiscoveryAgent screens, SKEMPI 2.0 via `ppi-mutation-ddg`, and a
  post-cutoff IntAct/BioGRID slice. All are run closed-world, because their labels are retrievable online.

## Setup

```bash
uv sync
```
