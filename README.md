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
| `docs/superpowers/specs/` | Dated design documents |
| `src/` | The code, one package; module map in [`src/README.md`](src/README.md) |
| `tests/` | pytest suite mirroring `src/` |
| `data/` | Private TMP data, git-ignored; never shared outside the configured LLM |

## Benchmarks

- **Private (built):** human transmembrane PPIs; test positives have no evidence in any public PPI
  database, with 1:1 degree-preserving negatives. See `CLAUDE.md` for the build commands.
- **PRING dev sandbox (built):** an internal, closed-world benchmark built from PRING human data, for
  debugging the agent before it sees the private test set. Its 1:1 train and val splits form the agent's
  reference KB; the test proteins are the wave-discovery candidates. An anonymised-ID arm
  measures memorisation. Its numbers are never reported. See the [design](docs/superpowers/specs/2026-10-23-pring-sandbox-design.md).
- **Wave discovery benchmark (designed, not built):** the agent searches all pairs among the sandbox's test
  proteins in 10 waves of 200 picks, seeing only the hits among its own picks, against no-feedback and
  permuted-feedback arms. See the [design](docs/superpowers/specs/2026-10-23-pring-wave-discovery-design.md).
- **Other public tracks (surveyed):** [2026-10-19 survey](docs/research-reports/2026-10-19-public-ppi-benchmarks.md):
  BioDiscoveryAgent screens, SKEMPI 2.0 via `ppi-mutation-ddg`, and a post-cutoff IntAct/BioGRID slice.
  All are run closed-world, because their labels are retrievable online.

## Setup

```bash
uv sync
```
