---
name: results-report
description: How to report experiment results in this repo, in a chat reply, or the human-readable results/<run>/summary.html page. Use whenever you present numbers from a run, write or revise a results record or summary page, make result figures, or summarise an analysis report, even if the user only asks "how did it go" or "what are the results".
---

# Results report

Present results as an ML paper would. Each table has a caption that names the metric, the benchmark and split, and
whether higher is better. It has one row per named model, baseline or control (controls sit in the same table), one
column per benchmark, screen, held-out line or replicate plus a mean column, and the best value in bold. Explain the
metric once in one sentence. Give validation and test as separate tables or columns; validation only selects, test is
reported once. Contrasts with bootstrap intervals go in a second table. Never use bullet dumps or arm × scorer pivots
as the main presentation, and name every row by what it is, not by an internal label.

Describe a discovery campaign as the literature does, in one set of terms: a **round** is one pick → test → feedback
cycle, the **batch** is the pairs picked per round, a **run** is one independent campaign
(a "replicate" in the code). Give budgets as runs × rounds × batch, e.g. "4 runs × 7 rounds × 200 pairs".

Report in chat by default: one line on what ran (model, rounds, replicates, code SHA), the result tables, then a
short **Reading** section of bold one-line findings, each backed by a table. Provenance stays in the run's own
`run.json` and `result.json` files. Never write a `README.md` (or any other Markdown record) for a run, an
analysis or a results folder, anywhere.

## Summary page (only when asked)

Write an HTML page only when the user asks for one:

- **Location:** `results/<run>/` holds only `summary.html` and `figures/*.svg`; no README, index or copied reports.
  `results/` is git-ignored and the GitHub repo is public, so sandbox numbers never leave the machine.
- **Build:** load the `artifact-design` skill for its design rules, then write `summary.html` as a local,
  self-contained file. Never publish it with the Artifact tool or upload it anywhere. Because nothing wraps it, the
  file carries its own `<!doctype html>`, charset and viewport meta, `<title>` and `<style>`; colours are tokens with a
  dark-mode block. Check it once with a headless-Chrome screenshot, fix what it shows, and stop.
- **Story:** follow the reader's order, not the code's: the question; the benchmark (dataset, splits with protein
  and PPI counts, the candidate space); the discovery loop (what one round does and how many rounds ran); the agent
  (model, tools); results (the tables above, then figures); **Reading**. Numbers in the page come from the run's
  `results.json` or `result.json` files, never retyped from memory.
- **Figures:** vector only. Analysis code saves `.svg` (matplotlib with `svg.fonttype: none`); if only PDFs exist,
  convert with `pdftocairo -svg`. Insert with `<img src="figures/<name>.svg">` and a caption. The same rule holds
  for figures in PDF reports: SVG or PDF, never PNG.
- **Scope:** show only what the run is reporting now. Archived or dropped arms stay out of the page.
