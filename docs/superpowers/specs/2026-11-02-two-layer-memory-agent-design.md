# Two-layer memory agent (first version) — design

**Status:** 2026-11-02, decisions grilled with the user in conversation. Built the same day: split builder
`src/data/builders/build_pring_memory.py`, memory layers `src/agent/memory.py`, tools in `src/agent/tools.py`
(`CaseRoundTools`, `ReviewTools`, `TestTools`), prompts `src/agent/prompts/memory/`, runner `src/eval/memory_agent.py`. Extends the closed-loop
discovery benchmark ([design](2026-10-23-pring-wave-discovery-design.md)) and reuses the classification-test
protocol ([design](2026-11-01-pring-classification-test-design.md)). Calls marked *(my call)* are the
designer's defaults, not the user's decisions.

## Why
The discovery runs measure hits per round, but they have no held-out test: nothing shows whether what the
agent learns from feedback transfers to pairs it never selected. This version adds a fixed final test set and
a memory that the agent can carry to that test set: tested cases and lessons distilled from them.

## Fixed during a run
The LLM, the predictor (B0), the prompt templates and the tool code. Only the two memory layers change.

## Split (user decisions)
- The 2,018 PRING BFS test proteins are split at random (fixed seed) into **discovery proteins (60%,
  ~1,211)** and **final-test proteins (40%, ~807)**. A protein is in one set only.
- **Discovery candidates:** every non-self pair with both ends among discovery proteins (~733k pairs, ~10.2k
  hits, ~1.4%). Hits are the PRING positives among them (closed-world, as before).
- **Final test set:** PRING's released labelled test pairs (`hidden/test.csv`) with both ends among
  final-test proteins, no self pairs, balanced 1:1 by subsampling the larger class with a fixed seed: about
  5,140 + 5,140 = 10,280 pairs (built; `data/PRING/memory_split.json`). The discovery view has 1,211 proteins,
  732,655 pairs and 10,334 hits.
- **Cross pairs** (one end in each set) are in neither set. About half of the PRING positives are cross pairs.
- **The ribosome stays** on both sides (user decision). Reports give round hits and final-test metrics with and
  without ribosome–ribosome pairs *(my call)*.
- The KB is unchanged (PRING train + val, `sandbox/kb.csv`).

## Memory layers
**Case memory** (written by the system, never by the LLM). One record per valid tested pair: pair, round, B0
p_uv, the selector's `ranking_score` and `uncertainty`, `rationale` (selection reason), `evidence_ids`, result
(hit / not detected). All original records stay available through the `cases` tool.

**Experience memory** (written by the reviewer). Each lesson: id, text, scope (in words: the pairs or proteins
it applies to), supporting case ids, contrary case ids, update record (created in round t, revised in rounds
…, retired in round t with a reason). Lessons are hypotheses: new feedback can revise or retire them. At most
30 active lessons *(my call)*. Retired lessons stay in the file and the agent does not see them.

**`cases` tool** (paged, canonical order; filters combine):
- `protein`: cases that touch this protein;
- `similar_to`: cases whose proteins are sequence homologs of a given protein (needs a new mmseqs search of the
  2,018 test proteins against the discovery proteins, `case_homologs.tsv` in each view). A final-test protein
  reaches only discovery cases, because only discovery pairs are cases. Only 281 of the 807 final-test proteins
  (and 398 of 1,211 discovery proteins) have such a homolog, so `query` carries most of the transfer;
- `query`: text match on the case proteins' annotations (family, Pfam, GO, keyword);
- `pair`, `result`, `round`.
Lessons cite cases by pair id.

## One round (t = 1 … 10, K = 200)
1. **Selector session** (fresh pi session, as today): the active lessons are in the round prompt; tools as
   today on the discovery candidates, with `memory` replaced by `cases`; one `submit_batch` of 200 rows.
2. **Oracle:** results for the valid rows (validity rules unchanged). The system adds one case per valid row.
3. **Reviewer session** (fresh pi session, same LLM and thinking level, its own fixed prompt template). Input:
   current active lessons and the new round's cases with results. Tools: `cases`, protein, KB and `predict_ppi`
   tools; no candidate ranking, no `submit_batch`. Output: one `update_lessons` call with operations `add`
   (text, scope, supporting, contrary), `revise` (id + changed fields), `retire` (id + reason). The system
   rejects an operation that cites an unknown case id or an `add` over the limit, and logs it. The reviewer
   also runs after round 10, so the final test reads lessons that include round 10.
4. **Save:** `memory/round<t>/cases.jsonl` and `memory/round<t>/lessons.json` before round t + 1. A round with
   saved memory is replayed, so runs resume.

## Final test (protocol as `pring_test`)
Pairs shuffled with seed 0, batches of 200, one fresh pi session per batch, 80 tool calls, one
`submit_predictions` call (p and evidence_ids per pair). The memory is read-only and frozen at `round10/`.
Tools: `predict_ppi`, `protein_info`, `search_proteins`, `similar_proteins`, `kb_partners`, and `cases`
where the arm has cases. No `top_candidates`, `top_for_protein` or `submit_batch`.

**Arms (one run each, user decision):**
1. **The memory arm:** lessons (in the prompt) + cases.
2. **The cases-only arm:** cases, no lessons.
3. **The empty-memory arm:** nothing from the rounds; the agent predicts the test set directly.
4. **B0 alone:** evaluator-side logits, no LLM.

Comparisons: cases-only − empty-memory = value of the raw revealed labels; memory − cases-only = value of
the lessons. With one run per arm these are point estimates, not significance tests.

**Metrics:** as `pring_test` (AUROC, AP, max-F1 set, ECE, Brier, coverage, tool calls, tokens, cost), all
pairs and without ribosome–ribosome pairs. Rounds: hits per round and cumulative, as before. Memory: lesson
count, adds/revisions/retirements per round, cases cited per lesson.

**LLM:** `deepseek/deepseek-flash`, thinking `high`, for selector, reviewer and every test arm. A smoke run
first: 2 rounds + reviewer, then 1 test batch per LLM arm.

## Claim boundary
PRING labels are public and can be memorised; all numbers are internal (project rule). One run per arm
cannot separate an effect from sampling variance.

## Not in this version
Replicates; a permuted-feedback arm; a test after every round; RSI of prompts, tools or the predictor; the
private TMP split; the 22T mouse split.
