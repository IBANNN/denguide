# Run 05: baseline on dev set v2, attempt 1

**Date:** 2026-11-25 · **Status:** ❌ failed (Gemma out of GPU memory at d16) · **Runtime:** Colab T4 · **Code:** `5709f88`
· **Outputs commit:** `e051518` · [All runs](../README.md)

## Changes since Run 04

| Commit | Change | Changes results? |
|---|---|---|
| `5709f88` | Dev set v2: 25 more questions (d16-d40), 40 in total | **yes** (new questions, cutoff re-tuned on 11 out-of-scope questions) |

Code, prompt (v2) and models were the same as Run 04.

## Settings

Cutoff re-tuned on dev set v2: **0.61** (unchanged) · gate on the best chunk · top-k 4 · prompt v2 · Qwen3-8B float16 ·
Gemma 3 12B float32 · BGE-M3 on the GPU during the LLM runs.

## Result

**Retrieval (40 questions, 29 in scope):** recall@4 0.931, MRR 0.759, precision@4 0.379. The two misses were labeling
problems in the new questions, corrected before Run 06: d22's answer is also stated on p. 30 (gold was p. 29 only),
and d17 was ambiguous (the guideline reports odds ratios for low platelets from several reviews).

**Cutoff.** The groups now overlap. The lowest in-scope top score is 0.613 (d17), and three out-of-scope questions score
above 0.61: d33 ibuprofen (0.620), d34 dexamethasone (0.629), d40 lagundi/kamote (0.633). Cutoffs 0.60 to 0.63 all
reach 37 of 40; the tie-break kept 0.61.

**Qwen3-8B (complete, 40 questions):**

| Measure | Value |
|---|---|
| In-scope answered | 29 of 29 |
| Out-of-scope refused | 10 of 11 (d33 and d34 passed the cutoff and were refused by the prompt; d40 was answered) |
| Citation accuracy | 1.000; 12 of 75 sentences without a citation |
| Taglish answered in Filipino | 1 of 7 in scope (d29 only) |
| Mean latency | 8.45 s |

d40 ("Effective ba ang lagundi o kamote tops para tumaas ang platelet…?") was answered with the papaya evidence: the
model swapped the herbs it was asked about for one the excerpts cover. This is a new failure type (I-13).

**Gemma 3 12B-IT:** answered d01-d16, then ran out of GPU memory on d16's long prompt (excerpts with large tables).
Its results were not saved; its first 16 answers are only in `notebook.ipynb`.

## Problems found and what changed after

| Issue | Problem | Change for Run 06 |
|---|---|---|
| I-11 | Gemma 3 12B (float32) ran out of GPU memory with BGE-M3 also on the GPU | BGE-M3 moves to the CPU before any LLM loads; PyTorch `expandable_segments`; the app embeds on the CPU too |
| I-12 | On dev set v2 the cutoff cannot separate in-scope from out-of-scope questions (3 out-of-scope above 0.61) | none yet (experiment candidates: reranker scores, hybrid retrieval) |
| I-13 | An out-of-scope herbal question (d40) was answered with another herb's evidence | none yet |
| I-06 | Taglish: Qwen answered 1 of 7 in Filipino | none yet (prompt v3 candidate) |
| labels | d17 ambiguous, d22 missing p. 30 | d17 reworded, d22 gold pages 29;30 |

## Files

| File | What it is |
|---|---|
| `notebook.ipynb` | notebook after the run, with the out-of-memory error |
| `results/dev_qwen3-8b.csv` | Qwen3-8B answers on all 40 dev questions |
| `results/citation_check.csv` | Qwen3-8B: pages given, pages cited, cited pages not given |
| `results/cutoff_tuning.png` | cutoff curve on dev set v2 (overlapping groups) |
| `results/chunk_lengths.png`, `results/settings_override.json` | chart and the saved cutoff |
