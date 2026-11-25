# Run 03: Step 2, dev runs, attempt 2

**Date:** 2026-11-25 · **Status:** ✅ done · **Runtime:** Colab T4 · **Code:** `dc574ee` (result changes from `5bb1978`)
· [All runs](../README.md)

## Changes since Run 02

| Commit | Change | Changes results? |
|---|---|---|
| `5bb1978` | Cutoff is a gate on the best chunk: if it passes, all 4 chunks go to the LLM (`DenguideRAG.relevant()`) | **yes** (fixes I-03) |
| `5bb1978` | Gemma 3 12B runs with float32 compute (`CONFIG["compute_dtype"]`) | **yes** (fixes I-02) |
| `5bb1978` | `save_to_github()` sets a git name and email in Colab and pushes leftover commits | no (setup) |
| `316eb3e`, `0b4eb2b`, `dc574ee` | Run log, numbered run folders, Run 02 notebook | no (record keeping) |

## Why

Both result changes came from Run 02. Gemma gave only empty output in float16 (I-02). The cutoff had been tuned on
the top score of each question but was applied to every chunk, so the LLM lost lower-ranked chunks that held the
answer (I-03).

## Settings

Cutoff 0.61 as a **gate on the best chunk** · Qwen3-8B float16 · Gemma 3 12B **float32** · prompt v1 · everything else
as Run 02.

## Result

| Model | In-scope answered | Out-of-scope refused | Citation accuracy | Mean latency |
|---|---|---|---|---|
| Qwen3-8B | 12 of 12 | 3 of 3 | 25 of 25 | 10.91 s |
| Gemma 3 12B-IT | 12 of 12 | 3 of 3 | 18 of 20 | 25.25 s |

I-02 and I-03 confirmed fixed: Gemma answers normally, and d03 now gives the IgM/IgG test from p. 40.

## Problems found and what changed after

| Issue | Problem | Change for Run 04 |
|---|---|---|
| I-04 to I-06 | Still present (verb wording, other groups' criteria, English answers to Taglish) | prompt v2 rules 4, 5, 8 |
| I-07 | Qwen adds sentences without a citation, some not in the excerpts | prompt v2 rules 1, 2 |
| I-08 | Gemma copies "we recommend", so answers read as Denguide's own advice | prompt v2 rule 3 |
| I-09 | Gemma cited 2 pages it was not given (p. 33, p. 93) | prompt v2 rule 2; citation accuracy built into every run |

Prompt v2 was approved by Jessica Dorosan on 2026-11-25 (commit `8cd9609`).

## Files

| File | What it is |
|---|---|
| `notebook.ipynb` | notebook after the run |
| `results/dev_qwen3-8b.csv` | Qwen3-8B answers, pages used, latency |
| `results/dev_gemma3-12b.csv` | Gemma 3 12B answers, pages used, latency |
| `results/citation_check.csv` | per answer: pages given, pages cited, cited pages not given |
| `results/cutoff_tuning.png`, `results/chunk_lengths.png` | charts from this run |
| `results/settings_override.json` | the saved cutoff |
