# Run 02: Step 2, dev runs, attempt 1

**Date:** 2026-11-25 · **Status:** ❌ superseded by Run 03 · **Runtime:** Colab T4 · **Code:** `c588cf7` with `43df3db`
· [All runs](../README.md)

## Changes since Run 01

No code changes. This run added the LLM step: Qwen3-8B and Gemma 3 12B-IT, both in 4-bit NF4 with float16 compute,
on the 15 development questions, with prompt v1.

## Settings

Same retrieval as Run 01 (cutoff 0.61, top-k 4) · cutoff applied as a **filter on each chunk** · Qwen3-8B float16 ·
Gemma 3 12B float16 · prompt v1 · 512 new tokens, greedy decoding.

## Result

| Model | In-scope answered | Out-of-scope refused | Mean latency |
|---|---|---|---|
| Qwen3-8B | 11 of 12 | 3 of 3 | 11.06 s |
| Gemma 3 12B-IT | 0 of 12 (empty output) | 3 of 3 | 54.2 s |

## Problems found and what changed after

| Issue | Problem | Change for Run 03 |
|---|---|---|
| I-02 | Gemma 3 gave empty output in float16 (no native bfloat16 on the T4) | Gemma runs in float32 (`5bb1978`) |
| I-03 | The cutoff was applied to each chunk, so d03's answer chunk (p. 40, rank 3) never reached the LLM | Cutoff became a gate on the best chunk (`5bb1978`) |
| I-04 | Qwen d09: "recommends against" for the guideline's "suggests against" | none yet (later prompt v2) |
| I-05 | d11 mixed other groups' admission criteria with the DOH criteria | none yet (later prompt v2) |
| I-06 | Taglish question answered in English (d12) | none yet (later prompt v2) |

## Files

| File | What it is |
|---|---|
| `notebook.ipynb` | notebook after the run, with Qwen's answers and Gemma's empty output |
| `results/dev_qwen3-8b.csv` | Qwen3-8B answers, pages used, latency |
| `results/dev_gemma3-12b.csv` | Gemma 3 12B output (empty answers) |
| `results/settings_override.json` | the saved cutoff |
