# Run 04: Step 2, dev runs with prompt v2

**Date:** 2026-11-25 · **Status:** ✅ done · **Runtime:** Colab T4 · **Code:** `8cd9609` · **Outputs commit:** `8903db0`
· [All runs](../README.md)

## Changes since Run 03

| Commit | Change | Changes results? |
|---|---|---|
| `8cd9609` | Prompt v2: rules for I-04 to I-09, approved by Jessica Dorosan | **yes** |
| `8cd9609` | Citation check built into `run_questions()` and `summarize()`; each result row records its prompt version | no (new measurement) |
| `58a762d` | Change note in each run folder | no (record keeping) |

Only the prompt changed between Run 03 and Run 04, so the differences below come from the prompt.

## Why

Run 03 showed six repeated problems in the answers (I-04 to I-09). Prompt v2 adds one rule for each kind of problem.

## Settings

Same as Run 03 (cutoff 0.61 as a gate, top-k 4, Qwen3-8B float16, Gemma 3 12B float32, greedy, 512 tokens) except the
prompt: **v2**.

## Result: prompt v1 (Run 03) against v2 (Run 04)

| Measure | Qwen v1 | Qwen v2 | Gemma v1 | Gemma v2 | Issue |
|---|---|---|---|---|---|
| In-scope answered / out-of-scope refused | 12/12, 3/3 | 12/12, 3/3 | 12/12, 3/3 | 12/12, 3/3 | |
| Cited pages among the excerpts given | 25/25 | 24/24 | 18/20 | **22/23** | I-09 |
| Sentences without a citation | 12/44 | **4/32** | 7/31 | **2/31** | I-07 |
| Answers with "we recommend / we suggest" | 2 | **0** | 8 | **0** | I-08 |
| Guideline verb kept (first sentence, 7 questions) | 2/7 | **6/7** | 7/7 | 7/7 | I-04 |
| Recommendation answers stating certainty (of 9) | 7 | 6 | 6 | 4 | I-10 (new) |
| Taglish questions answered in Filipino or Taglish | 1/2 | 0/2 | 0/2 | 0/2 | I-06 |
| Mean latency, all 15 questions | 10.91 s | 8.88 s | 25.25 s | 28.14 s | |

The same table is in `results/prompt_v1_vs_v2.csv`.

**What improved.** No answer speaks as "we" anymore (I-08). Uncited sentences dropped from 12 to 4 for Qwen and from 7
to 2 for Gemma (I-07). Qwen kept the guideline's verb in 6 of 7 answers instead of 2 (I-04). Gemma's citation errors
dropped from 2 to 1 (I-09). Both models now name WHO, PAHO or the Malaysia CPG when they use other groups' criteria (I-05).

**What did not improve.** Both models still answer the two Taglish questions in English, and Qwen answered d11 in
Filipino with v1 but in English with v2 (I-06). Qwen still writes "recommends against" tawa-tawa, where the guideline
says "suggests against" (d08). Strength and certainty were left out more often, mostly by Gemma (I-10).

**Gemma's p. 93 citation (I-09).** The tawa-tawa evidence table on p. 96 says "1 observational study (n = 93)". Gemma
read the sample size as a page number. This is a model error on numbers in tables, not a retrieval error.

## Problems found

| Issue | Status after this run |
|---|---|
| I-04 verb | ✅ mostly fixed (Qwen 6/7); d08 still wrong for Qwen |
| I-05 other groups | ✅ fixed: other groups are named |
| I-06 Taglish | ❌ not fixed |
| I-07 uncited sentences | ✅ mostly fixed (Qwen 4/32, Gemma 2/31 sentences) |
| I-08 "we recommend" | ✅ fixed |
| I-09 pages not given | 🟡 1 left (Gemma, "n = 93" read as p. 93) |
| I-10 certainty left out (new) | 🔍 open |

## Files

| File | What it is |
|---|---|
| `notebook.ipynb` | notebook after the run |
| `results/dev_qwen3-8b.csv`, `results/dev_gemma3-12b.csv` | answers, pages given and cited, prompt version, latency |
| `results/citation_check.csv` | per answer: pages given, pages cited, cited pages not given |
| `results/prompt_v1_vs_v2.csv` | the comparison table above (paper: prompt design results) |
| `results/cutoff_tuning.png`, `results/chunk_lengths.png` | charts from this run |
| `results/settings_override.json` | the saved cutoff |
