# Run 01: Step 1, index and cutoff

**Date:** 2026-11-25 · **Status:** ✅ done · **Runtime:** Colab T4 · **Code:** `c588cf7`, cutoff rerun on `43df3db`
· [All runs](../README.md)

## Changes before this run

This was the first run. These commits came between the initial project (`6aa7cc7`) and the run:

| Commit | Change | Changes results? |
|---|---|---|
| `8a69f7e` | `save_to_github()` cell: Colab outputs are copied to `outputs/` and pushed | no (record keeping) |
| `9f8f0d0` | Install without `-U` (keeps Colab's torch); token hidden in git messages; Section 6 imports `free_gpu`; blind ratings matched by row code; AI tools table | no (setup and safety) |
| `c588cf7` | Tokens stripped of line breaks; anything token-shaped hidden in git output | no (setup) |
| `3a747ab` | Autoreload made optional (Colab's IPython cannot load it on Python 3.13) | no (setup) |
| `43df3db` | `tune_cutoff()` takes the middle of tied cutoffs instead of the lowest | **yes:** cutoff 0.59 became 0.61 |

## Why

`43df3db` came from this run (issue I-01). Cutoffs 0.59 to 0.62 all separated the dev questions perfectly, and the
code picked 0.59, only 0.002 above an out-of-scope question. The middle value leaves a similar margin on both sides.
The tuning cell was rerun after the change.

## Settings

Chunks 350 words with 50 overlap · BGE-M3 · top-k 4 · cutoff **0.61** · no LLM in this run.

## Result

151 chunks from 114 pages; 53 tables rebuilt. Dev retrieval: recall@4 1.000, MRR 0.792, precision@4 0.333. Unseen set
hash recorded (`e7baf3a4…c11`). Full numbers in [All runs](../README.md#run-01-step-1-index-and-cutoff).

## Files

| File | What it is |
|---|---|
| `notebook.ipynb` | notebook after the run |
| `results/chunk_lengths.png` | chunk length histogram (paper Section 2) |
| `results/cutoff_tuning.png` | cutoff curve with 0.61 marked (paper Section 4.3) |
| `results/settings_override.json` | the saved cutoff |
