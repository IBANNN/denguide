# Denguide run log

One section per Colab run, newest at the bottom. Each run has a saved notebook copy in `notebooks/runs/`
and its output files in `outputs/`. Numbers here are copied from the notebook outputs.

| Step | Date | Notebook copy | Outputs commit |
|---|---|---|---|
| 1. Index and cutoff | 2026-11-25 | [2026-11-25_Step1_index_and_cutoff.ipynb](../notebooks/runs/2026-11-25_Step1_index_and_cutoff.ipynb) | `06113ae` |
| 2. Dev runs, attempt 1 (superseded) | 2026-11-25 | not saved | `81aa305`, files in `outputs/attempts/2026-11-25_Step2_attempt1/` |

---

## Step 1: cleaning, BGE-M3 index, cutoff tuning (2026-11-25)

**Setup.** Google Colab, Tesla T4 (15 GB). Code at commit `c588cf7`; cutoff rerun after `43df3db` (middle of tied cutoffs).

| Library | Version |
|---|---|
| torch | 2.14.1+cu130 |
| transformers | 5.22.1 |
| accelerate | 1.15.0 |
| bitsandbytes | 0.50.2 |
| llama-index-core | 0.14.26 |
| llama-index-vector-stores-chroma | 0.7.0 |
| llama-index-embeddings-huggingface | 0.9.0 |
| chromadb | 1.5.9 |
| pymupdf | 2.1 |
| streamlit | 1.67.0 |
| google-genai | 2.12.1 |
| pandas | 2.2.3 |

### Cleaning and chunking (notebook Section 2)

| Item | Value |
|---|---|
| PDF pages kept | 114 of 211 |
| Paragraphs after cleaning | 635 |
| Tables rebuilt | 53 |
| Characters after cleaning | 195,083 |
| Chunks | 151 |
| Mean / max words per chunk | 220.8 / 424 |

Chunks per guideline question: 1A 18, 1B 13, 2 17, 3 13, 4 14, 5 11, 6 11, 7 14, 8 13, other sections 27.
Chart: `outputs/results/chunk_lengths.png`.

### Index (notebook Section 2.3)

BAAI/bge-m3, 1024 dimensions, 151 chunks embedded in 22.2 s on the T4. Chroma collection `denguide`, cosine distance.
Files: `outputs/chroma_db/`, `outputs/data/processed/chunks.jsonl`.

### Retrieval check (notebook Section 4)

"Should ORS be given to patients with probable dengue?" returned four Question 4 chunks (p. 71 three times, p. 75),
cosine 0.706 to 0.725, with the recommendation chunk among them.

### Development set retrieval and cutoff (notebook Section 4.3)

| Metric (12 in-scope dev questions, top 4) | Value |
|---|---|
| Recall@4 (gold page retrieved) | 1.000 |
| MRR | 0.792 |
| Precision@4 | 0.333 |

Precision is low by design of the gold labels: most dev questions list one gold page, and the other retrieved chunks
come from neighbouring pages of the same section.

| id | type | in scope | top score | retrieved pages |
|---|---|---|---|---|
| d01 | recommendation | 1 | 0.7846 | 26 \| 33,34 \| 28 \| 32 |
| d02 | numeric_table | 1 | 0.6693 | 26,27 \| 28,29 \| 34 \| 32 |
| d03 | recommendation | 1 | 0.6259 | 33,34 \| 26 \| 40 \| 45,46 |
| d04 | recommendation | 1 | 0.7646 | 52 \| 53,54 \| 59,60 \| 62 |
| d05 | recommendation | 1 | 0.7229 | 64 \| 64,65 \| 66 \| 68 |
| d06 | recommendation | 1 | 0.7251 | 71 \| 71 \| 71 \| 75 |
| d07 | recommendation | 1 | 0.7812 | 84 \| 79 \| 84 \| 86 |
| d08 | herbal | 1 | 0.7231 | 98 \| 96 \| 90 \| 91 |
| d09 | prevention | 1 | 0.7663 | 103 \| 105,106 \| 102 \| 104,105 |
| d10 | numeric_table | 1 | 0.6789 | 27,28 \| 5 \| 1 \| 6 |
| d11 | taglish | 1 | 0.6362 | 52 \| 33,34 \| 59,60 \| 62 |
| d12 | taglish | 1 | 0.7641 | 90,91 \| 90 \| 94 \| 97 |
| d13 | out_of_scope | 0 | 0.5597 | 75,76 \| 76,77 \| 1 \| 58,59 |
| d14 | out_of_scope | 0 | 0.5875 | 65,66 \| 65 \| 44,45 \| 47,48 |
| d15 | out_of_scope | 0 | 0.4794 | 79 \| 86 \| 87 \| 86 |

**Cutoff: 0.61.** The lowest in-scope top score is 0.626 (d03) and the highest out-of-scope top score is 0.588 (d14).
Cutoffs 0.59 to 0.62 all reach 100% refusal accuracy on the dev set. The middle of that range (0.61) was kept for a
similar margin on both sides. The gap is narrow (about 0.04), so some unseen questions may fall on the wrong side.
Saved in `outputs/settings_override.json`; chart in `outputs/results/cutoff_tuning.png`.

### Unseen set (notebook Section 5.1)

40 questions (15 recommendation, 8 numeric, 7 prevention/herbal, 5 out-of-scope, 5 Taglish), written by Adrian Ivan Aldea.
Gold pages were converted from PDF to printed page numbers before any tuning. The file stays on Drive only.

SHA-256 `e7baf3a48cb94f559d4467fc75410836f546c7c23ed2ac9eaeb400bb1ae88c11`, recorded 2026-11-25.

---

## Step 2, attempt 1: dev runs of both LLMs (2026-11-25, superseded)

**Setup.** Same Colab T4 session as Step 1. Code at commit `c588cf7` with the cutoff fix from `43df3db`; cutoff 0.61.
Both models in 4-bit NF4 with float16 compute. Result files: `outputs/attempts/2026-11-25_Step2_attempt1/`.

| Model | GPU memory after load | Smoke test | In-scope answered | Out-of-scope refused | Mean latency |
|---|---|---|---|---|---|
| Qwen3-8B | 7.79 GB (incl. 2.1 GB BGE-M3) | OK | 11 of 12 | 3 of 3 | 11.06 s |
| Gemma 3 12B-IT | 9.39 GB (incl. 2.1 GB BGE-M3) | empty reply | 0 of 12 (empty output) | 3 of 3 | 54.2 s |

**Problem 1: Gemma 3 gave empty output in float16.** Every reply was empty and each one ran to the 512-token limit
(about 67 s). Gemma 3 was trained in bfloat16, and the T4 has no native bfloat16. Float16 overflow is the likely cause.
**Change:** Gemma runs with float32 compute (`CONFIG["compute_dtype"]`, commit `5bb1978`).

**Problem 2: the cutoff removed a needed chunk (d03).** The cutoff was tuned on the top retrieval score only, but the
code applied it to each chunk. For d03 ("Which rapid test is recommended for a patient who consults more than 5 days
after symptom onset?") the recommendation on p. 40 ranked third with a score below 0.61, so only pp. 33-34 and 26 reached
the LLM. Qwen3-8B said IgM "may be considered" and then gave the refusal sentence.
**Change:** the cutoff became a gate on the best chunk. When the best chunk passes, all four chunks go to the LLM
(`DenguideRAG.relevant()`, commit `5bb1978`). Out-of-scope questions still get the refusal without an LLM call.

**Qwen3-8B answer notes (other 11 in-scope questions).** Facts matched the guideline on every question checked
(d01, d02, d04, d06, d07, d08, d10, d12). Points for review:
- d09 says "recommends against" where the guideline says "suggests against" (weak recommendation is still stated).
- d11 lists admission criteria from pp. 59-60 next to the DOH criteria without naming the other source.
- d12 is a Taglish question answered in English.
