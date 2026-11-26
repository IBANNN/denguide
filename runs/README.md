# Denguide runs

Every Colab run gets a number, failed runs included. Each run has its own folder here:

```
runs/RunNN_<date>_<step>_<short name>/
├── README.md          what changed since the previous run, why, the commits, and what the change did
├── notebook.ipynb     the notebook as it was after the run (when it was downloaded)
└── results/           the result files of that run (CSVs, charts, settings)
```

`outputs/` always holds the latest files from Google Drive. This folder keeps every run apart, so runs can be compared.

How to read this page:
- **Run index:** one line per run.
- **Settings per run:** what was different in each run.
- **Issue tracker:** every problem a run found, what changed, and whether a later run confirmed the fix.
- **Run details:** the full numbers for each run.

## Run index

| Run | Date | Step | Status | Code | What changed | Main result | Issues found |
|---|---|---|---|---|---|---|---|
| [Run 01](#run-01-step-1-index-and-cutoff) | 2026-11-25 | 1. Index and cutoff | ✅ done | `c588cf7`, `43df3db` | first run | 151 chunks; dev recall@4 1.0, MRR 0.792; cutoff 0.61 | I-01 |
| [Run 02](#run-02-step-2-dev-runs-attempt-1) | 2026-11-25 | 2. Dev runs | ❌ superseded | `43df3db` | both LLMs added (float16) | Qwen 11 of 12 in-scope answered; Gemma empty | I-02 to I-06 |
| [Run 03](#run-03-step-2-dev-runs-attempt-2) | 2026-11-25 | 2. Dev runs | ✅ done | `dc574ee` (fixes from `5bb1978`) | cutoff as a gate; Gemma in float32 | both models: 12 of 12 in-scope answered, 3 of 3 refused; citation accuracy Qwen 25/25, Gemma 18/20 | I-07, I-08, I-09 |
| [Run 04](#run-04-step-2-dev-runs-with-prompt-v2) | 2026-11-25 | 2. Dev runs | ✅ done | `8cd9609` | **prompt v2** (approved by Jessica Dorosan); citation check built into the evaluation | no "we"; uncited sentences Qwen 12→4, Gemma 7→2; citation accuracy Qwen 24/24, Gemma 22/23; Taglish still English | I-10 |
| [Run 05](#run-05-baseline-on-dev-set-v2-attempt-1) | 2026-11-25 | Experiments | ❌ failed | `5709f88` | **dev set v2 (40 questions)** | retrieval recall@4 0.931, MRR 0.759; cutoff 0.61 no longer separates; Qwen complete (refusal 0.975); Gemma out of GPU memory at d16 | I-11, I-12, I-13 |
| Run 06 | | Experiments | ⏳ next | | **BGE-M3 on the CPU during LLM runs**; d17 and d22 labels corrected | | |

## Settings per run

| Setting | Run 01 | Run 02 | Run 03 | Run 04 | Run 05 | Run 06 |
|---|---|---|---|---|---|---|
| Chunk size / overlap (words) | 350 / 50 | 350 / 50 | 350 / 50 | 350 / 50 | 350 / 50 | 350 / 50 |
| Embedding model | BGE-M3 | BGE-M3 | BGE-M3 | BGE-M3 | BGE-M3 | BGE-M3 |
| Top-k | 4 | 4 | 4 | 4 | 4 | 4 |
| Similarity cutoff | 0.61 | 0.61 | 0.61 | 0.61 | 0.61 (re-tuned) | re-tuned |
| How the cutoff is applied | (retrieval only) | filter on each chunk | **gate on the best chunk** | gate on the best chunk | gate | gate |
| Qwen3-8B compute dtype | | float16 | float16 | float16 | float16 | float16 |
| Gemma 3 12B compute dtype | | float16 | **float32** | float32 | float32 | float32 |
| Prompt | | v1 | v1 | **v2** | v2 | v2 |
| Max new tokens / decoding | | 512 / greedy | 512 / greedy | 512 / greedy | 512 / greedy | 512 / greedy |
| BGE-M3 device during LLM runs | | GPU | GPU | GPU | GPU | **CPU** |
| Dev set | v1 (15) | v1 (15) | v1 (15) | v1 (15) | **v2 (40)** | v2 (40), d17/d22 fixed |

Prompt v1 is `SYSTEM_PROMPT` in `src/rag.py` as of commit `6aa7cc7`. Prompt v2 (`PROMPT_VERSION = "v2"`) adds rules for
I-04 to I-09. It was drafted from the Run 03 development answers and approved by Jessica Dorosan on 2026-11-25, a member
who did not write the unseen set. From Run 04 on, every result row records its prompt version. Any prompt change gets a
new version number here.

## Dev set versions

| Version | Questions | In scope / out of scope | Taglish | Used in |
|---|---|---|---|---|
| v1 | d01-d15 | 12 / 3 | 2 | Runs 01-04 |
| v2 | d01-d40 | 29 / 11 | 9 (2 of them out of scope) | Run 05 on |

**Why v2.** With 3 out-of-scope and 2 Taglish questions, v1 was too small to compare experiments: the cutoff gap rested
on three questions, and one Taglish answer changed the Taglish rate by 50 points.

**How d16-d40 were made.** Claude Code drafted them from the cleaned guideline text (`outputs/data/processed/`), in
the categories of the unseen set. Claude Code never read the unseen questions. Checks before adding them:
- every gold page was confirmed by finding the answer's key phrase on that page in the cleaned text and in the index;
- the 8 out-of-scope topics (NSAIDs, corticosteroids, fogging, vitamin C/zinc, typhoid antibiotics, tourniquet test,
  lagundi, kamote tops) have no mention in the index;
- a script compared each draft with the 40 unseen questions and printed only similarity scores. Five drafts scored
  too close (text similarity above 0.70 or word overlap above 0.50) and were replaced, without anyone reading which
  unseen questions they matched. The kept questions score at most 0.68 and 0.40.

d16-d40 still need a check against the PDF by Jessica Dorosan or Ricci Balugay (the course requires AI output to be
validated). The first 15 questions are unchanged, so v1 results stay comparable on d01-d15.

**Label fixes after Run 05.** Two new questions had labeling problems: d22's answer is also stated on p. 30 (gold pages
now 29;30), and d17 was ambiguous because the guideline reports odds ratios for low platelets from several reviews
(now "In the pooled analysis of 20 studies…"). The reworded d17 was checked again against the unseen set (scores 0.47 / 0.13).

## Unseen set versions

| Version | Questions | Written by | Date | SHA-256 |
|---|---|---|---|---|
| v1 | U01-U40 | Adrian Ivan Aldea, before any tuning | 2026-11-25 | `e7baf3a48cb94f559d4467fc75410836f546c7c23ed2ac9eaeb400bb1ae88c11` (superseded) |
| v2 | U01-U50 | U01-U40 as above; U41-U50 by Claude Code | 2026-11-26 | `bef34b6a803391c44c0bee5222316cbef933379ea17a0fca3ea7a70252bbd55e` |

**Why v2.** The course asks for at least 50 unseen questions. The first 40 rows are byte for byte the v1 file.

**How U41-U50 were made.** Claude Code wrote them from the cleaned guideline text after Runs 01-05, without reading
U01-U40, and only from facts no development question uses. Checks before adding them:
- every gold page was confirmed by finding the answer's key phrase on that page in the cleaned text and in the index;
- the two out-of-scope topics (incubation period, antihistamines for itching) have no mention in the index;
- a script compared each draft with U01-U40 and with d01-d40 and printed only similarity scores. Three drafts scored too
  close (one was an exact match of an existing unseen question) and were replaced. The kept questions score at most
  0.69 text similarity and 0.33 word overlap.

**Limitation to declare.** The assistant that wrote U41-U50 also helped tune the prompt. These ten questions were never
used for tuning, and the final run uses the v2 hash.

Mix of v2: 19 recommendation, 10 numeric, 8 prevention and herbal, 6 out-of-scope, 7 Taglish (2 of them out of scope);
43 in scope, 7 out of scope.

## Issue tracker

| ID | Found in | Area | Problem | Change | Status |
|---|---|---|---|---|---|
| I-01 | Run 01 | Retrieval | Cutoffs 0.59 to 0.62 tied and the lowest (0.59) was picked, 0.002 above an out-of-scope score | Take the middle of a tie (`43df3db`) | ✅ fixed in Run 01 |
| I-02 | Run 02 | Model | Gemma 3 12B gave empty output in float16 on the T4 | Gemma in float32 (`5bb1978`) | ✅ confirmed in Run 03 |
| I-03 | Run 02 | Retrieval | Cutoff applied to each chunk, so the answer chunk for d03 (p. 40, rank 3) never reached the LLM | Cutoff as a gate on the best chunk (`5bb1978`) | ✅ confirmed in Run 03 |
| I-04 | Run 02 | Prompt | Qwen d09 says "recommends against" where the guideline says "suggests against" | Prompt v2 rule 4: keep the guideline's own verb | 🟡 Run 04: Qwen 2/7 → 6/7; d08 still "recommends against" |
| I-05 | Run 02 | Prompt | d11 lists admission criteria from other groups (pp. 59-60, 2009 WHO) next to the DOH criteria | Prompt v2 rule 5: other groups named or left out | ✅ confirmed in Run 04 (WHO, PAHO, Malaysia named) |
| I-06 | Run 02 | Prompt | Taglish questions answered in English (Qwen d12; Gemma d11 and d12) | Prompt v2 rule 8: Filipino or Taglish in, Filipino or Taglish out | ❌ not fixed: Run 04 0/2 both models; Run 05 Qwen 1/7 |
| I-07 | Run 03 | Prompt | Qwen adds sentences without a citation, some not in the excerpts (d03: why IgM/IgG works after day 5) | Prompt v2 rules 1-2: no own explanations, every sentence cited | 🟡 Run 04: uncited sentences Qwen 12/44 → 4/32, Gemma 7/31 → 2/31 |
| I-08 | Run 03 | Prompt | Gemma copies the guideline's "we recommend", so the answer reads as Denguide's own recommendation | Prompt v2 rule 3: third person ("The CPG recommends") | ✅ confirmed in Run 04 (0 answers with "we") |
| I-09 | Run 03 | Model | Gemma cited 2 pages it was not given (d02 p. 33, d08 p. 93; the tawa-tawa recommendation is on p. 90) | Prompt v2 rule 2: cite only excerpt-label pages; citation accuracy now measured in every run | 🟡 Run 04: Gemma 18/20 → 22/23; the one left is "n = 93" (p. 96 table) read as p. 93 |
| I-10 | Run 04 | Prompt | Strength and certainty left out more often with v2 (Gemma 6 → 4 of 9 recommendation answers, Qwen 7 → 6) | none yet | 🔍 open |
| I-11 | Run 05 | Setup | Gemma 3 12B (float32) ran out of GPU memory on a long prompt, with BGE-M3 also on the GPU | BGE-M3 on the CPU before any LLM loads; `expandable_segments`; app embeds on the CPU | 🔁 check in Run 06 |
| I-12 | Run 05 | Retrieval | On dev set v2 the cutoff no longer separates the groups: 3 out-of-scope questions score 0.620-0.633, the lowest in-scope 0.613 | none yet (experiments: reranker, hybrid retrieval) | 🔍 open |
| I-13 | Run 05 | Prompt | Out-of-scope herbal question (d40, lagundi/kamote) answered with papaya evidence instead of a refusal | none yet | 🔍 open |
| I-14 | Run 01 (found 2026-11-26) | Cleaning | In 5 of 53 tables (pp. 29-30, 42-44) a two-line column header is cut: "No. of Studies (No. of" instead of "No. of Studies (No. of participants)". The values are intact. | none yet | 🔍 open, minor |

Status key: ✅ fixed and confirmed · 🟡 improved, not fully fixed · 🔁 fixed, waiting for the next run to confirm · ❌ change did not work · 🔍 open.
Prompt issues are reviewed by a member who did not write the unseen set. Colab setup fixes (token handling,
autoreload, git identity in the save cell) are in the git history and do not change results.

---

## Run details

### Run 01: Step 1, index and cutoff

**2026-11-25 · Colab T4 · ✅ done** · folder [`Run01_2026-11-25_Step1_index-and-cutoff`](Run01_2026-11-25_Step1_index-and-cutoff/) · [changes in this run](Run01_2026-11-25_Step1_index-and-cutoff/README.md) · outputs commit `06113ae`

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

**Cleaning and chunking (notebook Section 2)**

| Item | Value |
|---|---|
| PDF pages kept | 114 of 211 |
| Paragraphs after cleaning | 635 |
| Tables rebuilt | 53 |
| Characters after cleaning | 195,083 |
| Chunks | 151 |
| Mean / max words per chunk | 220.8 / 424 |

Chunks per guideline question: 1A 18, 1B 13, 2 17, 3 13, 4 14, 5 11, 6 11, 7 14, 8 13, other sections 27.

**Index (notebook Section 2.3).** BAAI/bge-m3, 1024 dimensions, 151 chunks embedded in 22.2 s. Chroma collection
`denguide`, cosine distance.

**Retrieval check (notebook Section 4).** "Should ORS be given to patients with probable dengue?" returned four
Question 4 chunks (p. 71 three times, p. 75), cosine 0.706 to 0.725, with the recommendation chunk among them.

**Development set retrieval (notebook Section 4.3)**

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
similar margin on both sides (I-01). The gap is narrow (about 0.04), so some unseen questions may fall on the wrong side.

**Unseen set (notebook Section 5.1).** 40 questions (15 recommendation, 8 numeric, 7 prevention/herbal, 5 out-of-scope,
5 Taglish), written by Adrian Ivan Aldea. Gold pages were converted from PDF to printed page numbers before any tuning.
The file stays on Drive only. SHA-256 `e7baf3a48cb94f559d4467fc75410836f546c7c23ed2ac9eaeb400bb1ae88c11`, recorded 2026-11-25.

---

### Run 02: Step 2, dev runs, attempt 1

**2026-11-25 · Colab T4 · ❌ superseded by Run 03** · folder [`Run02_2026-11-25_Step2_dev-runs-attempt1`](Run02_2026-11-25_Step2_dev-runs-attempt1/) · [changes in this run](Run02_2026-11-25_Step2_dev-runs-attempt1/README.md) · outputs commit `81aa305` · notebook saved (downloaded before the Run 03 restart)

| Model | GPU memory after load | Smoke test | In-scope answered | Out-of-scope refused | Mean latency |
|---|---|---|---|---|---|
| Qwen3-8B | 7.79 GB (incl. 2.1 GB BGE-M3) | OK | 11 of 12 | 3 of 3 | 11.06 s |
| Gemma 3 12B-IT | 9.39 GB (incl. 2.1 GB BGE-M3) | empty reply | 0 of 12 (empty output) | 3 of 3 | 54.2 s |

**I-02: Gemma 3 gave empty output in float16.** Every reply was empty and each one ran to the 512-token limit
(about 67 s). Gemma 3 was trained in bfloat16, and the T4 has no native bfloat16. Float16 overflow is the likely cause.

**I-03: the cutoff removed a needed chunk (d03).** The cutoff was tuned on the top retrieval score only, but the code
applied it to each chunk. For d03 ("Which rapid test is recommended for a patient who consults more than 5 days after
symptom onset?") the recommendation on p. 40 ranked third below 0.61, so only pp. 33-34 and 26 reached the LLM.
Qwen3-8B said IgM "may be considered" and then gave the refusal sentence.

**Qwen3-8B answers (other 11 in-scope questions).** Facts matched the guideline on every question checked (d01, d02,
d04, d06, d07, d08, d10, d12). Open points: I-04 (d09 wording), I-05 (d11 attribution), I-06 (d12 language).

---

### Run 03: Step 2, dev runs, attempt 2

**2026-11-25 · Colab T4 · ✅ done** · folder [`Run03_2026-11-25_Step2_dev-runs-attempt2`](Run03_2026-11-25_Step2_dev-runs-attempt2/) · [changes in this run](Run03_2026-11-25_Step2_dev-runs-attempt2/README.md) · code `dc574ee`
· outputs commit `12421aa` (its message says "Step 2a…" by mistake; it holds the Run 03 files)

Both fixes confirmed: Gemma answers in float32 (I-02), and d03 now gets p. 40 and answers with the IgM/IgG test (I-03).
Retrieval is the same as Run 01 (recall@4 1.0, MRR 0.792, precision@4 0.333), since both models share it.

| Model | GPU memory after load | In-scope answered | Out-of-scope refused | Refusal accuracy | Citation accuracy | Mean / median latency |
|---|---|---|---|---|---|---|
| Qwen3-8B (float16) | 7.79 GB | 12 of 12 | 3 of 3 | 1.000 | 25 of 25 cited pages given | 10.91 / 11.75 s |
| Gemma 3 12B-IT (float32) | 11.28 GB | 12 of 12 | 3 of 3 | 1.000 | 18 of 20 cited pages given | 25.25 / 24.09 s |

GPU memory includes 2.1 GB for BGE-M3. Citation accuracy counts the distinct pages an answer cites that were among its
excerpts (`results/citation_check.csv`). Out-of-scope questions were refused before any LLM call (0.03 s).

**How the two models answer.** Gemma repeats the guideline text almost word for word, with strength and certainty in
the guideline's own format. It stays close to the excerpts, but it keeps the panel's "we recommend" (I-08) and twice
cited a page it was not given (I-09). Qwen paraphrases, explains the recommendation, and cites every page correctly,
but adds some uncited sentences (I-07) and once softened "suggests" into "recommends" (I-04). Qwen is about 2.3 times
faster. Both mixed other groups' criteria into d11 (I-05), and both answered some Taglish questions in English (I-06).

| id | Qwen3-8B | Gemma 3 12B-IT |
|---|---|---|
| d03 | IgM/IgG RDT, strong, low certainty (p. 40), plus an uncited rationale | guideline sentence word for word (p. 40) |
| d08 | suggests against tawa-tawa, weak, very low (p. 90) | same content, cites p. 93 instead of p. 90 |
| d09 | "recommends against" plant-based non-DEET (weak, low) | "we suggest against", weak, low (p. 102) |
| d11 (Taglish) | answered in Filipino; adds pp. 59-60 criteria as DOH basis | answered in English; adds 2009 WHO criteria (p. 59) |
| d12 (Taglish) | answered in English | answered in English |

### Run 04: Step 2, dev runs with prompt v2

**2026-11-25 · Colab T4 · ✅ done** · folder [`Run04_2026-11-25_Step2_dev-runs-prompt-v2`](Run04_2026-11-25_Step2_dev-runs-prompt-v2/) · [changes in this run](Run04_2026-11-25_Step2_dev-runs-prompt-v2/README.md) · code `8cd9609` · outputs commit `8903db0`

Only the prompt changed from Run 03 (v1 to v2), so any difference in the answers comes from the prompt.

Changes since Run 03:

| Commit | Change | Changes results? |
|---|---|---|
| `8cd9609` | Prompt v2: rules for I-04 to I-09 (approved by Jessica Dorosan) | **yes** |
| `8cd9609` | Citation check built into `run_questions()` and `summarize()`; every result row records its prompt version | no (new measurement) |

**Result against Run 03** (full table in [the Run 04 note](Run04_2026-11-25_Step2_dev-runs-prompt-v2/README.md) and `results/prompt_v1_vs_v2.csv`):

| Measure | Qwen v1 → v2 | Gemma v1 → v2 |
|---|---|---|
| In-scope answered / out-of-scope refused | 12/12, 3/3 → same | 12/12, 3/3 → same |
| Cited pages among the excerpts given | 25/25 → 24/24 | 18/20 → 22/23 |
| Sentences without a citation | 12/44 → 4/32 | 7/31 → 2/31 |
| Answers with "we recommend / we suggest" | 2 → 0 | 8 → 0 |
| Guideline verb kept (7 questions) | 2/7 → 6/7 | 7/7 → 7/7 |
| Recommendation answers stating certainty (of 9) | 7 → 6 | 6 → 4 |
| Taglish answered in Filipino or Taglish | 1/2 → 0/2 | 0/2 → 0/2 |
| Mean latency (all 15) | 10.91 → 8.88 s | 25.25 → 28.14 s |

Prompt v2 is better on five of the six target issues and keeps answering and refusing as before. It did not fix the
language of Taglish answers (I-06), and certainty was left out a little more often (I-10). Gemma's one citation error
comes from reading "n = 93" in a table as a page number.

### Run 05: baseline on dev set v2, attempt 1

**2026-11-25 · Colab T4 · ❌ failed** · folder [`Run05_2026-11-25_Exp_baseline-dev-v2-attempt1`](Run05_2026-11-25_Exp_baseline-dev-v2-attempt1/) · [changes in this run](Run05_2026-11-25_Exp_baseline-dev-v2-attempt1/README.md) · code `5709f88` · outputs commit `e051518`

First run on the 40-question dev set. Retrieval: recall@4 0.931, MRR 0.759, precision@4 0.379 (both misses were label
problems, fixed for Run 06). The cutoff stayed 0.61 but no longer separates the groups (I-12). Qwen3-8B finished all 40:
29 of 29 in scope answered, 10 of 11 out of scope refused (two of them by the prompt after passing the cutoff),
citation accuracy 1.000, Taglish in Filipino 1 of 7, 8.45 s per question. Gemma 3 12B ran out of GPU memory at d16 (I-11).

### Run 06: baseline on dev set v2, attempt 2

**⏳ next** · save message `save_to_github("Run 06: baseline on dev set v2 (40 questions)")`.
Same as Run 05 with BGE-M3 on the CPU during the LLM runs and the two label fixes. This run is the
baseline for the experiments (top-k, chunk size, reranker, hybrid retrieval, prompt v3).
