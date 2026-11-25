# Denguide

A retrieval-augmented (RAG) chatbot for DOH Dengue Guideline lookup in primary care.
AI3 Machine Project, Group 11: Aldea, Adrian Ivan D. · Balugay, Ricci Marcelino P. · Dorosan, Jessica Danielle D.

Denguide answers clinician questions using only the DOH Clinical Practice Guideline on Dengue (May 2023).
Each answer cites the guideline page. Questions the guideline does not cover get a fixed refusal.
The project compares two open-weight LLMs inside the same pipeline: **Qwen3-8B** and **Gemma 3 12B-IT**.

## Repository layout

```
denguide/
├── app.py                     Streamlit chat app
├── src/
│   ├── config.py              paths and every setting (chunk size, top-k, cutoff, model IDs)
│   ├── ingest.py              PDF -> clean paragraphs -> chunks
│   ├── index.py               BGE-M3 embeddings + Chroma vector store
│   ├── llm.py                 Qwen3-8B / Gemma 3 12B in 4-bit (needs a GPU)
│   ├── rag.py                 retrieval, cutoff, grounded prompt, refusal
│   └── evaluate.py            metrics, cutoff tuning, judge, blind ratings
├── notebooks/
│   ├── Denguide_RAG_Group11.ipynb   Colab notebook (Sections 0-10 of the spec), latest run
│   └── runs/                  dated copy of the notebook after each step's run
├── eval/
│   ├── dev/dev_questions.csv  15 development questions (tuning only)
│   └── unseen/                template only; the real unseen set stays private (see below)
├── data/raw/                  CPG Dengue 2023.pdf
├── tests/                     CPU tests with stand-in models
├── docs/                      progress report, model rationale, progress deck, run_log.md (results per step)
└── outputs/                   copy of the Drive work folder, pushed from Colab (chunks, index, results, logs)
```

Generated files (chunks, Chroma database, results, logs) go to the **work folder**: Google Drive on Colab,
or `workspace/` locally. Set it with the `DENGUIDE_DIR` environment variable. At the end of each Colab session,
`save_to_github()` in the notebook copies the Drive folder to `outputs/` in this repo and pushes it.
The unseen question set stays out of `outputs/` until the final run.

## How we work

| Where | What |
|---|---|
| **Claude Code / VS Code (laptop)** | Edit code, run `python -m pytest -q`, commit and push |
| **GitHub** | Single source of truth; issues, milestones and pull requests |
| **Google Colab (T4 GPU)** | Pull the latest code, build the index, run the LLMs, save results to Drive, push them to `outputs/` |

Branch per task (`step1-index`, `step2-qwen-dev`, ...), open a pull request, and have one other member review it.

## Running on Colab

1. Runtime > Change runtime type > **T4 GPU**.
2. Colab Secrets: `HF_TOKEN` (Hugging Face read token), `GITHUB_TOKEN` (fine-grained token for this repo with Contents: Read and write), optional `GEMINI_API_KEY` (judge).
3. Accept the Gemma license at https://huggingface.co/google/gemma-3-12b-it with the same Hugging Face account.
4. Open `notebooks/Denguide_RAG_Group11.ipynb` in Colab (File > Open notebook > GitHub) and run from the top.
5. End of session: set `GITHUB_USER` in Section 0.2 to your username and run the last cell (`save_to_github`).
   Save the notebook with File > Save a copy in GitHub.

## Running locally (CPU, no LLM)

```bash
pip install -r requirements.txt
python -m pytest -q                      # tests with stand-in models
python -c "from src.ingest import run_ingest; print(run_ingest()[2])"   # cleaning + chunking
```
The LLMs and BGE-M3 need a GPU for reasonable speed, so model runs happen on Colab.

## Unseen test set (leakage control)

- One member writes about 40 questions from the guideline **before any tuning**.
- The file lives only in the private Drive work folder (`eval/unseen/unseen_questions.csv`), not in this repo.
- Its SHA-256 hash is recorded in the progress report. The notebook checks the hash before the final run.
- It is committed here only after the final run (`FINAL_RUN_DONE = True` in the notebook).

## Status

- [x] Topic, users, knowledge base, model selection (see `docs/`)
- [x] Cleaning and chunking (114 of 211 pages kept, 53 tables rebuilt, 151 chunks)
- [x] Code for indexing, RAG, evaluation and app (tested with stand-ins)
- [x] Step 1: unseen set written and hashed; Colab setup; BGE-M3 index; cutoff tuned on dev set (0.61, see `docs/run_log.md`)
- [ ] Step 2: practice runs of Qwen3-8B and Gemma 3 12B
- [ ] Step 3: freeze settings; final unseen run, both models
- [ ] Step 4: judge, blind ratings, charts, app screenshots, consultation
- [ ] Step 5: IEEE paper, notebook write-up, deck, video
- [ ] Step 6: submit and demo

## AI use

Parts of this code and documentation were drafted with Claude (Anthropic). The group reviews, runs and
explains every part, and declares the AI use with full transcripts in the References section, as the course requires.
