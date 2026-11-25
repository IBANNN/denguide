# Denguide: notes for Claude Code

AI3 Machine Project, Group 11. RAG chatbot over the DOH Dengue Clinical Practice Guideline (May 2023)
for primary care providers. See README.md for the layout and workflow.

## Facts to keep consistent
- Knowledge base: `data/raw/CPG Dengue 2023.pdf`, 211 pages. Kept: PDF p. 2 and pp. 12-124 (printed pp. 1-113).
- Cleaning output: 114 pages kept, 53 tables rebuilt, 151 chunks (350 words, 50-word overlap).
- Models: BGE-M3 (embeddings), Qwen3-8B (main), Gemma 3 12B-IT (comparison), Gemini Flash (optional judge).
- **Never use Llama or Mistral models** (group decision; Mixtral counts as Mistral).
- Retrieval: top-4, cosine similarity cutoff tuned on the dev set only; below the cutoff the bot refuses without calling the LLM.
- Both LLMs must share the same chunks, retriever, prompt and decoding settings. Only the LLM changes.

## Rules
- The unseen question set is private. Do not create, read or commit `eval/unseen/unseen_questions.csv` before the final run.
- Do not tune anything on the unseen set.
- Generated files go to the work folder (`DENGUIDE_DIR`, default `workspace/`), never to git.
- This machine has no GPU for the LLMs. `src/llm.py` runs on Colab. Test everything else with `python -m pytest -q`.
- Keep code readable and commented: every member must be able to explain it at the demo.
- Run the tests before committing. Use a branch and a pull request for each task.

## Writing style for the paper and notebook text
Natural, concise, modest. Sentences about two lines max. No em dashes. Avoid gerunds and sentences starting
with -ing words. Avoid "however", "moreover", "although". Avoid which/that relative clauses. Prefer "due to"
over "because". Interpret results instead of listing values. No claims beyond the data.
Notebook references use APA; the IEEE paper uses IEEE style.
