"""
Denguide configuration.

Two folders:
    REPO_DIR  the GitHub repo (code, guideline PDF, development questions)
    WORK_DIR  outputs and private files (chunks, Chroma database, results, logs, unseen questions)
              Set with the DENGUIDE_DIR environment variable. On Colab the notebook points it to
              Google Drive. Locally it defaults to REPO_DIR/workspace (ignored by git).

Tuned values (for example the similarity cutoff) are saved to WORK_DIR/settings_override.json
and loaded automatically, so the notebook and the app always use the same settings.
"""
import json
import os
from pathlib import Path

# ---------------------------------------------------------------- paths
REPO_DIR = Path(__file__).resolve().parents[1]
WORK_DIR = Path(os.environ.get("DENGUIDE_DIR", REPO_DIR / "workspace"))
BASE_DIR = WORK_DIR   # kept for older notebook cells

PDF_PATH = REPO_DIR / "data" / "raw" / "CPG Dengue 2023.pdf"
DEV_QUESTIONS = REPO_DIR / "eval" / "dev" / "dev_questions.csv"

PROCESSED_DIR = WORK_DIR / "data" / "processed"
PARAGRAPHS_PATH = PROCESSED_DIR / "paragraphs.jsonl"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
CHROMA_DIR = WORK_DIR / "chroma_db"
UNSEEN_QUESTIONS = WORK_DIR / "eval" / "unseen" / "unseen_questions.csv"   # private, never in git before the final run
RESULTS_DIR = WORK_DIR / "results"
LOGS_DIR = WORK_DIR / "logs"
CONFIG_PATH = RESULTS_DIR / "config.json"
OVERRIDE_PATH = WORK_DIR / "settings_override.json"

# ---------------------------------------------------------------- settings
CONFIG = {
    # Knowledge base
    # PDF page numbers to keep (1-based). Page 2 = disclaimer and intended users.
    # Pages 12-124 = printed pages 1-113 (main body). Front matter lists and
    # appendices (pp. 125-211) are excluded.
    "keep_pdf_pages": [2] + list(range(12, 125)),

    # Chunking (word counts; 1 word is about 1.3 BGE-M3 tokens)
    "chunk_words": 350,      # about 450 tokens
    "overlap_words": 50,     # about 65 tokens carried into the next chunk
    "min_chunk_words": 12,   # drop tiny fragments
    # Sections left out of the index. "Key Clinical Issues and Questions" restates every
    # guideline question without answering it, so it outranked the real answers in testing.
    "exclude_sections": ["Key Clinical Issues and Questions"],

    # Embeddings and vector store
    "embed_model": "BAAI/bge-m3",
    "embed_max_length": 1024,
    "collection": "denguide",

    # Retrieval
    "top_k": 4,
    "sim_cutoff": 0.45,      # cosine similarity; tune with tune_cutoff() on the DEV set only

    # Generators (Hugging Face IDs)
    "llms": {
        "Qwen3-8B": "Qwen/Qwen3-8B",
        "Gemma 3 12B": "google/gemma-3-12b-it",
    },
    # Number format for 4-bit compute. "auto" = bfloat16 on GPUs that support it (L4, A100),
    # float16 on the T4. If Gemma gives empty or garbled text on a T4, set its value to "float32".
    "compute_dtype": {"Qwen3-8B": "auto", "Gemma 3 12B": "auto"},
    "max_new_tokens": 512,
    "do_sample": False,          # greedy decoding = same answer every run

    # App limits
    "max_query_chars": 500,

    # Optional LLM judge (Gemini API key needed)
    "judge_model": "gemini-2.5-flash",

    "seed": 42,
}


def ensure_dirs():
    """Create every output folder if it does not exist yet."""
    for d in [PROCESSED_DIR, CHROMA_DIR, UNSEEN_QUESTIONS.parent, RESULTS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def save_config(path=CONFIG_PATH):
    """Freeze the current settings to results/config.json (used before the unseen run)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(CONFIG, indent=2))
    return path


def load_frozen_config(path=CONFIG_PATH):
    return json.loads(Path(path).read_text())


def save_override(**values):
    """Store tuned settings (e.g. sim_cutoff=0.52) in WORK_DIR so the app picks them up."""
    current = json.loads(OVERRIDE_PATH.read_text()) if OVERRIDE_PATH.exists() else {}
    current.update(values)
    OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
    OVERRIDE_PATH.write_text(json.dumps(current, indent=2))
    CONFIG.update(values)
    return current


# Apply saved overrides on import.
if OVERRIDE_PATH.exists():
    CONFIG.update(json.loads(OVERRIDE_PATH.read_text()))
