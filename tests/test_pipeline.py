"""
CPU-only checks of the pipeline. Real models (BGE-M3, Qwen3-8B, Gemma 3 12B) are replaced
by stand-ins from fakes.py, so these tests check the plumbing, not answer quality.

Run:  python -m pytest -q
"""
import pytest

from fakes import FakeLLM, HashEmbedding
from src import evaluate as ev
from src.config import DEV_QUESTIONS
from src.index import build_index, load_index
from src.ingest import run_ingest
from src.rag import REFUSAL, DenguideRAG


@pytest.fixture(scope="module")
def ingest_output():
    return run_ingest()


@pytest.fixture(scope="module")
def rag(ingest_output):
    _, chunks, _ = ingest_output
    build_index(chunks, HashEmbedding(), rebuild=True)
    return DenguideRAG(load_index(HashEmbedding()), llm=FakeLLM(), cutoff=0.30)


def test_ingest_counts(ingest_output):
    paragraphs, chunks, stats = ingest_output
    assert stats["pages_kept"] == 114
    assert 120 <= stats["chunks"] <= 200
    assert stats["tables"] >= 40
    assert all(c["section"] != "REFERENCES" for c in chunks)
    assert all(c["section"] != "Key Clinical Issues and Questions" for c in chunks)


def test_tables_keep_decimals(ingest_output):
    paragraphs, _, _ = ingest_output
    text = " ".join(p["text"] for p in paragraphs if p["kind"] == "table")
    assert "0.89" in text and "0 89." not in text


def test_chunks_have_pages_and_questions(ingest_output):
    _, chunks, _ = ingest_output
    assert all(c["pages"] for c in chunks)
    questions = {c["question"] for c in chunks if c["question"]}
    assert {"1A", "1B", "2", "3", "4", "5", "6", "7", "8"} <= questions


def test_input_validation():
    assert DenguideRAG.validate("") is not None
    assert DenguideRAG.validate("x" * 1000) is not None
    assert DenguideRAG.validate("Is ORS recommended?") is None


def test_retrieval_and_refusal(rag):
    hits = rag.retrieve("Should ORS be given to patients with probable dengue?")
    assert len(hits) == rag.top_k
    assert hits == sorted(hits, key=lambda h: h["score"], reverse=True)
    high_cutoff = DenguideRAG(rag.retriever._index, cutoff=0.999)
    assert high_cutoff.answer("What antibiotic should be given for leptospirosis?")["answer"] == REFUSAL


def test_evaluation_on_dev_set(rag):
    dev = ev.load_questions(DEV_QUESTIONS)
    assert len(dev) == 15
    table = ev.retrieval_table(rag, dev)
    metrics = ev.retrieval_metrics(table)
    assert 0 <= metrics["recall@4"] <= 1
    cutoff, curve = ev.tune_cutoff(table)
    assert 0.3 <= cutoff <= 0.8
    results = ev.run_questions(rag, dev, "fake")
    summary = ev.summarize({"fake": results})
    assert {"refusal_accuracy", "latency_mean_s"} <= set(summary.columns)
