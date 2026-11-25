"""
Evaluation on the development and unseen question sets.

Question CSV columns
    id, question, type, reference_answer, gold_pages, in_scope
    gold_pages: printed CPG page numbers separated by ";" (empty for out-of-scope questions)
    in_scope:   1 if the guideline answers it, 0 if not

Metrics
    Retrieval precision@k, recall@k (hit rate), MRR   - from gold_pages, before the cutoff
    Refusal accuracy                                  - out-of-scope refused, in-scope answered
    Faithfulness                                      - optional Gemini judge on answered questions
    Latency                                           - seconds per question
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CONFIG


# ---------------------------------------------------------------- question files
def load_questions(path):
    df = pd.read_csv(path, dtype={"gold_pages": str, "id": str}).fillna({"gold_pages": "", "reference_answer": ""})
    df["in_scope"] = df["in_scope"].astype(int)
    return df


def file_sha256(path):
    """Fingerprint of the unseen question file. Record it when the file is created."""
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _gold(pages_str):
    return {p.strip() for p in str(pages_str).split(";") if p.strip()}


def _hit_pages(hit):
    return {p.strip() for p in hit["pages"].split(",") if p.strip()}


# ---------------------------------------------------------------- retrieval only
def retrieval_table(rag, questions):
    """Retrieve for every question (no LLM). Used for metrics and cutoff tuning."""
    rows = []
    for q in questions.itertuples():
        hits = rag.retrieve(q.question)
        gold = _gold(q.gold_pages)
        relevant = [bool(gold & _hit_pages(h)) for h in hits]
        rows.append({
            "id": q.id, "type": q.type, "in_scope": q.in_scope,
            "top_score": hits[0]["score"] if hits else -1.0,
            "retrieved_pages": " | ".join(h["pages"] for h in hits),
            "scores": " | ".join(f"{h['score']:.3f}" for h in hits),
            "relevant_flags": relevant,
        })
    return pd.DataFrame(rows)


def retrieval_metrics(table):
    """precision@k, recall@k (hit rate) and MRR over in-scope questions."""
    t = table[table["in_scope"] == 1]
    if t.empty:
        return {}
    precision = t["relevant_flags"].apply(lambda f: sum(f) / len(f) if f else 0.0)
    recall = t["relevant_flags"].apply(lambda f: 1.0 if any(f) else 0.0)
    mrr = t["relevant_flags"].apply(lambda f: next((1 / (i + 1) for i, x in enumerate(f) if x), 0.0))
    k = len(t["relevant_flags"].iloc[0])
    return {f"precision@{k}": round(precision.mean(), 3), f"recall@{k}": round(recall.mean(), 3),
            "mrr": round(mrr.mean(), 3), "n_in_scope": int(len(t))}


def tune_cutoff(table, grid=None):
    """
    Pick the similarity cutoff that best separates in-scope from out-of-scope questions
    using the top retrieval score. Run on the DEV set only.
    """
    grid = grid if grid is not None else np.round(np.arange(0.30, 0.80, 0.01), 2)
    rows = []
    for c in grid:
        answered = table["top_score"] >= c
        correct = np.where(table["in_scope"] == 1, answered, ~answered)
        rows.append({"cutoff": c, "refusal_accuracy": correct.mean(),
                     "in_scope_kept": answered[table["in_scope"] == 1].mean(),
                     "out_scope_refused": (~answered)[table["in_scope"] == 0].mean()})
    curve = pd.DataFrame(rows)
    best = curve.sort_values(["refusal_accuracy", "in_scope_kept"], ascending=False).iloc[0]
    return float(best["cutoff"]), curve


# ---------------------------------------------------------------- full runs
def run_questions(rag, questions, model_name, out_csv=None):
    """Answer every question with the current LLM and save one row per question."""
    rows = []
    for q in questions.itertuples():
        res = rag.answer(q.question)
        gold = _gold(q.gold_pages)
        rows.append({
            "id": q.id, "model": model_name, "type": q.type, "in_scope": q.in_scope,
            "question": q.question, "reference_answer": q.reference_answer,
            "answer": res["answer"], "refused": res["refused"], "error": res["error"],
            "latency_s": res["latency_s"],
            "top_score": res["retrieved"][0]["score"] if res["retrieved"] else -1.0,
            "retrieved_pages": " | ".join(h["pages"] for h in res["retrieved"]),
            "used_pages": " | ".join(h["pages"] for h in res["used"]),
            "relevant_flags": [bool(gold & _hit_pages(h)) for h in res["retrieved"]],
            "context": "\n\n".join(f"[p. {h['pages']}] {h['text']}" for h in res["used"]),
        })
        print(f"{q.id:>6} | {res['latency_s']:>5}s | refused={res['refused']} | {res['answer'][:70]!r}")
    df = pd.DataFrame(rows)
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df


def refusal_metrics(df):
    answered = ~df["refused"]
    correct = np.where(df["in_scope"] == 1, answered, ~answered)
    out = df[df["in_scope"] == 0]
    ins = df[df["in_scope"] == 1]
    return {"refusal_accuracy": round(correct.mean(), 3),
            "out_of_scope_refused": round(out["refused"].mean(), 3) if len(out) else None,
            "in_scope_wrongly_refused": round(ins["refused"].mean(), 3) if len(ins) else None}


def latency_metrics(df):
    lat = df.loc[df["error"].isna(), "latency_s"]
    return {"latency_mean_s": round(lat.mean(), 2), "latency_median_s": round(lat.median(), 2)}


# ---------------------------------------------------------------- optional LLM judge
JUDGE_PROMPT = """You check whether an answer is supported by the given context from a clinical guideline.

Context:
{context}

Answer:
{answer}

Split the answer into separate factual claims (ignore page citations).
For each claim decide if the context directly supports it.
Return JSON only, in this form:
{{"claims": [{{"claim": "...", "supported": true}}], "notes": "short reason for any unsupported claim"}}"""


def judge_faithfulness(df, api_key, model=None, pause_s=4.0, max_retries=3):
    """
    Score faithfulness = supported claims / total claims for each answered question.
    Uses the Gemini API (free tier is rate limited, so the loop pauses between calls).
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    model = model or CONFIG["judge_model"]
    scores, notes = [], []
    for row in df.itertuples():
        if row.refused or row.error or not row.context:
            scores.append(np.nan); notes.append("refused or no context"); continue
        prompt = JUDGE_PROMPT.format(context=row.context, answer=row.answer)
        for attempt in range(max_retries):
            try:
                resp = client.models.generate_content(
                    model=model, contents=prompt,
                    config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json"))
                data = json.loads(resp.text)
                claims = data.get("claims", [])
                supported = sum(bool(c.get("supported")) for c in claims)
                scores.append(supported / len(claims) if claims else np.nan)
                notes.append(data.get("notes", ""))
                break
            except Exception as e:                       # rate limit or bad JSON: wait and retry
                if attempt == max_retries - 1:
                    scores.append(np.nan); notes.append(f"judge error: {e}")
                time.sleep(pause_s * (attempt + 2))
        time.sleep(pause_s)
    out = df.copy()
    out["faithfulness"] = scores
    out["judge_notes"] = notes
    return out


# ---------------------------------------------------------------- summary
def summarize(run_dfs):
    """One row per model with every metric. run_dfs: dict model_name -> results DataFrame."""
    rows = []
    for name, df in run_dfs.items():
        row = {"model": name}
        row.update(retrieval_metrics(df))
        row.update(refusal_metrics(df))
        if "faithfulness" in df:
            row["faithfulness_mean"] = round(df["faithfulness"].mean(), 3)
        if "correctness" in df:
            row["correctness_mean_1to5"] = round(df["correctness"].mean(), 2)
        row.update(latency_metrics(df))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- saved results and manual ratings
def load_results(path):
    """Reload a saved run. relevant_flags is stored as text in the CSV, so parse it back."""
    import ast
    df = pd.read_csv(path, dtype={"id": str})
    df["relevant_flags"] = df["relevant_flags"].apply(ast.literal_eval)
    df["error"] = df["error"].where(df["error"].notna(), None)
    return df


def export_rating_sheet(run_dfs, path, overwrite=False):
    """
    One sheet for two raters to score correctness (1 = wrong, 5 = fully correct) without seeing the model name.
    Each answer gets a row code (R001, R002, ...). The key file links row codes to models,
    so raters may sort or filter the sheet. An existing sheet is kept unless overwrite=True,
    since it may already hold ratings.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists and may hold ratings. "
                              "Use overwrite=True only to start the ratings again.")
    rows = []
    for name, df in run_dfs.items():
        part = df[["id", "question", "reference_answer", "answer"]].copy()
        part["model"] = name
        rows.append(part)
    sheet = pd.concat(rows).sample(frac=1, random_state=CONFIG["seed"]).reset_index(drop=True)
    sheet.insert(0, "row", [f"R{i:03d}" for i in range(1, len(sheet) + 1)])
    sheet["rater1_1to5"] = ""
    sheet["rater2_1to5"] = ""
    sheet.drop(columns="model").to_csv(path, index=False)          # blind copy for raters
    sheet[["row", "id", "model"]].to_csv(path.with_name(path.stem + "_key.csv"), index=False)
    return sheet


def attach_ratings(df, model_name, rated_path):
    """Average the two raters' scores and add a 'correctness' column to one model's results."""
    rated_path = Path(rated_path)
    rated = pd.read_csv(rated_path, dtype={"id": str, "row": str})
    key = pd.read_csv(rated_path.with_name(rated_path.stem + "_key.csv"), dtype={"id": str, "row": str})
    rated = rated.merge(key[["row", "model"]], on="row", how="left")   # match by row code, not row order
    rated = rated[rated["model"] == model_name].copy()
    rated["correctness"] = rated[["rater1_1to5", "rater2_1to5"]].astype(float).mean(axis=1)
    # Drop an older correctness column so the cell can be run again safely.
    return df.drop(columns="correctness", errors="ignore").merge(rated[["id", "correctness"]], on="id", how="left")
