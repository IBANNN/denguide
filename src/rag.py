"""
Query pipeline: question -> retrieve top-k chunks -> cutoff check -> grounded prompt -> LLM answer.
"""
import time

from .config import CONFIG
from .index import score_to_cosine

REFUSAL = "This question is not covered by the DOH Dengue Clinical Practice Guideline (2023)."

# Prompt versions are tracked in runs/README.md. v1 was used up to Run 03.
# v2 targets issues I-04 to I-09 found on the development set (rules 1-5 and 8).
PROMPT_VERSION = "v2"

SYSTEM_PROMPT = f"""You are Denguide, an assistant for Filipino primary care providers.
You answer questions using only the excerpts from the DOH Clinical Practice Guideline on Dengue (2023) given in the user message.

Rules:
1. Use only facts stated in the excerpts. Do not add outside medical knowledge or your own explanations.
2. End every sentence with a citation in the form (CPG p. X). Use only the page numbers shown in the excerpt labels.
3. Write about the guideline in the third person, for example "The CPG recommends" or "The CPG suggests". Never write "we recommend" or "we suggest".
4. Keep the guideline's own verb for each recommendation: recommends, suggests, recommends against, or suggests against. Give its strength and certainty of evidence as written.
5. Excerpts from the section RECOMMENDATIONS FROM OTHER GROUPS are not DOH recommendations. Leave them out, or name the group, for example "WHO (2009) suggests".
6. If the excerpts do not answer the question, reply only with: "{REFUSAL}"
7. Do not give patient-specific doses or diagnoses that the excerpts do not state.
8. Answer in the language of the question. If the question is in Filipino or Taglish, answer in Filipino or Taglish. Keep medical terms in English.
9. Be concise: at most 150 words unless a list is needed."""


def format_context(hits):
    """Number each excerpt and label it with its page and section."""
    blocks = []
    for i, h in enumerate(hits, 1):
        pages = h["pages"].replace(",", ", ")
        blocks.append(f"[Excerpt {i} | CPG p. {pages}]\n{h['text']}")
    return "\n\n".join(blocks)


def build_messages(question, hits):
    user = (f"Guideline excerpts:\n\n{format_context(hits)}\n\n"
            f"Question: {question}\n\nAnswer using only the excerpts above.")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


def is_refusal(answer):
    a = answer.lower()
    return "not covered by the doh dengue" in a or a.strip() == REFUSAL.lower()


class DenguideRAG:
    """Retriever + optional LLM. Without an LLM it still supports retrieval-only use."""

    def __init__(self, index, llm=None, top_k=None, cutoff=None):
        self.top_k = top_k or CONFIG["top_k"]
        self.cutoff = CONFIG["sim_cutoff"] if cutoff is None else cutoff
        self.retriever = index.as_retriever(similarity_top_k=self.top_k)
        self.llm = llm

    # ------------------------------------------------------------ input checks
    @staticmethod
    def validate(question):
        """Return an error message for bad input, or None when the question is fine."""
        if question is None or not question.strip():
            return "Please type a question."
        if len(question) > CONFIG["max_query_chars"]:
            return f"Please shorten the question to {CONFIG['max_query_chars']} characters or fewer."
        if len(question.strip()) < 3:
            return "The question is too short."
        return None

    # ------------------------------------------------------------ retrieval
    def retrieve(self, question):
        """Top-k chunks, best first, with cosine similarity scores."""
        hits = []
        for r in self.retriever.retrieve(question):
            m = r.node.metadata
            hits.append({
                "chunk_id": r.node.node_id, "text": r.node.get_content(),
                "score": round(score_to_cosine(r.score), 4),
                "pages": str(m.get("pages", "")), "question": m.get("question", ""),
                "section": m.get("section", ""), "chapter": m.get("chapter", ""),
            })
        return sorted(hits, key=lambda h: h["score"], reverse=True)

    def relevant(self, hits):
        """
        The cutoff is a gate on the best hit, the same score tune_cutoff() uses.
        If the best hit passes, the LLM gets all top-k hits; a lower-ranked chunk can hold the answer.
        """
        if hits and hits[0]["score"] >= self.cutoff:
            return hits
        return []

    # ------------------------------------------------------------ full answer
    def answer(self, question):
        """Run the whole pipeline and return a result dict (used by evaluation)."""
        start = time.perf_counter()
        result = {"question": question, "answer": "", "refused": False, "error": None,
                  "retrieved": [], "used": [], "latency_s": None}
        error = self.validate(question)
        if error:
            result.update(error=error, latency_s=0.0)
            return result
        hits = self.retrieve(question)
        used = self.relevant(hits)
        result.update(retrieved=hits, used=used)
        if not used:                                   # nothing similar enough: refuse, no LLM call
            result.update(answer=REFUSAL, refused=True)
        elif self.llm is None:
            result["answer"] = "(retrieval only) " + used[0]["text"][:400]
        else:
            text = self.llm.generate(build_messages(question, used))
            result.update(answer=text, refused=is_refusal(text))
        result["latency_s"] = round(time.perf_counter() - start, 2)
        return result

    def stream_answer(self, question, used_hits):
        """Yield answer pieces for the app. used_hits come from retrieve() + relevant()."""
        if not used_hits:
            yield REFUSAL
            return
        yield from self.llm.stream(build_messages(question, used_hits))
