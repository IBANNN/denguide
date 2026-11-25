"""
Query pipeline: question -> retrieve top-k chunks -> cutoff check -> grounded prompt -> LLM answer.
"""
import time

from .config import CONFIG
from .index import score_to_cosine

REFUSAL = "This question is not covered by the DOH Dengue Clinical Practice Guideline (2023)."

SYSTEM_PROMPT = f"""You are Denguide, an assistant for Filipino primary care providers.
You answer questions using only the excerpts from the DOH Clinical Practice Guideline on Dengue (2023) given in the user message.

Rules:
1. Use only facts stated in the excerpts. Do not add outside medical knowledge.
2. After each statement, cite the page in the form (CPG p. X).
3. When the guideline gives a recommendation, state its strength and certainty of evidence as written.
4. If the excerpts do not answer the question, reply only with: "{REFUSAL}"
5. Do not give patient-specific doses or diagnoses that the excerpts do not state.
6. Reply in the language of the question (English or Filipino). Keep medical terms in English.
7. Be concise: at most 150 words unless a list is needed."""


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
        return [h for h in hits if h["score"] >= self.cutoff]

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
