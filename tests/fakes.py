"""Test-only stand-ins for BGE-M3 and the LLMs, so tests run on a laptop CPU without downloads."""
from typing import List
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from llama_index.core.embeddings import BaseEmbedding

_vec = HashingVectorizer(n_features=1024, alternate_sign=False, ngram_range=(1, 2), norm="l2", stop_words="english")

class HashEmbedding(BaseEmbedding):
    def _emb(self, t): return _vec.transform([t]).toarray()[0].tolist()
    def _get_text_embedding(self, text: str) -> List[float]: return self._emb(text)
    def _get_query_embedding(self, query: str) -> List[float]: return self._emb(query)
    async def _aget_query_embedding(self, query: str) -> List[float]: return self._emb(query)

class FakeLLM:
    """Echoes the first sentence of the first excerpt with a page citation, or refuses on a keyword."""
    name = "fake"
    def generate(self, messages):
        user = messages[-1]["content"]
        if "leptospirosis" in user.lower().split("question:")[-1]:
            from src.rag import REFUSAL; return REFUSAL
        first = user.split("]\n", 1)[1].split("\n")[1][:200]
        page = user.split("CPG p. ", 1)[1].split("]", 1)[0]
        return f"{first} (CPG p. {page})"
    def stream(self, messages):
        for w in self.generate(messages).split(" "): yield w + " "
    def unload(self): pass
