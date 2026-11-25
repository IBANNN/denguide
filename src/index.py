"""
Step 2 of the indexing pipeline: chunks -> BGE-M3 embeddings -> persistent Chroma collection.

The collection uses cosine distance. LlamaIndex reports a score of exp(-distance),
so retrieval code converts it back to cosine similarity with 1 + ln(score).
"""
import math

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from .config import CONFIG, CHROMA_DIR

META_KEYS = ["pages", "page_start", "page_end", "chapter", "question",
             "question_title", "section", "n_words"]


def get_embed_model(device=None):
    """Load BGE-M3 through the LlamaIndex Hugging Face integration (downloads ~2.3 GB once)."""
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    return HuggingFaceEmbedding(
        model_name=CONFIG["embed_model"],
        max_length=CONFIG["embed_max_length"],
        embed_batch_size=16,
        normalize=True,
        device=device,
    )


def _collection(rebuild=False):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    name = CONFIG["collection"]
    if rebuild and name in [c.name for c in client.list_collections()]:
        client.delete_collection(name)
    return client.get_or_create_collection(name, configuration={"hnsw": {"space": "cosine"}})


def chunks_to_nodes(chunks):
    """Wrap each chunk dict in a LlamaIndex TextNode. Metadata is stored but not embedded."""
    nodes = []
    for c in chunks:
        node = TextNode(id_=c["chunk_id"], text=c["text"], metadata={k: c[k] for k in META_KEYS})
        # The section header is already inside the text, so metadata is kept out of
        # the embedding and out of the LLM prompt (we format the prompt ourselves).
        node.excluded_embed_metadata_keys = META_KEYS
        node.excluded_llm_metadata_keys = META_KEYS
        nodes.append(node)
    return nodes


def build_index(chunks, embed_model, rebuild=True):
    """Embed all chunks and store them in Chroma. Returns (index, collection)."""
    collection = _collection(rebuild=rebuild)
    storage = StorageContext.from_defaults(vector_store=ChromaVectorStore(chroma_collection=collection))
    index = VectorStoreIndex(chunks_to_nodes(chunks), storage_context=storage,
                             embed_model=embed_model, show_progress=True)
    return index, collection


def load_index(embed_model):
    """Re-open the saved Chroma collection without re-embedding."""
    collection = _collection(rebuild=False)
    if collection.count() == 0:
        raise RuntimeError("The Chroma collection is empty. Run the indexing step first.")
    store = ChromaVectorStore(chroma_collection=collection)
    return VectorStoreIndex.from_vector_store(store, embed_model=embed_model)


def score_to_cosine(score):
    """Convert the LlamaIndex Chroma score exp(-cosine_distance) back to cosine similarity."""
    return 1.0 + math.log(score) if score and score > 0 else -1.0
