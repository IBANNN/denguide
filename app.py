"""
Denguide web app (Streamlit).

Run from the project folder:   streamlit run app.py
The app reuses the same index, retriever and prompt as the notebook (src/ package).
"""
import logging
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))   # make `src` importable
from src.config import CONFIG, LOGS_DIR, ensure_dirs        # noqa: E402
from src.index import get_embed_model, load_index           # noqa: E402
from src.rag import DenguideRAG, REFUSAL                     # noqa: E402

ensure_dirs()
logging.basicConfig(filename=LOGS_DIR / "app.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("denguide.app")

RETRIEVAL_ONLY = "Retrieval only (no LLM)"
EXAMPLES = [
    "Is the NS1 rapid diagnostic test recommended for suspected dengue?",
    "Should ORS be given to patients with probable dengue?",
    "Kailan dapat i-admit sa ospital ang pasyenteng may dengue?",
]

st.set_page_config(page_title="Denguide", page_icon="🦟", layout="centered")


# ---------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner="Loading the guideline index...")
def get_rag():
    # BGE-M3 runs on the CPU so the LLM has the whole GPU (Gemma 3 12B in float32 needs it on a T4).
    return DenguideRAG(load_index(get_embed_model(device="cpu")))


@st.cache_resource
def model_slot():
    """Holds the one LLM kept on the GPU. Switching models unloads the old one first."""
    return {"name": None, "llm": None}


def get_llm(name):
    slot = model_slot()
    if slot["name"] != name:
        from src.llm import LocalLLM, free_gpu          # imported here so retrieval-only mode needs no GPU
        if slot["llm"] is not None:
            slot["llm"].unload()
            slot.update(name=None, llm=None)
            free_gpu()
        slot["llm"] = LocalLLM(name)
        slot["name"] = name
    return slot["llm"]


# ---------------------------------------------------------------- display helpers
def show_sources(hits, title):
    if not hits:
        return
    with st.expander(title):
        for h in hits:
            st.markdown(f"**CPG p. {h['pages'].replace(',', ', ')}** · similarity {h['score']:.2f}")
            st.caption(h["text"][:1200] + ("..." if len(h["text"]) > 1200 else ""))


def show_message(msg):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            show_sources(msg.get("sources"), msg.get("sources_title", "Sources"))
            if msg.get("meta"):
                st.caption(msg["meta"])


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Answer model", list(CONFIG["llms"]) + [RETRIEVAL_ONLY])
    st.caption(f"Top-k: {CONFIG['top_k']} · similarity cutoff: {CONFIG['sim_cutoff']}")
    st.caption("Knowledge base: DOH Clinical Practice Guideline on Dengue (May 2023)")
    if st.button("Clear chat"):
        st.session_state.messages = []
    with st.expander("Example questions"):
        for q in EXAMPLES:
            st.write("• " + q)

# ---------------------------------------------------------------- main page
st.title("🦟 Denguide")
st.caption("Guideline lookup for primary care providers · AI3 Group 11")
st.info("For health professionals. Answers come only from the 2023 DOH dengue CPG and "
        "support, but do not replace, clinical judgment.")

if "messages" not in st.session_state:
    st.session_state.messages = []
for m in st.session_state.messages:
    show_message(m)

question = st.chat_input("Ask about dengue diagnosis, management or prevention")
if question is not None:
    problem = DenguideRAG.validate(question)
    if problem:
        st.warning(problem)
        st.stop()

    user_msg = {"role": "user", "content": question}
    st.session_state.messages.append(user_msg)
    show_message(user_msg)

    with st.chat_message("assistant"):
        start = time.perf_counter()
        reply = {"role": "assistant", "content": "", "sources": [], "sources_title": "Sources"}
        try:
            with st.spinner("Searching the guideline..."):
                rag = get_rag()
                hits = rag.retrieve(question)
                used = rag.relevant(hits)

            if not used:                                  # nothing above the cutoff: refuse
                reply["content"] = REFUSAL
                reply.update(sources=hits[:2], sources_title="Closest excerpts (below the cutoff)")
                st.markdown(REFUSAL)
            elif mode == RETRIEVAL_ONLY:
                reply["content"] = "Retrieval-only mode: the most relevant guideline excerpts are listed below."
                reply["sources"] = used
                st.markdown(reply["content"])
            else:
                with st.spinner(f"Loading {mode} (first time takes a few minutes)..."):
                    rag.llm = get_llm(mode)
                reply["content"] = st.write_stream(rag.stream_answer(question, used))
                reply["sources"] = used

            latency = time.perf_counter() - start
            reply["meta"] = f"{mode} · {latency:.1f} s"
            show_sources(reply["sources"], reply["sources_title"])
            st.caption(reply["meta"])
            log.info("mode=%s latency=%.2f refused=%s q=%r", mode, latency, not used, question)
        except Exception:                                 # model, GPU or index failure
            log.exception("Failed to answer %r", question)
            reply["content"] = ("Sorry, something went wrong while answering. Please try again. "
                                "If it keeps happening, switch to *Retrieval only* in the sidebar.")
            st.error(reply["content"])
        st.session_state.messages.append(reply)
