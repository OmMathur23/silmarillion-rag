import sys
sys.path.insert(0, "src")

import streamlit as st
from rag_pipeline import RAGPipeline

st.set_page_config(
    page_title="Silmarillion RAG",
    page_icon="💍",
    layout="centered",
)


@st.cache_resource
def load_pipeline():
    """
    Cached so the Chroma connection + API config only happen ONCE per
    session, not on every single message (Streamlit reruns the whole
    script top-to-bottom on every interaction - without this cache,
    you'd reconnect to Chroma on every question).
    """
    return RAGPipeline(top_k=5)


st.title("The Silmarillion — Q&A")
st.caption("Ask anything about Tolkien's Silmarillion. Answers are grounded in the actual text via RAG.")

pipeline = load_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay chat history on every rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources used"):
                for s in message["sources"]:
                    st.markdown(f"- **{s['section']}** (similarity: {s['similarity']:.3f})")

question = st.chat_input("Ask a question about the Silmarillion...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant lore and generating an answer..."):
            result = pipeline.answer(question)

        st.markdown(result["answer"])
        with st.expander("Sources used"):
            for s in result["sources"]:
                st.markdown(f"- **{s['section']}** (similarity: {s['similarity']:.3f})")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })