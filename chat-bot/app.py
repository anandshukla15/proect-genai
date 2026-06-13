"""
Streamlit UI for the document-grounded RAG chatbot.

Run with:
    streamlit run app.py
"""

import streamlit as st

from rag_chatbot import build_vectorstore, build_qa_chain, ask, NOT_FOUND

st.set_page_config(page_title="Doc RAG Chatbot", page_icon="📄")
st.title("📄 Chat with your PDF")
st.caption("Answers come only from the uploaded document. No outside knowledge.")


@st.cache_resource(show_spinner="Building vector index...")
def load_chain(file_bytes: bytes, name: str):
    """Cache the chain per uploaded file so we don't re-index on every question."""
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        vs = build_vectorstore(tmp_path)
        return build_qa_chain(vs)
    finally:
        os.unlink(tmp_path)


with st.sidebar:
    st.header("1. Upload a PDF")
    uploaded = st.file_uploader("Your notes / document", type="pdf")
    st.markdown(
        "Set `GROQ_API_KEY` or `GOOGLE_API_KEY` in a `.env` file before running."
    )

if not uploaded:
    st.info("👈 Upload a PDF in the sidebar to get started.")
    st.stop()

chain = load_chain(uploaded.getvalue(), uploaded.name)

# Keep chat history in session state.
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("pages"):
            st.caption(f"Sources: pages {msg['pages']}")

question = st.chat_input("Ask a question about the document...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the document..."):
            out = ask(chain, question)
        answer = out["answer"]
        st.markdown(answer)

        pages = None
        # Only show sources when we actually answered from the document.
        if NOT_FOUND.lower() not in answer.lower():
            pages = sorted({d.metadata.get("page", "?") for d in out["sources"]})
            if pages:
                st.caption(f"Sources: pages {pages}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "pages": pages}
    )
