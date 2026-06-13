"""
Core RAG pipeline for a document-grounded chatbot.

Flow (matches the assignment steps):
  1. Load   -> PyPDFLoader reads the PDF into LangChain Documents
  2. Split  -> RecursiveCharacterTextSplitter chunks the text
  3. Embed  -> HuggingFaceEmbeddings (sentence-transformers, local & free)
  4. Store  -> FAISS builds a local vector index from the chunks
  5. Retrieve + Generate -> RetrievalQA wires the retriever into an LLM
                            with a strict prompt that refuses to hallucinate.

Run the CLI directly:
    python rag_chatbot.py path/to/notes.pdf
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

# A small, fast, CPU-friendly embedding model. ~80MB, no API key needed.
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# The exact phrase the bot must use when the answer is not in the document.
NOT_FOUND = "Not found in document"

# Strict prompt: the model may ONLY use the retrieved context. This is what
# stops hallucination - if the context doesn't contain the answer, it must
# return the NOT_FOUND phrase verbatim instead of guessing.
PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=f"""You are a helpful assistant that answers questions using ONLY the
context provided below. The context comes from a single document.

Rules:
- Answer using ONLY facts found in the context.
- If the answer is not contained in the context, reply with exactly:
  "{NOT_FOUND}"
- Do not use any outside knowledge. Do not guess.

Context:
{{context}}

Question: {{question}}

Answer:""",
)


# ----------------------------------------------------------------------------
# Steps 1-4: build the vector index from a PDF
# ----------------------------------------------------------------------------
def build_vectorstore(
    pdf_path: str | Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> FAISS:
    """Load a PDF, split it, embed the chunks, and return a FAISS index."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # 1. Load
    docs = PyPDFLoader(str(pdf_path)).load()
    if not docs:
        raise ValueError("No text extracted from the PDF (is it scanned/image-only?)")

    # 2. Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    # 3. Embed (local sentence-transformers) + 4. Store (FAISS)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return FAISS.from_documents(chunks, embeddings)


# ----------------------------------------------------------------------------
# Pick a free-tier chat model (Groq by default, Gemini as fallback)
# ----------------------------------------------------------------------------
def get_llm(temperature: float = 0.0):
    """Return a chat LLM based on whichever API key is configured.

    temperature=0 keeps answers deterministic and grounded.
    """
    if os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq

        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        return ChatGroq(model=model, temperature=temperature)

    if os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)

    raise RuntimeError(
        "No LLM API key found. Set GROQ_API_KEY (https://console.groq.com) "
        "or GOOGLE_API_KEY (https://aistudio.google.com) in your .env file."
    )


# ----------------------------------------------------------------------------
# Step 5: wire retriever + LLM into a RetrievalQA chain
# ----------------------------------------------------------------------------
def build_qa_chain(vectorstore: FAISS, k: int = 4) -> RetrievalQA:
    """Create a RetrievalQA chain that returns the answer and its sources."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )


def ask(chain: RetrievalQA, question: str) -> dict:
    """Ask a question; returns {'answer': str, 'sources': list[Document]}."""
    result = chain.invoke({"query": question})
    return {
        "answer": result["result"].strip(),
        "sources": result.get("source_documents", []),
    }


# ----------------------------------------------------------------------------
# CLI entry point for quick testing
# ----------------------------------------------------------------------------
def main() -> None:
    import sys

    if len(sys.argv) < 2:
        print("Usage: python rag_chatbot.py <path-to-pdf>")
        raise SystemExit(1)

    print("Building index... (first run downloads the embedding model)")
    vs = build_vectorstore(sys.argv[1])
    chain = build_qa_chain(vs)
    print("Ready. Ask questions (Ctrl-C / empty line to quit).\n")

    try:
        while True:
            q = input("Q: ").strip()
            if not q:
                break
            out = ask(chain, q)
            print(f"A: {out['answer']}")
            pages = sorted({d.metadata.get("page", "?") for d in out["sources"]})
            print(f"   (sources: pages {pages})\n")
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")


if __name__ == "__main__":
    main()
