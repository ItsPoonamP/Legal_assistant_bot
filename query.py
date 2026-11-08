import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import streamlit as st

from Hybrid_Retriver import HybridRetriever   # NEW

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

DB_DIR = "faiss_index"

# ------------------------------
# Retrieval settings (NEW)
# ------------------------------
USE_BM25 = True      # hybrid search: keyword (BM25) + semantic (FAISS)
USE_RERANK = True    # cross-encoder reranking of the fused shortlist
TOP_K = 4            # chunks sent to the LLM

# ==============================
# 1️⃣ Embeddings + FAISS + HYBRID RETRIEVER
# ==============================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    DB_DIR,
    embeddings,
    allow_dangerous_deserialization=True
)

# Same name as before, so `from query import rag_chain, retriever` in app.py
# keeps working, and `retriever.invoke(question)` still returns a list of Documents.
retriever = HybridRetriever(
    vectorstore,
    top_k=TOP_K,
    use_bm25=USE_BM25,
    use_rerank=USE_RERANK,
)

# ==============================
# 2️⃣ LLM (VALID MODEL)
# ==============================

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0,
    api_key=GROQ_API_KEY
)

# ==============================
# 3️⃣ Helper Functions (SAFE)
# ==============================

def get_question(input_):
    """Accepts string OR dict"""
    if isinstance(input_, str):
        return input_
    return input_["question"]

def get_history(input_):
    return input_.get("chat_history", [])

def format_history(history):
    if not history:
        return "No prior conversation."
    return "\n".join(
        f"{m['role'].capitalize()}: {m['content']}"
        for m in history
    )

def format_docs(docs):
    if not docs:
        return "NO RELEVANT LEGAL TEXT FOUND."
    return "\n\n".join(doc.page_content for doc in docs)

# ==============================
# 4️⃣ Prompt (LEGAL SAFE)
# ==============================

prompt = ChatPromptTemplate.from_template("""
You are a legal research assistant.

Rules:
- Answer ONLY from the provided legal context
- Use conversation history for follow-up questions
- If the answer is not found, say:
  "I don't know based on the provided legal documents."
- Do NOT provide legal advice

Conversation History:
{chat_history}

Legal Context:
{context}

Question:
{question}

Answer (include disclaimer):
""")

# ==============================
# 5️⃣ ✅ HISTORY-AWARE RAG CHAIN (now with hybrid retrieval + rerank)
# ==============================

rag_chain = (
    {
        "question": RunnableLambda(get_question),
        "context": RunnableLambda(get_question)
                   | RunnableLambda(retriever.retrieve)   # CHANGED: was `| retriever`
                   | format_docs,
        "chat_history": RunnableLambda(get_history) | format_history,
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ==============================
# 6️⃣ CLI TEST (OPTIONAL)
# ==============================

if __name__ == "__main__":
    print("⚖️ Legal RAG Assistant (type 'exit' to quit, '/debug <question>' to compare retrievers)")
    chat_history = []

    while True:
        query = input(">> ")

        if query.lower() == "exit":
            break

        # NEW: see what dense / BM25 / fused / reranked each returned
        if query.startswith("/debug "):
            for stage, items in retriever.debug(query[7:]).items():
                print(f"\n{stage}:")
                for t in items:
                    print("  -", t)
            print()
            continue

        response = rag_chain.invoke({
            "question": query,
            "chat_history": chat_history
        })

        print("\nAnswer:\n", response, "\n")

        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": response})
