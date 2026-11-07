import os
import pickle

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.retrievers import EnsembleRetriever
from langchain_community.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_groq import ChatGroq

import streamlit as st
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

DB_DIR = "faiss_index"

# ==============================
# 1️⃣ Embeddings + FAISS
# ==============================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    DB_DIR,
    embeddings,
    allow_dangerous_deserialization=True
)

# ==============================
# 2️⃣ BM25 chunks (generate if missing)
# ==============================
if not os.path.exists("bm25_chunks.pkl"):
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    loader = DirectoryLoader(
        "data", glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    raw_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(raw_docs)
    with open("bm25_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
else:
    with open("bm25_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

# ==============================
# 3️⃣ Hybrid Retriever
# ==============================
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 4

faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

base_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.4, 0.6]
)

# ==============================
# 4️⃣ Cross-Encoder Re-ranker
# ==============================
cross_encoder = HuggingFaceCrossEncoder(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)
reranker = CrossEncoderReranker(model=cross_encoder, top_n=3)

retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=base_retriever
)

# ==============================
# 5️⃣ LLM
# ==============================
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0,
    api_key=GROQ_API_KEY
)

# ==============================
# 6️⃣ Helper Functions
# ==============================
def get_question(input_):
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
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown document")
        category = doc.metadata.get("category", "general")
        formatted.append(
            f"[Source: {source} | Category: {category}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)

def get_sources(docs):
    return list(set(doc.metadata.get("source", "Unknown") for doc in docs))

# ==============================
# 7️⃣ Query Rewriting
# ==============================
rewrite_prompt = ChatPromptTemplate.from_template("""
Given this conversation history and a follow-up question, 
rewrite the follow-up question to be completely self-contained.
If it's already self-contained, return it unchanged.

Conversation History:
{chat_history}

Follow-up Question: {question}

Rewritten Question (return ONLY the question, nothing else):
""")

rewrite_chain = rewrite_prompt | llm | StrOutputParser()

def rewrite_question(input_):
    history = input_.get("chat_history", [])
    question = get_question(input_)
    if not history:
        return question
    return rewrite_chain.invoke({
        "chat_history": format_history(history),
        "question": question
    })

# ==============================
# 8️⃣ Main Prompt
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
# 9️⃣ RAG Chain
# ==============================
rag_chain = (
    {
        "question": RunnableLambda(rewrite_question),
        "context": RunnableLambda(rewrite_question) | retriever | format_docs,
        "chat_history": RunnableLambda(get_history) | format_history,
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ==============================
# 🔟 Answer + Sources helper
# ==============================
def get_answer_with_sources(input_dict):
    rewritten_q = rewrite_question(input_dict)
    docs = retriever.invoke(rewritten_q)
    sources = get_sources(docs)
    answer = rag_chain.invoke(input_dict)
    return {"answer": answer, "sources": sources}

# ==============================
# CLI TEST (OPTIONAL)
# ==============================
if __name__ == "__main__":
    print("⚖️ Legal RAG Assistant (type 'exit' to quit)")
    chat_history = []
    while True:
        query = input(">> ")
        if query.lower() == "exit":
            break
        result = get_answer_with_sources({
            "question": query,
            "chat_history": chat_history
        })
        print("\nAnswer:\n", result["answer"])
        print("Sources:",
