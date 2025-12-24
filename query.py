import os


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
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

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

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
# 5️⃣ ✅ HISTORY-AWARE RAG CHAIN
# ==============================

rag_chain = (
    {
        "question": RunnableLambda(get_question),
        "context": RunnableLambda(get_question) | retriever | format_docs,
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
    print("⚖️ Legal RAG Assistant (type 'exit' to quit)")
    chat_history = []

    while True:
        query = input(">> ")
        if query.lower() == "exit":
            break

        response = rag_chain.invoke({
            "question": query,
            "chat_history": chat_history
        })

        print("\nAnswer:\n", response, "\n")

        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": response})



