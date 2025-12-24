import streamlit as st
from dotenv import load_dotenv
from query import rag_chain, retriever

load_dotenv()

st.set_page_config(page_title="Legal Chatbot", layout="wide")
st.title(" Legal Document Assistant")

st.info("This tool provides legal information only. It is not legal advice.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Clear chat
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []

# Display chat
for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])

# Input
user_question = st.chat_input("Ask a legal question")

if user_question:
    st.chat_message("user").write(user_question)

    with st.spinner("Analyzing legal documents..."):
        answer = rag_chain.invoke({
            "question": user_question,
            "chat_history": st.session_state.chat_history
        })

    st.chat_message("assistant").write(answer)

    st.session_state.chat_history.append(
        {"role": "user", "content": user_question}
    )
    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer}
    )

# Sources
with st.expander("📚 Show Legal Sources"):
    if user_question:
        docs = retriever.invoke(user_question)

        for i, doc in enumerate(docs, 1):
            st.markdown(f"**Source {i}:**")
            st.write(doc.page_content[:500] + "...")
            st.divider()
