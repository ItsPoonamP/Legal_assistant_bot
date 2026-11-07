import os
import pickle
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

DATA_DIR = "data"
DB_DIR = "faiss_index"

# Load markdown files
loader = DirectoryLoader(
    DATA_DIR,
    glob="**/*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
documents = loader.load()
print(f"📄 Documents loaded: {len(documents)}")

if not documents:
    raise ValueError("❌ No markdown files found in data/")

# Tag each document with category based on filename
for doc in documents:
    filename = doc.metadata.get("source", "").lower()
    if "contract" in filename:
        doc.metadata["category"] = "contract_law"
    elif "criminal" in filename or "ipc" in filename:
        doc.metadata["category"] = "criminal_law"
    elif "ip" in filename or "patent" in filename:
        doc.metadata["category"] = "ip_law"
    else:
        doc.metadata["category"] = "general"

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " "]
)
chunks = text_splitter.split_documents(documents)
print(f"✂️ Chunks created: {len(chunks)}")

# Save chunks for BM25
with open("bm25_chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)
print("✅ BM25 chunks saved")

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create and save FAISS index
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local(DB_DIR)
print("✅ FAISS index created successfully")
