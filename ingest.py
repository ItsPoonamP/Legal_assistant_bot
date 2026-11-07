import os
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
print(f"📄 Documents loaded: {len(documents)}")

if not documents:
    raise ValueError("❌ No markdown files found in data/")

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " "]  # tries these in order
)

chunks = text_splitter.split_documents(documents)
print(f"✂️ Chunks created: {len(chunks)}")

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS index
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local(DB_DIR)

print("✅ FAISS index created successfully")
import pickle

# Save chunks for BM25 (keyword search) — BM25 can't be saved like FAISS
import os

# Generate BM25 chunks on the fly if file doesn't exist
if not os.path.exists("bm25_chunks.pkl"):
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    loader = DirectoryLoader(
        "data",
        glob="**/*.md",
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
print("✅ BM25 chunks saved for hybrid search")


#from langchain_text_splitters import CharacterTextSplitter  from langchain_community.embeddings import HuggingFaceEmbeddings
