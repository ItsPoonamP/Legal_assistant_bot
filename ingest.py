import os
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
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

# Split documents
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
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



#from langchain_text_splitters import CharacterTextSplitter  from langchain_community.embeddings import HuggingFaceEmbeddings