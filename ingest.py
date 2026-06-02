
import os
os.environ["TIKTOKEN_CACHE_DIR"] = r"C:\\Users\\hardikan\\OneDrive - Bertrandt AG\\RAG_Pipeline_Project\\tiktoken_cache"

import os
from dotenv import load_dotenv

load_dotenv()

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma



# ---- CONFIG ----
DOCS_PATH = "./docs"
CHROMA_PATH = "./chroma_db"

# ---- STEP 1: Load PDFs ----
def load_documents():
    documents = []

    for file in os.listdir(DOCS_PATH):
        if file.endswith(".pdf"):
            file_path = os.path.join(DOCS_PATH, file)
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)

    return documents


# ---- STEP 2: Chunk documents ----
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)
    return chunks


# ---- STEP 3: Create embeddings ----
def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    db.persist()
    print("✅ Vector store created and persisted.")


# ---- MAIN PIPELINE ----
if __name__ == "__main__":
    print("Loading documents...")
    documents = load_documents()

    print(f"Loaded {len(documents)} pages")

    print("Splitting documents...")
    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print("Creating vector store...")
    create_vector_store(chunks)

    print("✅ Ingestion complete!")