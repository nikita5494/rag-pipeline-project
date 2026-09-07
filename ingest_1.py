"""
RAG Pipeline: Document Ingestion
Loads PDFs, chunks them, embeds with HuggingFace (free, local), stores in ChromaDB
"""

import os
import sys
from pathlib import Path

# LangChain document loaders and text splitters
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings: HuggingFace (free, runs locally)
from langchain_community.embeddings import HuggingFaceEmbeddings

# Vector store
from langchain_community.vectorstores import Chroma

# Configuration
DOCS_PATH = "./docs"  # Folder containing your PDFs
CHROMA_PATH = "./chroma_db"  # Where to persist the vector store

def load_documents():
    """Load all PDF documents from the docs folder."""
    print(f"Loading documents from {DOCS_PATH}...")
    documents = []
    
    # Check if docs folder exists
    if not os.path.exists(DOCS_PATH):
        print(f"Error: {DOCS_PATH} folder not found!")
        print(f"Please create a '{DOCS_PATH}' folder and add your PDF files.")
        sys.exit(1)
    
    # Load all PDFs from docs folder
    pdf_files = list(Path(DOCS_PATH).glob("*.pdf"))
    if not pdf_files:
        print(f"Error: No PDF files found in {DOCS_PATH}/")
        print(f"Please add your PDF files to the '{DOCS_PATH}' folder.")
        sys.exit(1)
    
    print(f"Found {len(pdf_files)} PDF files")
    
    for pdf_file in pdf_files:
        print(f"  Loading {pdf_file.name}...")
        loader = PyPDFLoader(str(pdf_file))
        loaded_docs = loader.load()
        documents.extend(loaded_docs)
        print(f"    → {len(loaded_docs)} pages loaded")
    
    print(f"\nTotal pages loaded: {len(documents)}")
    return documents

def split_documents(documents):
    """Split documents into chunks for embedding."""
    print("\nSplitting documents into chunks...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,           # Size of each chunk in characters
        chunk_overlap=100,        # Overlap between chunks to preserve context
        separators=["\n\n", "\n", ".", " ", ""]  # Split on these in order
    )
    
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    # Show a sample chunk
    if chunks:
        print(f"\nSample chunk (first 200 chars):")
        print(f"  {chunks[0].page_content[:200]}...")
    
    return chunks

def create_vector_store(chunks):
    """Embed chunks and store in ChromaDB using HuggingFace embeddings."""
    print("\nCreating vector store...")
    
    # Initialize HuggingFace embeddings (free, runs locally)
    # First run will download the model (~350 MB), subsequent runs use cached model
    print("  Initializing HuggingFace embeddings (all-MiniLM-L6-v2)...")
    print("  First run may take 1–2 minutes to download the embedding model...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Fast, free, accurate
        # Alternative models if you want to experiment:
        # "sentence-transformers/all-mpnet-base-v2" (more accurate, slower)
        # "sentence-transformers/paraphrase-MiniLM-L6-v2" (good for semantic similarity)
    )
    
    print("  Embedding documents and storing in ChromaDB...")
    
    # Create Chroma vector store from documents
    # If persist_directory exists, this will overwrite it
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name="automotive_docs"
    )
    
    print(f"\n✓ Vector store created successfully!")
    print(f"  Location: {CHROMA_PATH}/")
    print(f"  Documents stored: {len(chunks)}")
    
    return db

def main():
    """Run the full ingestion pipeline."""
    print("=" * 60)
    print("AUTOMOTIVE RAG PIPELINE - DOCUMENT INGESTION")
    print("=" * 60)
    print(f"Using HuggingFace embeddings (free, local, no API key needed)")
    print(f"Vector store: ChromaDB at {CHROMA_PATH}/\n")
    
    # Step 1: Load documents
    documents = load_documents()
    
    # Step 2: Split into chunks
    chunks = split_documents(documents)
    
    # Step 3: Create vector store (embed and store)
    db = create_vector_store(chunks)
    
    print("\n" + "=" * 60)
    print("✓ INGESTION PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nYour vector store is ready for retrieval queries!")
    print()

if __name__ == "__main__":
    main()