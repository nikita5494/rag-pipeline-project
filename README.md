
# 🚀 RAG Pipeline Project

This repository contains a basic implementation of a **Retrieval Augmented Generation (RAG) pipeline** using LangChain, OpenAI embeddings, and ChromaDB.

The goal of this project is to build a system that can:
- Ingest PDF documents
- Store them as embeddings in a vector database
- Enable question-answering over the stored knowledge

---

## 📂 Project Structure

rag-pipeline-project/
│
├── ingest.py            # Ingestion pipeline (load → chunk → embed → store)
├── requirements.txt     # Project dependencies
├── docs/                # Source PDFs
│   ├── *.pdf

---

## ⚙️ Current Functionality

### ✅ Step 1–3: Data Ingestion Pipeline

The project currently implements:

1. **Document Loading**
   - PDFs are loaded using `PyPDFLoader`

2. **Text Chunking**
   - Uses `RecursiveCharacterTextSplitter`
   - `chunk_size = 800`
   - `chunk_overlap = 100`

3. **Embedding Generation**
   - Uses OpenAI embeddings (`text-embedding-3-small`)

4. **Vector Storage**
   - Stores embeddings in a local **ChromaDB** database

---

## ▶️ How to Run

### 1. Clone or download the repository

```bash
git clone <your-repo-url>
cd rag-pipeline-project
