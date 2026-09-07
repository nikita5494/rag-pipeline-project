"""
RAG Retrieval Chain
Query your automotive knowledge base
"""

import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
 # or use Anthropic/local LLM
from langchain_classic.chains import RetrievalQA

CHROMA_PATH = "./chroma_db"

def create_retrieval_chain():
    """Create a RAG chain that retrieves documents and answers questions."""
    
    # Load the vector store (created by ingest.py)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="automotive_docs"
    )
    
    # Create retriever
    retriever = db.as_retriever(search_kwargs={"k": 4})  # Return top 4 chunks
    
    # Create LLM (you can use OpenAI, Anthropic, or local LLM)
    llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Only 600MB, runs on CPU
    task="text-generation",
    # pipeline_kwargs={
    #     "device": -1,  # CPU only (-1), no GPU needed
    # }
    )
    
    # Create RAG chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Stuff all retrieved docs into context
        retriever=retriever,
        return_source_documents=True
    )
    
    return qa

def query_rag(question):
    """Query the RAG system."""
    qa = create_retrieval_chain()
    result = qa.invoke({"query": question})
    
    print(f"\n{'='*60}")
    print(f"Q: {question}")
    print(f"{'='*60}")
    print(f"\nA: {result['result']}")
    print(f"\n📚 Sources:")
    for i, doc in enumerate(result.get('source_documents', []), 1):
        print(f"  {i}. {doc.metadata.get('source', 'Unknown')} (page {doc.metadata.get('page', 'N/A')})")
    print()

if __name__ == "__main__":
    # Example queries
    queries = [
        "What is ASIL and how is it determined?",
        "Explain the difference between AUTOSAR Classic and Adaptive Platform",
        "What are the safety requirements for AEB?",
    ]
    
    for q in queries:
        query_rag(q)