#!/usr/bin/env python3
"""
rag_query.py

Examples:
  python rag_query.py --question "Explain the steps to solve a quadratic equation." --collection maths_grade_10
  python rag_query.py --question "What does the trapezoid diagram show?" --collection maths_grade_10 --summarize
"""

import argparse
import os
import json
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import Ollama
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from qdrant_client import QdrantClient
from langchain.cache import InMemoryCache


# ---------------------------
# Qdrant Initialization
# ---------------------------
def init_qdrant_vectorstore(collection_name="multimodal_docs",
                            url="http://localhost:6333",
                            embedding_model="nomic-embed-text"):
    """Initialize Qdrant vector store connection."""
    print(f"🔗 Connecting to Qdrant collection: {collection_name}")
    embeddings = OllamaEmbeddings(model=embedding_model)
    client = QdrantClient(url=url)
    vectorstore = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings,
        prefer_grpc=False
    )
    print("✅ Qdrant connection successful.")
    return vectorstore


# ---------------------------
# LLM Initialization
# ---------------------------
def build_rag_chain(llm_model="llama3", temperature=0.2, use_memory=False):
    """Initialize Ollama LLM with optional conversational memory."""
    print(f"🧠 Loading Ollama model: {llm_model}")
    llm = Ollama(model=llm_model, temperature=temperature)
    memory = ConversationBufferMemory(return_messages=True) if use_memory else None
    return llm, memory


# ---------------------------
# Context Summarization
# ---------------------------
def summarize_context(llm, retrieved_texts):
    """Summarize retrieved chunks before answer generation."""
    PROMPT = """Summarize the following extracted academic content 
in 3–5 concise sentences, capturing only the key mathematical or conceptual ideas.

{context}"""
    prompt = PromptTemplate(template=PROMPT, input_variables=["context"])
    input_text = "\n\n---\n\n".join(retrieved_texts)

    print("\n📄 Generating short summary of retrieved context...")
    response = llm.generate(
        [{"role": "user", "content": prompt.format(context=input_text)}]
    )
    summary = response.generations[0][0].text.strip()
    return summary


# ---------------------------
# Main Function
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Run a RAG query against indexed educational PDF content.")
    parser.add_argument("--question", required=True, help="Question to ask the RAG system.")
    parser.add_argument("--collection", default="multimodal_docs", help="Qdrant collection name.")
    parser.add_argument("--ollama_embedding_model", default="nomic-embed-text")
    parser.add_argument("--ollama_llm_model", default="llama3")
    parser.add_argument("--summarize", action="store_true", help="Enable summarization of retrieved context.")
    parser.add_argument("--use_memory", action="store_true", help="Enable conversational memory.")
    parser.add_argument("--cache", action="store_true", help="Enable simple prompt caching.")
    args = parser.parse_args()

    # Initialize vector store
    vectorstore = init_qdrant_vectorstore(
        collection_name=args.collection,
        embedding_model=args.ollama_embedding_model
    )

    # Initialize LLM
    llm, memory = build_rag_chain(
        llm_model=args.ollama_llm_model,
        use_memory=args.use_memory
    )

    # Optional caching
    if args.cache:
        print("⚙️ Enabling in-memory prompt cache...")
        cache = InMemoryCache()

    # Retrieval
    print(f"\n🔍 Retrieving most relevant chunks for: {args.question}")
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    retrieved_docs = retriever.get_relevant_documents(args.question)
    retrieved_texts = [d.page_content for d in retrieved_docs]

    if not retrieved_docs:
        print("❌ No documents retrieved. Ensure the collection is indexed correctly.")
        return

    # Optional summarization
    summary = None
    if args.summarize:
        summary = summarize_context(llm, retrieved_texts)
        print("\n=== Retrieved Context Summary ===\n")
        print(summary)
        print("\n=== End Summary ===\n")

    # Combine context
    assembled_context = "\n\n".join([
        f"Source (page {d.metadata.get('page', '?')}):\n{d.page_content}"
        for d in retrieved_docs
    ])

    if summary:
        prompt_text = (
            f"Use the following summary and retrieved contexts to answer the question.\n\n"
            f"Summary:\n{summary}\n\nContext:\n{assembled_context}\n\n"
            f"Question: {args.question}\n\n"
            "Answer concisely and include source page citations."
        )
    else:
        prompt_text = (
            f"Use the retrieved contexts to answer the question.\n\n"
            f"Context:\n{assembled_context}\n\n"
            f"Question: {args.question}\n\n"
            "Answer concisely and include source page citations."
        )

    # Generate final answer
    print("\n💬 Generating final answer from LLM...\n")
    response = llm.generate([{"role": "user", "content": prompt_text}])
    answer = response.generations[0][0].text.strip()

    print("\n=== Final RAG Answer ===\n")
    print(answer)
    print("\n=== Sources ===\n")
    for d in retrieved_docs:
        src = d.metadata.get('source', 'Unknown source')
        page = d.metadata.get('page', '?')
        print(f"- {src} (page {page}) snippet: {d.page_content[:180]}...")


if __name__ == "__main__":
    main()
