#!/usr/bin/env python3
"""
rag_query.py — Adaptive RAG Query Script with .env model priority
Author: Hari Om

Usage:
  python rag_query.py --question "Explain the steps to solve a quadratic equation."

Features:
- Always prioritizes model from .env (OLLAMA_MODEL)
- Falls back to smaller Ollama models only if needed
- Works with Qdrant Cloud, local Qdrant, or local in-memory fallback
- Compatible with both langchain_ollama and langchain_community
"""

import argparse
import importlib
import os
import sys
import json
import psutil
import numpy as np
from typing import List
from sklearn.neighbors import NearestNeighbors
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# =====================================
# ENVIRONMENT SETUP
# =====================================
os.environ["TRANSFORMERS_NO_TF"] = "1"
load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "tinyllama")  # always prioritize .env model

# Detect total RAM
def get_total_ram_gb() -> float:
    try:
        return round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except Exception:
        return 0.0

RAM_GB = get_total_ram_gb()
print(f"💾 System RAM detected: {RAM_GB} GB")

# =====================================
# IMPORT LLM + EMBEDDINGS
# =====================================
try:
    from langchain_ollama import OllamaEmbeddings, OllamaLLM
    print("✅ Using langchain_ollama for embeddings & LLM.")
except Exception:
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.llms import Ollama as OllamaLLM
    print("✅ Using langchain_community for embeddings & LLM.")

# =====================================
# IMPORT QDRANT
# =====================================
QdrantVectorstore = None
for mod in ("langchain_community.vectorstores", "langchain_qdrant"):
    try:
        m = importlib.import_module(mod)
        QdrantVectorstore = getattr(m, "Qdrant")
        print(f"✅ Using Qdrant vectorstore from: {mod}")
        break
    except Exception:
        continue

LOCAL_INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")
DEFAULT_COLLECTION = "multimodal_docs"

# =====================================
# INITIALIZE QDRANT OR LOCAL FALLBACK
# =====================================
def init_qdrant_vectorstore(collection_name=DEFAULT_COLLECTION, embedding_model="nomic-embed-text"):
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    print(f"🔗 Connecting to Qdrant: {qdrant_url}")
    embeddings = OllamaEmbeddings(model=embedding_model)

    try:
        if qdrant_api_key:
            client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            print("🌐 Using Qdrant Cloud (API key provided).")
        else:
            client = QdrantClient(url=qdrant_url)
            print("💻 Using local Qdrant.")
        client.get_collections()
        if QdrantVectorstore:
            return QdrantVectorstore(client=client, collection_name=collection_name, embeddings=embeddings)
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        print("➡️ Falling back to local in-memory index...")

    emb_path = os.path.join(LOCAL_INDEX_DIR, "embeddings.npy")
    docs_path = os.path.join(LOCAL_INDEX_DIR, "docs.json")

    if not (os.path.exists(emb_path) and os.path.exists(docs_path)):
        print("❌ Local index not found.")
        print(f"Expected files:\n  {emb_path}\n  {docs_path}")
        sys.exit(1)

    vectors = np.load(emb_path)
    with open(docs_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    nn = NearestNeighbors(n_neighbors=min(8, len(docs)), metric="cosine")
    nn.fit(vectors)

    class Doc:
        def __init__(self, text, meta):
            self.page_content = text
            self.metadata = meta

    class LocalRetriever:
        def __init__(self, emb, nn_index, vecs, doc_list, top_k=4):
            self.emb = emb
            self.nn = nn_index
            self.vecs = vecs
            self.docs = doc_list
            self.top_k = top_k
        def get_relevant_documents(self, query):
            qv = np.array(self.emb.embed_query(query)).reshape(1, -1)
            _, idxs = self.nn.kneighbors(qv, n_neighbors=min(self.top_k, len(self.docs)))
            return [Doc(self.docs[i]["text"], self.docs[i].get("meta", {})) for i in idxs[0]]

    class LocalVectorstore:
        def __init__(self, retriever): self.retriever = retriever
        def as_retriever(self, **kwargs): return self.retriever

    print("✅ Local fallback index loaded.")
    return LocalVectorstore(LocalRetriever(embeddings, nn, vectors, docs))

# =====================================
# HELPER FUNCTIONS
# =====================================
def simple_context_summary(docs, max_chars=1000):
    result = []
    for d in docs[:4]:
        src = d.metadata.get("source", "unknown")
        snippet = d.page_content[:300].replace("\n", " ")
        result.append(f"Source: {src}\n{snippet}")
    return "\n\n".join(result)[:max_chars]

# =====================================
# LOAD LLM (Prioritize .env model)
# =====================================
def load_llm_with_fallback(preferred_model: str):
    # Always prioritize .env model first
    models = [preferred_model, "phi3:mini", "tinyllama"]

    for m in models:
        try:
            print(f"🧠 Attempting to load model: {m} (base_url={OLLAMA_URL})")
            llm = OllamaLLM(model=m, temperature=0.2, base_url=OLLAMA_URL)
            print(f"✅ Loaded model: {m}")
            return llm, m
        except Exception as e:
            print(f"⚠️ Failed to load {m}: {e}")
            if "404" in str(e).lower():
                print(f"➡️ Run: ollama pull {m}")
            continue

    raise RuntimeError("❌ No LLM model could be loaded. Please pull tinyllama or phi3:mini.")

# =====================================
# MAIN QUERY FUNCTION
# =====================================
def query_rag(question: str, retriever, llm_model: str = OLLAMA_MODEL):
    print(f"\n🔎 Question: {question}")
    docs = retriever.get_relevant_documents(question)
    if not docs:
        print("❌ No relevant documents found.")
        return

    context = "\n\n".join(d.page_content for d in docs)
    prompt = f"""You are a clear, helpful tutor. Use the context below to answer.

Context:
{context}

Question: {question}

Answer:"""

    try:
        llm, model_name = load_llm_with_fallback(llm_model)
        print(f"💬 Generating answer using {model_name} ...")
        response = llm.invoke(prompt) if hasattr(llm, "invoke") else llm(prompt)
        print("\n=== FINAL ANSWER ===\n")
        print(response.strip() if isinstance(response, str) else str(response))
        print("\n=== SOURCES ===")
        for d in docs:
            print(f"- {d.metadata.get('source', 'unknown')}")
    except Exception as e:
        print(f"❌ LLM inference error: {e}")
        print("\n--- Context summary fallback ---\n")
        print(simple_context_summary(docs))

# =====================================
# CLI ENTRY
# =====================================
def main():
    parser = argparse.ArgumentParser(description="Run RAG query using Ollama + Qdrant.")
    parser.add_argument("--question", help="Question to ask the RAG system.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--embedding_model", default="nomic-embed-text")
    parser.add_argument("--llm_model", default=OLLAMA_MODEL)
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    if not args.question:
        args.question = input("Enter your question: ").strip()

    vectorstore = init_qdrant_vectorstore(args.collection, args.embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": args.k})
    query_rag(args.question, retriever, args.llm_model)

if __name__ == "__main__":
    main()
