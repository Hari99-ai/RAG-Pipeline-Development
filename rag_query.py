#!/usr/bin/env python3
"""
rag_query.py
-----------------------------------
Usage:
  python rag_query.py --question "Explain the steps to solve a quadratic equation."
-----------------------------------
This script:
✅ Connects to Qdrant (or local fallback index)
✅ Retrieves top-matching text chunks
✅ Uses Ollama LLM (llama3.2) to generate an answer
"""

import argparse
import importlib
import sys
import os
import json
import numpy as np
from typing import List
from sklearn.neighbors import NearestNeighbors
from qdrant_client import QdrantClient

# Prevent TensorFlow from auto-loading inside Transformers
os.environ["TRANSFORMERS_NO_TF"] = "1"


# -------------------------------
# Import Ollama integrations
# -------------------------------
try:
    from langchain_ollama import OllamaEmbeddings, OllamaLLM
    print("✅ Imported OllamaEmbeddings and OllamaLLM from langchain_ollama")
except ImportError:
    print("⚠️ langchain_ollama not found, using langchain_community instead...")
    try:
        from langchain_community.embeddings import OllamaEmbeddings
        from langchain_community.llms import Ollama as OllamaLLM
        print("✅ Imported from langchain_community.*")
    except ImportError as e:
        raise ImportError(
            "❌ Required modules missing.\n"
            "Install them using:\n"
            "pip install langchain-ollama langchain-community qdrant-client"
        ) from e


# -------------------------------
# Import Qdrant vectorstore
# -------------------------------
QdrantVectorstore = None
for mod in ["langchain_community.vectorstores", "langchain_qdrant"]:
    try:
        module = importlib.import_module(mod)
        QdrantVectorstore = getattr(module, "Qdrant")
        print(f"✅ Using Qdrant vectorstore from: {mod}")
        break
    except Exception:
        continue

if QdrantVectorstore is None:
    raise ImportError("❌ Qdrant vectorstore not found. Install with: pip install langchain-community qdrant-client")


# -------------------------------
# Local Fallback Index
# -------------------------------
LOCAL_INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")


def init_qdrant_vectorstore(collection_name="multimodal_docs",
                            url="http://localhost:6333",
                            embedding_model="nomic-embed-text"):
    """Initialize Qdrant or fallback to a local index."""
    print(f"🔗 Connecting to Qdrant collection '{collection_name}' at {url}")

    embeddings = OllamaEmbeddings(model=embedding_model)

    try:
        client = QdrantClient(url=url)
        client.get_collections()  # check connectivity
        vectorstore = QdrantVectorstore(client=client, collection_name=collection_name, embeddings=embeddings)
        print("✅ Connected to Qdrant successfully.")
        return vectorstore

    except Exception as e:
        print(f"❌ Cannot connect to Qdrant: {e}")
        print("Attempting local fallback index...")

        emb_path = os.path.join(LOCAL_INDEX_DIR, "embeddings.npy")
        docs_path = os.path.join(LOCAL_INDEX_DIR, "docs.json")

        if not (os.path.exists(emb_path) and os.path.exists(docs_path)):
            print(f"❌ Local index not found.\nExpected:\n  {emb_path}\n  {docs_path}")
            print("\nFix Options:\n"
                  "1️⃣ Start Qdrant: docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest\n"
                  "2️⃣ Export local embeddings and docs\n")
            sys.exit(1)

        vectors = np.load(emb_path)
        with open(docs_path, "r", encoding="utf-8") as f:
            docs = json.load(f)

        nn = NearestNeighbors(n_neighbors=min(8, len(docs)), metric="cosine")
        nn.fit(vectors)

        class SimpleDoc:
            def __init__(self, text, meta):
                self.page_content = text
                self.metadata = meta

        class LocalRetriever:
            def __init__(self, embeddings_obj, nn_index, vectors_array, docs_list, top_k=4):
                self.embeddings_obj = embeddings_obj
                self.nn = nn_index
                self.vectors = vectors_array
                self.docs = docs_list
                self.top_k = top_k

            def get_relevant_documents(self, query):
                qvec = np.array(self.embeddings_obj.embed_query(query)).reshape(1, -1)
                _, idxs = self.nn.kneighbors(qvec, n_neighbors=min(self.top_k, len(self.docs)))
                results = []
                for i in idxs[0]:
                    d = self.docs[i]
                    text = d.get("text", "")
                    meta = d.get("meta", {})
                    results.append(SimpleDoc(text, meta))
                return results

        class LocalVectorStore:
            def __init__(self, retriever):
                self.retriever = retriever

            def as_retriever(self, **kwargs):
                return self.retriever

        retriever = LocalRetriever(embeddings, nn, vectors, docs)
        print("✅ Local fallback index loaded (in-memory retriever).")
        return LocalVectorStore(retriever)


# -------------------------------
# Helper Functions
# -------------------------------
def build_retriever(vectorstore, k=4):
    try:
        return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    except TypeError:
        return vectorstore.as_retriever(k=k)


def simple_context_summary(docs, max_chars=1200):
    snippets = []
    for d in docs[:4]:
        src = d.metadata.get("source", "unknown")
        snippet = d.page_content[:400].rsplit(".", 1)[0] + "."
        snippets.append(f"Source: {src}\n{snippet}")
    return "\n\n".join(snippets)[:max_chars]


# -------------------------------
# Main RAG Query Function
# -------------------------------
def query_rag(question, retriever, llm_model="llama3.2:latest"):
    print(f"🧠 Asking: {question}")
    docs = retriever.get_relevant_documents(question)
    if not docs:
        print("❌ No relevant documents found.")
        return

    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"""Use the following context to answer the question clearly.

Context:
{context}

Question: {question}
"""

    try:
        llm = OllamaLLM(model=llm_model, temperature=0.2)
    except Exception as e:
        print(f"❌ Failed to initialize Ollama model: {e}")
        print(f"Try: ollama pull {llm_model}")
        print("\n--- Fallback (context only) ---\n")
        print(simple_context_summary(docs))
        return

    try:
        result = llm.invoke(prompt)
        print("\n=== FINAL ANSWER ===\n")
        print(result.strip())
        print("\n=== CONTEXT SOURCES ===")
        for d in docs:
            print(f"- {d.metadata.get('source', 'unknown')}: {d.page_content[:150]}...")
    except Exception as e:
        print("❌ LLM inference error:", e)
        print("\n--- Fallback (context only) ---\n")
        print(simple_context_summary(docs))


# -------------------------------
# CLI Entrypoint
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="RAG query pipeline using Ollama + Qdrant or local fallback index.")
    parser.add_argument("--question", required=False, help="Question to ask the RAG system.")
    parser.add_argument("--collection", default="multimodal_docs")
    parser.add_argument("--embedding_model", default="nomic-embed-text")
    parser.add_argument("--llm_model", default="llama3.2:latest")  # ✅ Default Llama 3.2 model
    parser.add_argument("--qdrant_url", default="http://localhost:6333")
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    if not args.question:
        args.question = input("Enter your question: ").strip()

    vectorstore = init_qdrant_vectorstore(
        collection_name=args.collection,
        url=args.qdrant_url,
        embedding_model=args.embedding_model
    )

    retriever = build_retriever(vectorstore, k=args.k)
    query_rag(args.question, retriever, llm_model=args.llm_model)


if __name__ == "__main__":
    main()
