#!/usr/bin/env python3
"""
Generate local index (embeddings.npy + docs.json) for rag_query.py fallback.

Usage:
  python generate_local_index.py --index-dir d:\all\multimodal_rag_pipeline\index --model nomic-embed-text
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ["USE_TF"] = "0"
os.environ["HF_HUB_DISABLE_TF"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import numpy as np

from langchain_huggingface import HuggingFaceEmbeddings


def load_docs(docs_path: Path):
    if not docs_path.exists():
        raise FileNotFoundError(f"docs.json not found at: {docs_path}")
    with docs_path.open("r", encoding="utf-8") as fh:
        docs = json.load(fh)
    if not isinstance(docs, list):
        raise ValueError("docs.json must contain a JSON list of documents (each a dict).")
    texts = []
    for d in docs:
        if isinstance(d, dict):
            text = d.get("text") or d.get("page_content") or d.get("content") or ""
            texts.append(text)
        else:
            texts.append(str(d))
    return docs, texts


def main():
    parser = argparse.ArgumentParser(description="Generate local embedding index for RAG fallback")
    parser.add_argument("--index-dir", type=Path, default=Path(__file__).resolve().parent / "index")
    parser.add_argument("--docs-file", type=str, default="docs.json")
    parser.add_argument("--emb-file", type=str, default="embeddings.npy")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="HuggingFace embedding model")
    args = parser.parse_args()

    index_dir: Path = args.index_dir
    index_dir.mkdir(parents=True, exist_ok=True)
    docs_path = index_dir / args.docs_file
    emb_path = index_dir / args.emb_file

    try:
        docs, texts = load_docs(docs_path)
    except Exception as e:
        print("❌ Failed to load docs.json:", e, file=sys.stderr)
        sys.exit(3)

    if len(texts) == 0:
        print("❌ No documents found in docs.json", file=sys.stderr)
        sys.exit(4)

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedder = HuggingFaceEmbeddings(model_name=args.model, model_kwargs={"device": device})
    except Exception as e:
        print("Failed to instantiate HuggingFaceEmbeddings:", e, file=sys.stderr)
        sys.exit(5)

    # embed in batches to avoid memory spikes
    vectors = []
    BATCH = 64
    try:
        for i in range(0, len(texts), BATCH):
            batch = texts[i : i + BATCH]
            vecs = embedder.embed_documents(batch)
            # embed_documents may return nested lists or numpy arrays
            vecs = np.asarray(vecs)
            if vecs.ndim == 1:
                vecs = vecs.reshape(1, -1)
            vectors.append(vecs)
        vectors = np.vstack(vectors).astype(np.float32)
    except Exception as e:
        print("Embedding generation failed:", e, file=sys.stderr)
        sys.exit(6)

    try:
        np.save(emb_path, vectors)
        # save docs unchanged for loader compatibility
        with (index_dir / args.docs_file).open("w", encoding="utf-8") as fh:
            json.dump(docs, fh, ensure_ascii=False, indent=2)
    except Exception as e:
        print("Failed to save index files:", e, file=sys.stderr)
        sys.exit(7)

    print(f"[OK] Generated {vectors.shape[0]} embeddings (dim={vectors.shape[1]})")
    print(f"[File] Saved embeddings to: {emb_path}")
    print(f"[File] Saved docs to: {index_dir / args.docs_file}")


if __name__ == "__main__":
    main()