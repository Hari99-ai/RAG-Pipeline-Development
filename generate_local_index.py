#!/usr/bin/env python3
"""
Generate local index (embeddings.npy + docs.json) for rag_query.py fallback.

Usage:
  python generate_local_index.py --index-dir d:\all\multimodal_rag_pipeline\index --model nomic-embed-text
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# try known Ollama embeddings locations
OllamaEmbeddings = None
_import_errs = []
for mod_path in ("langchain_ollama", "langchain_community.embeddings", "langchain_community.embeddings.ollama"):
    try:
        mod = __import__(mod_path, fromlist=["OllamaEmbeddings"])
        OllamaEmbeddings = getattr(mod, "OllamaEmbeddings")
        break
    except Exception as e:
        _import_errs.append(f"{mod_path}: {e!r}")

if OllamaEmbeddings is None:
    print("❌ Cannot find OllamaEmbeddings. Tried:\n" + "\n".join(_import_errs), file=sys.stderr)
    print("Install a compatible package set, e.g.:")
    print("  python -m pip install --force-reinstall langchain-ollama langchain-core langchain")
    sys.exit(2)


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
    parser.add_argument("--model", type=str, default="nomic-embed-text", help="Ollama embedding model")
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
        embedder = OllamaEmbeddings(model=args.model)
    except Exception as e:
        print("❌ Failed to instantiate OllamaEmbeddings:", e, file=sys.stderr)
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
        print("❌ Embedding generation failed:", e, file=sys.stderr)
        sys.exit(6)

    try:
        np.save(emb_path, vectors)
        # save docs unchanged for loader compatibility
        with (index_dir / args.docs_file).open("w", encoding="utf-8") as fh:
            json.dump(docs, fh, ensure_ascii=False, indent=2)
    except Exception as e:
        print("❌ Failed to save index files:", e, file=sys.stderr)
        sys.exit(7)

    print(f"✅ Generated {vectors.shape[0]} embeddings (dim={vectors.shape[1]})")
    print(f"📁 Saved embeddings to: {emb_path}")
    print(f"📁 Saved docs to: {index_dir / args.docs_file}")


if __name__ == "__main__":
    main()