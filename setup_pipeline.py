#!/usr/bin/env python3
"""
setup_pipeline.py
-------------------------------------
Usage:
    python setup_pipeline.py
    python setup_pipeline.py --collection maths_grade_10 --ollama_model nomic-embed-text
    python setup_pipeline.py --export_index
-------------------------------------
This script:
✅ Extracts text + images from PDF
✅ Splits text into chunks
✅ Embeds each chunk via Ollama
✅ Uploads to Qdrant or saves locally (if --export_index)
✅ Displays live progress visualization
"""
import argparse
import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import fitz  # PyMuPDF
from qdrant_client import QdrantClient, models as qmodels
from colorama import Fore, Style, init

# ---- Color Output Setup ----
init(autoreset=True)

# ---- Embeddings Import (with fallback) ----
try:
    from langchain_ollama import OllamaEmbeddings
    print(Fore.GREEN + "✅ Using OllamaEmbeddings from: langchain_ollama")
except Exception:
    from langchain_community.embeddings import OllamaEmbeddings
    print(Fore.YELLOW + "✅ Using OllamaEmbeddings from: langchain_community.embeddings")

# ---- Ollama Model Import ----
Ollama = None
for mod_path in (
    "langchain_ollama",
    "langchain_community.llms",
    "langchain_community.chat_models",
):
    try:
        mod = __import__(mod_path, fromlist=["Ollama"])
        Ollama = getattr(mod, "Ollama")
        print(Fore.GREEN + f"✅ Loaded Ollama from: {mod_path}")
        break
    except Exception:
        continue

if not Ollama:
    raise ImportError(Fore.RED + "❌ Could not import Ollama. Please install langchain-ollama or langchain-community.")

# ---- Parameters ----
CHUNK_SIZE = 800
CHUNK_OVERLAP = 128
IMAGE_OUTPUT_DIR = "extracted_images"
INDEX_DIR = "index"
DEFAULT_PDF_PATH = r"D:\all\multimodal_rag_pipeline\data\Maths_Grade_10.pdf"
QDRANT_URL = "http://localhost:6333"


# -------------------------------------------------------------
#                 PDF Extraction (Text + Images)
# -------------------------------------------------------------
def ensure_dirs():
    Path(IMAGE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)


def extract_pdf_multimodal(pdf_path):
    """Extract text and images from PDF pages."""
    print(Fore.CYAN + f"📘 Extracting text and images from: {pdf_path}")
    doc = fitz.open(pdf_path)
    blocks = []

    for pnum, page in enumerate(tqdm(doc, desc="🔍 Processing PDF pages", colour="cyan"), start=1):
        page_text = page.get_text("blocks")
        imgs = page.get_images(full=True)
        saved_images = []

        for img in imgs:
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n > 3:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img_name = f"page{pnum}_img{xref}.png"
                img_path = os.path.join(IMAGE_OUTPUT_DIR, img_name)
                pix.save(img_path)
                saved_images.append({"path": img_path})
                pix = None
            except Exception as e:
                print(Fore.YELLOW + f"⚠️ Skipping image {xref}: {e}")

        combined_text = "\n\n".join([b[4] for b in page_text if b[4].strip()])
        blocks.append({
            "page": pnum,
            "text": combined_text,
            "images": saved_images,
            "meta": {"source": os.path.basename(pdf_path), "page": pnum}
        })
    return blocks


# -------------------------------------------------------------
#                    Text Chunking
# -------------------------------------------------------------
def chunk_text(blocks, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks while keeping metadata."""
    print(Fore.CYAN + "✂️ Splitting text into chunks...")
    chunks = []
    for b in tqdm(blocks, desc="📄 Chunking pages", colour="green"):
        text = b["text"].strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            chunk_meta = dict(b["meta"])
            chunk_meta["images"] = [img["path"] for img in b["images"]]
            chunks.append({"text": chunk_text, "meta": chunk_meta})
            if end == len(text):
                break
            start = end - overlap
    return chunks


# -------------------------------------------------------------
#               Embedding + Qdrant or Local Export
# -------------------------------------------------------------
def embed_and_upsert(chunks, collection_name, ollama_model, export_index=False):
    """Embed text chunks and store in Qdrant or export locally."""
    print(Fore.CYAN + f"🔗 Connecting to Qdrant at {QDRANT_URL}...")

    try:
        client = QdrantClient(url=QDRANT_URL)
        client.get_collections()
        print(Fore.GREEN + "✅ Connected to external Qdrant instance (Docker).")
    except Exception:
        print(Fore.YELLOW + "⚠️ Qdrant not found, using in-memory mode.")
        client = QdrantClient(":memory:")

    embeddings = OllamaEmbeddings(model=ollama_model)
    vector_size = len(embeddings.embed_query("test"))
    print(Fore.BLUE + f"🧭 Detected embedding vector size: {vector_size}")

    try:
        client.get_collection(collection_name)
        print(Fore.CYAN + f"ℹ️ Using existing collection: {collection_name}")
    except Exception:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE)
        )
        print(Fore.GREEN + f"🆕 Created collection: {collection_name} (size={vector_size})")

    vectors, payloads = [], []
    print(Fore.CYAN + f"🔁 Embedding & upserting {len(chunks)} chunks...")
    for i, chunk in enumerate(tqdm(chunks, desc="⚙️ Embedding progress", colour="magenta")):
        vector = embeddings.embed_query(chunk["text"])
        payload = {"text": chunk["text"], **chunk["meta"]}
        vectors.append(vector)
        payloads.append(payload)
        client.upsert(
            collection_name=collection_name,
            points=[qmodels.PointStruct(id=i, vector=vector, payload=payload)]
        )

    print(Fore.GREEN + f"✅ Upsert complete. {len(chunks)} chunks added to {collection_name}.")

    # --- Local export ---
    if export_index:
        print(Fore.CYAN + "💾 Exporting local index files...")
        np.save(os.path.join(INDEX_DIR, "embeddings.npy"), np.array(vectors))
        with open(os.path.join(INDEX_DIR, "docs.json"), "w", encoding="utf-8") as f:
            json.dump(payloads, f, ensure_ascii=False, indent=2)
        print(Fore.GREEN + "✅ Local index exported to /index/embeddings.npy and /index/docs.json")


# -------------------------------------------------------------
#                        Main Entry
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default=DEFAULT_PDF_PATH)
    parser.add_argument("--collection", default="multimodal_docs")
    parser.add_argument("--ollama_model", default="nomic-embed-text")
    parser.add_argument("--export_index", action="store_true", help="Save local index for offline queries")
    args = parser.parse_args()

    ensure_dirs()

    print(Fore.YELLOW + "\n🚀 Starting Multimodal RAG Setup Pipeline...\n" + Style.RESET_ALL)

    blocks = extract_pdf_multimodal(args.pdf)
    chunks = chunk_text(blocks)
    print(Fore.CYAN + f"📄 Extracted {len(blocks)} pages → {len(chunks)} chunks total.")

    embed_and_upsert(chunks, args.collection, args.ollama_model, export_index=args.export_index)

    print(Fore.GREEN + "\n🎯 Pipeline complete! Ready for querying.\n")


if __name__ == "__main__":
    main()
