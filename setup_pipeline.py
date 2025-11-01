#!/usr/bin/env python3
"""
setup_pipeline.py
Usage:
    python setup_pipeline.py
    # (Optional) Override defaults:
    # python setup_pipeline.py --collection maths_grade_10 --ollama_model nomic-embed-text
"""

import argparse
import os
from pathlib import Path
import fitz  # PyMuPDF
from qdrant_client import QdrantClient, models as qmodels

# --- Updated imports for LangChain v0.3+ ---
from langchain_community.embeddings import OllamaEmbeddings

# --- Robust Ollama Import Handling ---
Ollama = None
_import_errors = []

for module_path in (
    "langchain_ollama.llms",
    "langchain_ollama.llm",
    "langchain_ollama",
    "langchain_community.chat_models",
    "langchain_community.llms",
    "ollama",
):
    try:
        mod = __import__(module_path, fromlist=["Ollama"])
        Ollama = getattr(mod, "Ollama")
        print(f"✅ Successfully imported Ollama from: {module_path}")
        break
    except Exception as e:
        _import_errors.append(f"{module_path}: {e!r}")

if Ollama is None:
    raise ImportError(
        "❌ Cannot locate 'Ollama' class in any known locations.\n"
        "Checked: langchain_ollama.llms, langchain_ollama.llm, langchain_ollama, "
        "langchain_community.chat_models, langchain_community.llms, ollama.\n\n"
        "Import errors:\n" + "\n".join(_import_errors) +
        "\n\nTo fix this, check installed versions:\n"
        "  python -m pip show langchain langchain-core langchain-ollama ollama\n\n"
        "If mismatched, reinstall compatible versions:\n"
        "  python -m pip install --force-reinstall langchain==0.3.3 langchain-core==0.3.10 langchain-community==0.3.2 langchain-ollama"
    )

# -------- Parameters ----------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 128
IMAGE_OUTPUT_DIR = "extracted_images"
DEFAULT_PDF_PATH = r"D:\all\multimodal_rag_pipeline\data\Maths_Grade_10.pdf"
# -----------------------------


def ensure_dirs():
    Path(IMAGE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def extract_pdf_multimodal(pdf_path):
    """Extracts text and images from PDF pages safely (handles CMYK / alpha)."""
    print(f"📘 Processing PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    blocks = []

    for pnum, page in enumerate(doc, start=1):
        page_text = page.get_text("blocks")
        imgs = page.get_images(full=True)
        saved_images = []

        for img_index, img in enumerate(imgs):
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)

                # ✅ Convert CMYK or alpha-channel images to RGB
                if pix.n > 3:
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                img_name = f"page{pnum}_img{xref}.png"
                img_path = os.path.join(IMAGE_OUTPUT_DIR, img_name)
                pix.save(img_path)
                saved_images.append({"path": img_path, "xref": xref})
                pix = None
            except Exception as e:
                print(f"⚠️ Skipping image {xref} on page {pnum}: {e}")

        combined_text = "\n\n".join([b[4] for b in page_text if b[4].strip()])
        blocks.append({
            "page": pnum,
            "text": combined_text,
            "images": saved_images,
            "meta": {"source": os.path.basename(pdf_path), "page": pnum}
        })

    return blocks


def chunk_text(blocks, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Chunk text with overlap; keep image metadata attached."""
    chunks = []
    for b in blocks:
        text = b["text"].strip()
        if not text:
            continue
        start = 0
        L = len(text)
        while start < L:
            end = min(start + chunk_size, L)
            chunk_text = text[start:end].strip()
            chunk_meta = dict(b["meta"])
            chunk_meta["page"] = b["page"]
            chunk_meta["images"] = [img["path"] for img in b["images"]]
            chunks.append({"text": chunk_text, "meta": chunk_meta})
            start = end - overlap
            if start < 0:
                start = 0
            if end == L:
                break
    return chunks


def embed_and_upsert(chunks, collection_name="default_collection", ollama_model="nomic-embed-text"):
    """Embed text chunks and upsert into Qdrant."""
    embeddings = OllamaEmbeddings(model=ollama_model)
    client = QdrantClient(url="http://localhost:6333")

    vector_size = getattr(embeddings, "model_output_dimension", 1536)
    try:
        client.get_collection(collection_name)
    except Exception:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE)
        )

    BATCH = 64
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        texts = [c["text"] for c in batch]
        vectors = embeddings.embed_documents(texts)
        points = []
        for idx, vec in enumerate(vectors):
            payload = {"text": batch[idx]["text"], **batch[idx]["meta"]}
            points.append(qmodels.PointStruct(id=i + idx, vector=vec, payload=payload))
        client.upsert(collection_name=collection_name, points=points)

    print(f"✅ Qdrant collection '{collection_name}' upserted with {len(chunks)} chunks.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default=DEFAULT_PDF_PATH, help="Path to PDF file")
    parser.add_argument("--collection", default="multimodal_docs", help="Qdrant collection name")
    parser.add_argument("--ollama_model", default="nomic-embed-text", help="Embedding model for Ollama")
    args = parser.parse_args()

    ensure_dirs()
    blocks = extract_pdf_multimodal(args.pdf)
    chunks = chunk_text(blocks)
    print(f"📄 Extracted {len(blocks)} page blocks → {len(chunks)} chunks.")
    embed_and_upsert(chunks, collection_name=args.collection, ollama_model=args.ollama_model)


if __name__ == "__main__":
    main()
