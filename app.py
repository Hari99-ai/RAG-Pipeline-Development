#!/usr/bin/env python3
"""
app.py - Streamlit UI for the Multimodal RAG Pipeline
=====================================================
LLM: Groq API  |  Embeddings: Ollama (nomic-embed-text)  |  Vector Store: Qdrant / Local
Run with:  streamlit run app.py
"""

import os
import sys
import json
import tempfile
import importlib

os.environ["USE_TF"] = "0"
os.environ["HF_HUB_DISABLE_TF"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import streamlit as st
import numpy as np
from sklearn.neighbors import NearestNeighbors
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels
import fitz  # PyMuPDF
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_INDEX_DIR = os.path.join(BASE_DIR, "index")
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGE_OUTPUT_DIR = os.path.join(BASE_DIR, "extracted_images")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 128
DEFAULT_COLLECTION = "multimodal_docs"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Groq models available for chat
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

# ---------------------------------------------------------------------------
# Embeddings Import (HuggingFace)
# ---------------------------------------------------------------------------
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------------------------------------------------------
# Groq LLM Import
# ---------------------------------------------------------------------------
from langchain_groq import ChatGroq

# ---------------------------------------------------------------------------
# Qdrant Vectorstore Import
# ---------------------------------------------------------------------------
QdrantVectorstore = None
for _mod in ("langchain_community.vectorstores", "langchain_qdrant"):
    try:
        _m = importlib.import_module(_mod)
        QdrantVectorstore = getattr(_m, "Qdrant")
        break
    except Exception:
        continue

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def check_ollama_health():
    """Check if Ollama is running."""
    try:
        import urllib.request
        url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        req = urllib.request.Request(f"{url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def check_qdrant_health():
    """Check if Qdrant is reachable."""
    try:
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        client = QdrantClient(url=url)
        client.get_collections()
        return True
    except Exception:
        return False


def check_groq_health():
    """Check if Groq API key is configured and working."""
    if not GROQ_API_KEY:
        return False
    try:
        llm = ChatGroq(model=GROQ_MODELS[0], api_key=GROQ_API_KEY, max_tokens=5)
        llm.invoke("Hi")
        return True
    except Exception:
        return False


def ollama_list_models():
    """Return list of locally available Ollama model names."""
    try:
        import subprocess
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return []
        lines = result.stdout.strip().split("\n")
        models = []
        for line in lines[1:]:
            parts = line.split()
            if parts:
                models.append(parts[0])
        return models
    except Exception:
        return []


def get_local_index_stats():
    """Return stats about the local index."""
    emb_path = os.path.join(LOCAL_INDEX_DIR, "embeddings.npy")
    docs_path = os.path.join(LOCAL_INDEX_DIR, "docs.json")
    stats = {"embeddings_exists": False, "docs_exists": False}
    if os.path.exists(emb_path):
        stats["embeddings_exists"] = True
        vecs = np.load(emb_path)
        stats["num_chunks"] = vecs.shape[0]
        stats["vector_dim"] = vecs.shape[1]
        stats["emb_size_mb"] = round(os.path.getsize(emb_path) / (1024 * 1024), 2)
    if os.path.exists(docs_path):
        stats["docs_exists"] = True
        stats["docs_size_kb"] = round(os.path.getsize(docs_path) / 1024, 2)
    return stats


def list_pdfs_in_data():
    """List PDF files in the data directory."""
    if not os.path.exists(DATA_DIR):
        return []
    return [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]


# ---------------------------------------------------------------------------
# PDF Extraction & Chunking
# ---------------------------------------------------------------------------
def extract_pdf_multimodal(pdf_path, progress_cb=None):
    """Extract text and images from PDF. Returns list of page blocks."""
    doc = fitz.open(pdf_path)
    blocks = []
    total = len(doc)
    os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

    for idx, page in enumerate(doc):
        page_text = page.get_text("blocks")
        imgs = page.get_images(full=True)
        saved_images = []

        for img in imgs:
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n > 3:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img_name = f"page{idx+1}_img{xref}.png"
                img_path = os.path.join(IMAGE_OUTPUT_DIR, img_name)
                pix.save(img_path)
                saved_images.append({"path": img_path})
                pix = None
            except Exception:
                pass

        combined_text = "\n\n".join([b[4] for b in page_text if b[4].strip()])
        blocks.append({
            "page": idx + 1,
            "text": combined_text,
            "images": saved_images,
            "meta": {"source": os.path.basename(pdf_path), "page": idx + 1}
        })

        if progress_cb:
            progress_cb(idx + 1, total)

    return blocks


def chunk_text(blocks, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split page blocks into overlapping text chunks."""
    chunks = []
    for b in blocks:
        text = b["text"].strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            ct = text[start:end].strip()
            meta = dict(b["meta"])
            meta["images"] = [img["path"] for img in b["images"]]
            chunks.append({"text": ct, "meta": meta})
            if end == len(text):
                break
            start = end - overlap
    return chunks


# ---------------------------------------------------------------------------
# Vectorstore Initialisation
# ---------------------------------------------------------------------------
def init_vectorstore(collection_name, embedding_model):
    """Initialise Qdrant vectorstore or fall back to local index."""
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Try Qdrant
    try:
        if qdrant_api_key:
            client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            client = QdrantClient(url=qdrant_url)
        client.get_collections()
        if QdrantVectorstore:
            vs = QdrantVectorstore(client=client, collection_name=collection_name, embeddings=embeddings)
            return vs, "qdrant"
    except Exception:
        pass

    # Fallback: local index
    emb_path = os.path.join(LOCAL_INDEX_DIR, "embeddings.npy")
    docs_path = os.path.join(LOCAL_INDEX_DIR, "docs.json")

    if not (os.path.exists(emb_path) and os.path.exists(docs_path)):
        return None, "none"

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
            self.index_dim = vecs.shape[1]

        def get_relevant_documents(self, query):
            qv = np.array(self.emb.embed_query(query)).reshape(1, -1)
            if qv.shape[1] != self.index_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: query={qv.shape[1]}d, index={self.index_dim}d. "
                    f"Ensure the embedding model matches the one used to create the index."
                )
            _, idxs = self.nn.kneighbors(qv, n_neighbors=min(self.top_k, len(self.docs)))
            return [Doc(self.docs[i]["text"], self.docs[i].get("meta", {})) for i in idxs[0]]

    class LocalVectorstore:
        def __init__(self, retriever):
            self.retriever = retriever
        def as_retriever(self, **kwargs):
            return self.retriever

    top_k = 4
    vs = LocalVectorstore(LocalRetriever(embeddings, nn, vectors, docs, top_k))
    return vs, "local"


# ---------------------------------------------------------------------------
# Index PDF into Qdrant / Local
# ---------------------------------------------------------------------------
def index_pdf(pdf_path, collection_name, embedding_model, export_local=True, progress_cb=None):
    """Full pipeline: extract -> chunk -> embed -> store."""
    blocks = extract_pdf_multimodal(pdf_path)
    chunks = chunk_text(blocks)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Try Qdrant first
    qdrant_ok = False
    client = None
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    try:
        client = QdrantClient(url=qdrant_url)
        client.get_collections()
        qdrant_ok = True
    except Exception:
        client = QdrantClient(":memory:")

    # Determine vector size
    test_vec = embeddings.embed_query("test")
    vector_size = len(test_vec)

    # Create/get collection
    try:
        client.get_collection(collection_name)
    except Exception:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE)
        )

    vectors_list = []
    payloads = []

    for i, chunk in enumerate(chunks):
        vector = embeddings.embed_query(chunk["text"])
        payload = {"text": chunk["text"], **chunk["meta"]}
        vectors_list.append(vector)
        payloads.append(payload)
        client.upsert(
            collection_name=collection_name,
            points=[qmodels.PointStruct(id=i, vector=vector, payload=payload)]
        )
        if progress_cb:
            progress_cb(i + 1, len(chunks))

    # Export local index
    if export_local:
        os.makedirs(LOCAL_INDEX_DIR, exist_ok=True)
        np.save(os.path.join(LOCAL_INDEX_DIR, "embeddings.npy"), np.array(vectors_list))
        with open(os.path.join(LOCAL_INDEX_DIR, "docs.json"), "w", encoding="utf-8") as f:
            json.dump(payloads, f, ensure_ascii=False, indent=2)

    return {
        "pages": len(blocks),
        "chunks": len(chunks),
        "images": sum(len(b["images"]) for b in blocks),
        "qdrant": qdrant_ok,
    }


# ---------------------------------------------------------------------------
# Groq LLM Loading
# ---------------------------------------------------------------------------
def load_groq_llm(model_name):
    """Load a Groq LLM. Returns (llm, model_name)."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set. Add it to your .env file.")
    llm = ChatGroq(
        model=model_name,
        api_key=GROQ_API_KEY,
        temperature=0.2,
        max_tokens=1024,
    )
    return llm, model_name


# ---------------------------------------------------------------------------
# Query RAG (using Groq)
# ---------------------------------------------------------------------------
def query_rag(question, retriever, llm_model, summarize=False):
    """Run a RAG query using Groq LLM. Returns (answer, sources, images, page_screenshots)."""
    docs = retriever.get_relevant_documents(question)
    if not docs:
        return "No relevant documents found.", [], [], []

    context = "\n\n".join(d.page_content for d in docs)
    sources = [d.metadata.get("source", "unknown") for d in docs]

    # Extract unique images from retrieved documents with robust path resolution
    images = []
    for d in docs:
        for img_path in d.metadata.get("images", []):
            if not img_path:
                continue
            # Normalize path backslashes
            normalized_path = img_path.replace("\\", "/")
            if not os.path.exists(normalized_path):
                # Attempt to find it relative to BASE_DIR/extracted_images
                basename = os.path.basename(normalized_path)
                workspace_path = os.path.join(BASE_DIR, "extracted_images", basename)
                if os.path.exists(workspace_path):
                    normalized_path = workspace_path
            
            if os.path.exists(normalized_path) and normalized_path not in images:
                images.append(normalized_path)

    # Render high-resolution screenshots of the matching pages from the source PDF
    page_screenshots = []
    for d in docs:
        source_pdf = d.metadata.get("source")
        page_num = d.metadata.get("page")
        if source_pdf and page_num:
            # Locate PDF path relative to DATA_DIR
            basename_pdf = os.path.basename(source_pdf)
            pdf_path = os.path.join(DATA_DIR, basename_pdf)
            if os.path.exists(pdf_path):
                try:
                    doc = fitz.open(pdf_path)
                    if 0 < page_num <= len(doc):
                        page = doc[page_num - 1]
                        pix = page.get_pixmap(dpi=110)
                        screenshot_dir = os.path.join(BASE_DIR, "page_screenshots")
                        os.makedirs(screenshot_dir, exist_ok=True)
                        screenshot_name = f"{os.path.splitext(basename_pdf)[0]}_page_{page_num}.png"
                        screenshot_path = os.path.join(screenshot_dir, screenshot_name)
                        pix.save(screenshot_path)
                        normalized_scr = screenshot_path.replace("\\", "/")
                        if normalized_scr not in page_screenshots:
                            page_screenshots.append(normalized_scr)
                except Exception:
                    pass

    try:
        llm, model_name = load_groq_llm(llm_model)

        if summarize:
            summary_prompt = f"""Summarize the following context in 3-5 bullet points:

Context:
{context}

Summary:"""
            summary_resp = llm.invoke(summary_prompt)
            summary_text = summary_resp.content if hasattr(summary_resp, "content") else str(summary_resp)

        prompt = f"""You are a clear, helpful tutor. Use the context below to answer.

Context:
{context}

Question: {question}

Answer:"""

        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)

        if summarize:
            answer = f"**Retrieved Context Summary:**\n{summary_text}\n\n---\n\n**RAG Answer:**\n{answer}"

    except Exception as e:
        answer = f"""⚠️ **Groq API failed or is unavailable. Showing retrieved context instead:**

---
### 📚 Retrieved Context

{context}
---"""

    return answer, sources, images, page_screenshots


# ===========================================================================
#  STREAMLIT APP
# ===========================================================================
st.set_page_config(page_title="RAG Pipeline", page_icon=":books:", layout="wide")
st.title("Multimodal RAG Pipeline")
st.caption("Powered by LangChain + Groq + Ollama Embeddings + Qdrant")

# -- Sidebar Configuration ---------------------------------------------------
with st.sidebar:
    st.header("Configuration")

    # LLM Model (Groq)
    st.subheader("LLM (Groq)")
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not found in .env")
        default_groq = GROQ_MODELS[0]
    else:
        st.success("Groq API key detected")

    default_groq = GROQ_MODELS[0]
    llm_model = st.selectbox("Groq Model", GROQ_MODELS, index=0)

    # Embedding Model (HuggingFace)
    st.subheader("Embeddings (HuggingFace)")
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    st.info(f"Model: {embedding_model}")

    # Retrieval settings
    st.subheader("Retrieval")
    top_k = st.slider("Top-K Results", min_value=2, max_value=10, value=4)
    collection_name = st.text_input("Collection Name", value=DEFAULT_COLLECTION)
    summarize = st.checkbox("Summarize Context", value=False)
    show_visual_context = st.checkbox("Show Visual Context (Images/Pages)", value=True)

    st.divider()
    if st.button("Reset Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# -- Initialise session state -------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "vs_source" not in st.session_state:
    st.session_state.vs_source = None

# -- Tabs --------------------------------------------------------------------
tab_chat, tab_index, tab_status = st.tabs(["Chat", "Index PDF", "System Status"])

# ===========================================================================
# TAB 1: Chat
# ===========================================================================
with tab_chat:
    # Initialise vectorstore on first visit
    if st.session_state.vectorstore is None:
        with st.spinner("Loading vectorstore..."):
            vs, src = init_vectorstore(collection_name, embedding_model)
            st.session_state.vectorstore = vs
            st.session_state.vs_source = src
        if vs is None:
            st.warning("No vector index available. Go to **Index PDF** tab to create one.")
        else:
            st.toast(f"Vectorstore loaded ({src})")

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)
            if show_visual_context:
                if msg.get("images"):
                    st.write("📷 **Extracted Diagrams / Figures:**")
                    cols = st.columns(min(3, len(msg["images"])))
                    for idx, img_path in enumerate(msg["images"]):
                        with cols[idx % len(cols)]:
                            if os.path.exists(img_path):
                                st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
                if msg.get("page_screenshots"):
                    st.write("📄 **Matching PDF Page Screenshots:**")
                    cols = st.columns(min(2, len(msg["page_screenshots"])))
                    for idx, scr_path in enumerate(msg["page_screenshots"]):
                        with cols[idx % len(cols)]:
                            if os.path.exists(scr_path):
                                st.image(scr_path, caption=os.path.basename(scr_path), use_container_width=True)

    # Chat input
    if st.session_state.vectorstore is not None:
        if prompt := st.chat_input("Ask a question about your documents..."):
            # User message
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": top_k})
                        answer, sources, images, page_screenshots = query_rag(prompt, retriever, llm_model, summarize=summarize)

                        # Format sources
                        unique_sources = list(dict.fromkeys(sources))
                        sources_html = "<br>".join(f"- `{s}`" for s in unique_sources)
                        full_response = f"{answer}\n\n---\n**Sources:**<br>{sources_html}"

                        st.markdown(full_response, unsafe_allow_html=True)

                        # Render associated images if any
                        if show_visual_context:
                            if images:
                                st.write("📷 **Extracted Diagrams / Figures:**")
                                cols = st.columns(min(3, len(images)))
                                for idx, img_path in enumerate(images):
                                    with cols[idx % len(cols)]:
                                        if os.path.exists(img_path):
                                            st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)

                            # Render page screenshots if any
                            if page_screenshots:
                                st.write("📄 **Matching PDF Page Screenshots:**")
                                cols = st.columns(min(2, len(page_screenshots)))
                                for idx, scr_path in enumerate(page_screenshots):
                                    with cols[idx % len(cols)]:
                                        if os.path.exists(scr_path):
                                            st.image(scr_path, caption=os.path.basename(scr_path), use_container_width=True)

                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": full_response,
                            "images": images,
                            "page_screenshots": page_screenshots
                        })
                    except Exception as e:
                        error_msg = f"Error: {e}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": error_msg,
                            "images": [],
                            "page_screenshots": []
                        })
    else:
        st.info("No index available. Switch to the **Index PDF** tab to get started.")

# ===========================================================================
# TAB 2: Index PDF
# ===========================================================================
with tab_index:
    st.subheader("Index a PDF Document")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Upload a PDF**")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_upload")

    with col2:
        st.markdown("**Or use an existing PDF**")
        existing_pdfs = list_pdfs_in_data()
        if existing_pdfs:
            selected_pdf = st.selectbox("PDF in /data folder", existing_pdfs)
            st.caption(f"Path: {os.path.join(DATA_DIR, selected_pdf)}")
        else:
            selected_pdf = None
            st.info("No PDFs found in the `data/` folder.")

    export_local = st.checkbox("Export local index (for offline use)", value=True)

    if st.button("Index Document", type="primary", use_container_width=True):
        pdf_path = None

        if uploaded_file is not None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(uploaded_file.getbuffer())
            tmp.close()
            pdf_path = tmp.name
        elif selected_pdf:
            pdf_path = os.path.join(DATA_DIR, selected_pdf)

        if pdf_path is None:
            st.warning("Please upload a PDF or select one from the data folder.")
        else:
            progress_bar = st.progress(0, text="Starting indexing...")
            status_text = st.empty()

            def progress_cb(current, total):
                pct = current / total
                progress_bar.progress(pct, text=f"Processing {current}/{total}...")
                status_text.text(f"Step: {current} of {total}")

            try:
                with st.spinner("Indexing PDF..."):
                    result = index_pdf(
                        pdf_path,
                        collection_name,
                        embedding_model,
                        export_local=export_local,
                        progress_cb=progress_cb,
                    )

                progress_bar.progress(1.0, text="Done!")
                st.success("Indexing complete!")
                st.json({
                    "Pages": result["pages"],
                    "Chunks": result["chunks"],
                    "Images extracted": result["images"],
                    "Stored in Qdrant": result["qdrant"],
                    "Local index exported": export_local,
                })

                # Reset vectorstore so chat tab re-initialises
                st.session_state.vectorstore = None
                st.session_state.vs_source = None

            except Exception as e:
                st.error(f"Indexing failed: {e}")
            finally:
                if uploaded_file is not None and pdf_path and pdf_path.startswith(tempfile.gettempdir()):
                    try:
                        os.unlink(pdf_path)
                    except Exception:
                        pass

# ===========================================================================
# TAB 3: System Status
# ===========================================================================
with tab_status:
    st.subheader("System Health")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Groq API**")
        groq_ok = check_groq_health()
        if groq_ok:
            st.success("Groq API is connected")
        else:
            if GROQ_API_KEY:
                st.error("Groq API key set but connection failed")
            else:
                st.error("GROQ_API_KEY not set in .env")

    with col2:
        st.markdown("**Embeddings (HuggingFace)**")
        st.success("HuggingFace Embeddings loaded locally")

    with col3:
        st.markdown("**Qdrant**")
        qdrant_ok = check_qdrant_health()
        if qdrant_ok:
            st.success("Qdrant is reachable")
        else:
            st.warning("Qdrant not reachable (local fallback used)")
            st.caption("Start with: `docker run -p 6333:6333 qdrant/qdrant`")

    st.divider()
    st.subheader("Local Index")
    stats = get_local_index_stats()
    if stats.get("embeddings_exists") and stats.get("docs_exists"):
        st.success("Local index found")
        c1, c2, c3 = st.columns(3)
        c1.metric("Chunks", stats.get("num_chunks", "?"))
        c2.metric("Vector Dim", stats.get("vector_dim", "?"))
        c3.metric("Index Size", f"{stats.get('emb_size_mb', '?')} MB")
    else:
        st.info("No local index found. Index a PDF to create one.")

    st.divider()
    st.subheader("Available Groq Models")
    for m in GROQ_MODELS:
        st.code(m)

    st.divider()
    st.subheader("Quick Actions")
    sc1, sc2 = st.columns(2)
    with sc1:
        if st.button("Reload Vectorstore", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.vs_source = None
            st.toast("Vectorstore will reload on next chat access")
    with sc2:
        if st.button("Refresh Status", use_container_width=True):
            st.rerun()



# cd d:\all\RAG-Pipeline-Development-main
# python -m streamlit run app.py