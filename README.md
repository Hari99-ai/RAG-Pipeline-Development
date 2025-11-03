# 🧠 Multimodal RAG Pipeline for Educational Content

A fully functional **Retrieval-Augmented Generation (RAG)** pipeline built with **LangChain**, **Ollama**, and **Qdrant**, designed to query and summarize multimodal PDF documents containing **text, mathematical formulas, and diagrams**.

This project was developed as part of an **Intern Interview Assignment** to demonstrate applied **LLM and Data Science** skills.

---

## 🚀 Objective

To build a **console-based RAG system** capable of:

- Parsing and indexing complex educational PDFs (text + images/formulas).  
- Storing vector embeddings in **Qdrant**.  
- Using **Ollama-supported LLMs** (e.g., `mistral`, `llama3`) for context-augmented responses.  
- Demonstrating caching, summarization, and multimodal query capabilities.

---

## 🧩 Tech Stack

| Component | Technology Used |
|------------|------------------|
| **Programming Language** | Python |
| **Framework** | LangChain |
| **LLM Provider** | Ollama |
| **Vector Store** | Qdrant |
| **PDF Parser** | PyMuPDF (`fitz`) |
| **Embeddings Model** | `nomic-embed-text` |
| **Caching & Memory** | LangChain Prompt/Conversation Memory |

---

## 📁 Repository Structure

RAG-Pipeline-Development/

│

├── data/ # Sample or indexed files

├── index/ # Vector index storage

├── setup_pipeline.py # Handles PDF parsing, embeddings & Qdrant setup

├── rag_query.py # Main RAG query engine

├── generate_local_index.py # Helper for local embedding generation

├── requirements.txt # Python dependencies

└── README.md # Documentation



---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

git clone https://github.com/Hari99-ai/RAG-Pipeline-Development.git
cd RAG-Pipeline-Development
### 2️⃣ Create and activate a virtual environment

python -m venv .venv
source .venv/bin/activate      # For Linux/Mac
.venv\Scripts\activate         # For Windows
### 3️⃣ Install dependencies

pip install -r requirements.txt
### 4️⃣ Start Qdrant (via Docker)

docker run -p 6333:6333 qdrant/qdrant
### 5️⃣ Verify Ollama is installed
Install Ollama from https://ollama.ai/download and pull required models:

ollama pull mistral
ollama pull nomic-embed-text

## 🧮 Usage
🔹 Step 1: Index the PDF

python setup_pipeline.py
Expected Output:

mathematica

✅ Connected successfully!
📚 PDF parsed and 50 chunks indexed in Qdrant.

🔹 Step 2: Ask Questions (RAG Query)

python rag_query.py --question "Explain the steps involved in solving a quadratic equation."
Expected Output:

Final Answer: [LLM-generated response]
Sources: [Chunk references]

🔹 Step 3: Summarization

python rag_query.py --summarize --question "What is Arithmetic Progression?"
Output:
1. Retrieved Context Summary: [Brief summary]
2. Final RAG Answer: [LLM output]
3. 
🔹 Step 4: Caching Demonstration
Run the same question twice to verify caching/memory usage.

🧩 Features Summary

✅ PDF parsing (text + image/formulas)

✅ Qdrant vector storage

✅ Ollama-based embeddings and generation

✅ Context summarization before generation

✅ Prompt/Conversational caching

✅ Command-line interface (no UI required)

📊 Example Queries

python rag_query.py --question "What does the diagram of a trapezoid represent?"
python rag_query.py --question "Who proposed the Pythagorean theorem?"
python rag_query.py --question "What is the formula associated with his discovery?"

🧑‍💻 Author
Hari Om

📧 hariom993126@gmail.com
