# 📘 RAG Assistant for RFP response generation

A domain-aware Retrieval-Augmented Generation (RAG) system powered by **local LLMs**, **FAISS vector search**, and an interactive **Gradio UI**. Designed for document-grounded question answering with transparency, offline support, and user feedback capture.

---

## 🚀 Features

- 🔍 **Local LLM inference** using `llama-cpp-python` (no API calls)
- 🤖 **Few-shot classification**, **query expansion**, and **optional summarization**
- 📂 **Recursive document chunking** and **FAISS-based retrieval**
- 🧠 **Hybrid search** + cosine reranking for high-relevance results
- 🖥️ **Gradio UI** with real-time query handling, feedback, and suggestions
- 📤 Export sessions to `.docx` and record feedback to `.csv`
- ✅ Fully **offline-capable** once dependencies are installed

---

## 🧰 Prerequisites

- Python ≥ 3.8 (recommended: 3.10)
- `.gguf` LLM files (e.g. TinyLLaMA, Mistral) downloaded to `models/`
- Input `.txt` files placed under `data/RFP_data/` for ingestion
- Linux/macOS recommended for full OCR support

---

## 🔧 Setup Instructions

### 🖥️ Run Locally (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/Nisarg-P-Patel/RAG-based-QA-System.git

# 2. Run full setup (installs dependencies, models, system tools)
chmod +x setup.sh
./setup.sh

# 3. Get into the venv 
source venv/bin/activate

# 4. Launch the Gradio UI
python -m app.app

```
---
## Project File/Folder Structure
```
RAG-based-QA-System/
├── app/                             ← Gradio UI, app logic & interaction handlers
│   ├── app.py                       ← Launches Gradio interface
│   ├── helper.py                    ← Query logic, feedback, doc export
│   └── local_llm_reader.py          ← Loads local GGUF models with LlamaCpp
│
├── models/                          ← Core LLM logic + RAG pipeline
│   ├── rag.py                       ← Full RAG orchestration
│   ├── summarizer.py                ← Summarization using DistilBART
│   ├── classifier.py                ← Few-shot classifier (DeBERTa-v3)
│   ├── download_gguf_models.py      ← Downloads Mistral / TinyLLaMA from HuggingFace
│   ├── mistral.gguf                 ← (7B) Local LLM model file
│   └── tinyllama.gguf               ← (1.1B) Local LLM model file
│
├── vectorstore/                     ← Embedding + FAISS indexing
│   ├── embedder.py                  ← SentenceTransformer (MiniLM) embeddings
│   ├── index.py                     ← FAISS builder
│   └── retriever.py                 ← Hybrid search + cosine rerank
│
├── data/
│   ├── preprocessing.py             ← Processes .txt/.pdf into vectorizable chunks
│   ├── RFP_data/                    ← Raw source documents
│   └── VectorDB-Data-Folder/        ← Preprocessed and indexed documents
│
├── visualization/
│   └── visualize.py                 ← Graphviz-based RAG architecture flowcharts
│
├── feedback/
│   ├── rag_feedback.csv             ← Free-form feedback from users
│   └── rag_votes.csv                ← Upvotes and downvotes logged from UI
│
├── exports/
│   └── rag_summary.docx             ← Word file summary of each session
│
├── requirements.txt                 ← Python dependencies
├── setup.py                         ← Python packaging config
├── setup.sh                         ← Shell setup (pip install + models + apt installs)
└── README.md                        ← You are here
```

---

## data folderr understanding

```
1. Once have all the data in data folder under RFP_data folder and two files run the command as:
python data/preprocessing.py

The above command will make sure the data is pre-processed and stored at the VectorDB-Data-Folder.

data/
├── RFP_data/
│   └── Website_data/web_content/batch_1/       ← Raw .txt and .pdf files
├── Support-RFP-FAQ.pdf                         ← Source FAQ PDF
├── sku_series_datasheet.csv                    ← CSV of product datasheet URLs
└── VectorDB-Data-Folder/                       ← Preprocessed output
    └── data in new processed format and structure
```
