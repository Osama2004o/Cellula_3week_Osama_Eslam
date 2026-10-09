# Cellula Week 3 — NLP Tasks

> **Authors:** Osama Eslam & Eslam  
> **Course:** Cellula NLP Track — Week 3

---

## 📖 Overview

This repository contains the deliverables for **Week 3** of the Cellula NLP course. It includes two tasks:

| Task | Notebook | Topic |
|------|----------|-------|
| **Task 0** | [`task0.ipynb`](task0.ipynb) | BigBird paper deep-dive — architecture, sparse attention, complexity analysis, and Hugging Face usage |
| **Task 1** | [`task1.ipynb`](task1.ipynb) | Personal Knowledge Base RAG pipeline — document loading, chunking, embedding, vector search, and LLM-powered Q&A |

---

## 🧠 Task 0 — BigBird: Transformers for Longer Sequences

A comprehensive study notebook covering the [BigBird paper](https://arxiv.org/abs/2007.14062) (Zaheer et al., NeurIPS 2020):

- **Motivation:** Why the quadratic attention bottleneck limits standard transformers to ~512 tokens.
- **Sparse Attention:** The three ingredients — sliding window (local), global tokens, and random attention.
- **Block-Sparse Implementation:** How the token-level idea maps to GPU/TPU-efficient block operations.
- **Theoretical Justification:** Graph-theory arguments (expander graphs, small-world networks) proving universal approximation and Turing completeness.
- **Complexity Comparison:** BERT vs Longformer vs BigBird — with FLOPs/memory plots.
- **Hands-On Usage:** Loading and running BigBird models via Hugging Face (`google/bigbird-roberta-base`, `google/bigbird-pegasus-large-arxiv`).
- **Applications:** Long-document QA, summarization, and genomics.

---

## 🤖 Task 1 — Personal Knowledge Base RAG Pipeline

An end-to-end **Retrieval-Augmented Generation (RAG)** system that answers questions about a personal knowledge base document.

### Pipeline Steps

1. **Document Loading** — Uses `TextLoader` (LangChain) to ingest [`osama_eslam_personal_knowledge_base.txt`](osama_eslam_personal_knowledge_base.txt).
2. **Chunking** — Splits the document with `TokenTextSplitter` (chunk size 700, overlap 100).
3. **Embedding** — Generates dense vectors using `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace BGE Embeddings.
4. **Vector Store** — Indexes chunks in a **FAISS** in-memory vector store for similarity search.
5. **Prompt Construction** — Retrieves the top-k similar chunks and formats them into a grounded prompt template.
6. **LLM Generation** — Sends the prompt to an LLM via **OpenRouter API** (`ChatOpenAI`) and displays the Markdown-formatted response.

### Key Technologies

- **LangChain** (document loaders, text splitters, vector stores, prompt templates)
- **FAISS** (Facebook AI Similarity Search)
- **HuggingFace Transformers** (sentence-transformers for embeddings)
- **OpenRouter API** (LLM inference)
- **Python 3.11**

---

## 📁 Project Structure

```
Cellula_3week_Osama_Eslam/
├── task0.ipynb                              # BigBird paper exploration notebook
├── task1.ipynb                              # Personal Knowledge Base RAG pipeline
├── osama_eslam_personal_knowledge_base.txt  # Knowledge base document (RAG source)
├── .env.example                             # Environment variable template
├── .env                                     # Actual API keys (git-ignored)
├── .gitignore                               # Python gitignore
├── LICENSE                                  # Apache 2.0
└── README.md                                # This file
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- An [OpenRouter](https://openrouter.ai/) API key

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Osama2004o/Cellula_3week_Osama_Eslam.git
   cd Cellula_3week_Osama_Eslam
   ```

2. **Install dependencies:**
   ```bash
   pip install langchain langchain-openai langchain-community langchain-classic faiss-cpu sentence-transformers python-dotenv
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and paste your OpenRouter API key:
   ```
   OPENROUTER_API = "your_openrouter_api_key_here"
   ```

4. **Run the notebooks** in Jupyter or VS Code.

---

## 📜 License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.