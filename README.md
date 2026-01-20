# 💎 Prism | RAG Chunk Visualizer

**Prism** is an open-source developer tool designed to "break open the black box" of RAG (Retrieval-Augmented Generation) pipelines. It allows ML engineers to visualize, audit, and tune their text splitting strategies before calculating expensive embeddings.

## 🚀 Why Prism?

Building a RAG pipeline usually involves blindly choosing a `chunk_size` (e.g., 500) and hoping for the best. This often leads to:
* **Broken Semantics:** Sentences cut in half, confusing the LLM.
* **Wasted Tokens:** Excessive overlap that bloats costs without adding context.
* **Poor Retrieval:** Important keywords getting split across two chunks.

**Prism** solves this by providing instant visual feedback on your splitting strategy.

## ✨ Key Features

* **🔍 Interactive Splitter:** Adjust `Chunk Size` and `Overlap` in real-time with a sliding window.
* **🎨 Overlap Highlighter:** Visualizes exactly which text is repeated between chunks (highlighted in yellow), helping you optimize token usage.
* **🧠 Retrieval Simulator:** Includes a local TF-IDF Search Engine. Type a query to see which chunks would actually be retrieved.
* **⚠️ Semantic Auditing:** Automatically detects "Bad Cuts" (chunks ending mid-sentence) and flags them with a warning badge.
* **💾 JSON Export:** One-click export of your tuned chunks, ready for ingestion into a Vector Database (Pinecone, Weaviate, etc.).

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Frontend:** Streamlit
* **Logic:** LangChain (RecursiveCharacterTextSplitter)
* **Search:** Scikit-Learn (TF-IDF & Cosine Similarity)

## ⚡ Quick Start

### Prerequisites
* Python 3.8 or higher
* Git

### Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/hasibullah1811/prism.git](https://github.com/hasibullah1811/prism.git)
    cd prism
    ```

2.  **Create a Virtual Environment** (Recommended)
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt