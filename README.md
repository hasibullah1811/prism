# 💎 Prism: RAG Visualizer

Prism is a full-stack developer tool for visualizing and debugging RAG (Retrieval-Augmented Generation) pipelines. It helps engineers understand how text chunking strategies (Token size, Overlap) affect retrieval accuracy.

## 🚀 Features

-   **Real-time Chunking:** Visualize how `RecursiveCharacterTextSplitter` breaks down documents.
-   **Overlap Highlighting:** Visually identify the context overlap between chunks (crucial for maintaining semantic meaning).
-   **Search Simulation:** Test retrieval using TF-IDF and Cosine Similarity to find which chunks match a user query.
-   **Tokenizer Stats:** Real-time token counting using `tiktoken`.

## 🛠️ Tech Stack

**Frontend:**
-   React (Vite)
-   Tailwind CSS (Modern UI)
-   Lucide React (Icons)

**Backend:**
-   Python (FastAPI)
-   LangChain (Splitting logic)
-   Scikit-learn (TF-IDF Search)
-   Tiktoken (OpenAI Tokenization)

## 📦 How to Run

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

2. **Start the Backend (Python)**
Navigate to the backend folder and activate the environment:

```bash
cd backend
# Activate virtual environment
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Server
uvicorn main:app --reload
```
Server running at: http://127.0.0.1:8000

3. **Start the Frontend (React)**
Open a new terminal and navigate to the frontend folder:
```bash
cd frontend

# Install Node modules
npm install

# Run Frontend
npm run dev
```
App running at: http://localhost:5173

## 🧠 Why this tool?
Building RAG pipelines involves many "silent" failures. If a document is split mid-sentence, or if the overlap is too small, the LLM loses context. Prism makes these invisible data issues visible, allowing engineers to tune their chunking parameters before deploying to production.

Created by hasibullah1811