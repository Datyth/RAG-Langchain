# Production-Ready RAG Pipeline with LangChain and FastAPI

This project implements a basic complete, deployment-ready Retrieval-Augmented Generation (RAG) pipeline using Python, LangChain, FastAPI, LangServe and API LLMs.
## 📜 Overview

The core idea of this RAG pipeline is to enhance the knowledge of an LLM with custom, private data. Instead of relying solely on its pretrained knowledge, the model can look up relevant information from a provided set of documents before answering a question.

This repository is structured as a production-ready application. It separates concerns into logical modules for data loading, vector storage, LLM interfacing, and the core RAG logic, all wrapped in a high-performance FastAPI server with LangServe.

## ✨ Features

* **Deployment-Ready API**: Built with **FastAPI** and **LangServe**, providing an instant, production-grade API for your RAG chain.
* **Built-in Playground**: Automatically includes a LangServe web interface for easy testing and interaction with your RAG chain.
* **Modular Architecture**: Code is professionally organized into a `src` directory with clear separation of concerns (e.g., `file_loader`, `vectorstore`, `llm_models`), making it easy to maintain and extend.
* **Persistent Vector Storage**: Uses an abstracted `VectorDB` class that supports both **ChromaDB** and **FAISS**. It automatically saves embeddings to disk and loads them on startup, saving you from re-processing files.
* **Efficient PDF Ingestion**: Loads and processes multiple PDF files from a directory using the efficient `PyMuPDFLoader`.
* **Modern LCEL Chain**: The RAG logic is built using the LangChain Expression Language (LCEL), not legacy classes, for a clear and composable pipeline.
* **Extensible LLM Support**: Easily swap LLMs by adding a new class to `src/rag/base/llm_models.py`. **Google Gemini** is implemented by default.

![API docs interface][assert/image_01.jpg]
![Question and Response][assert/response-example.jpg]

## ⚙️ How It Works: The API Pipeline

The process is broken down into a series of modular components orchestrated by `src/app.py` and `src/rag/rag_system/main.py`.

1.  **Ingestion & Processing**:
    * The `Loader` class in `file_loader.py` scans the `src/data_source/gen_ai/` directory for PDF files.
    * It uses `PyPDFLoader` to load documents and `PDFTextSplitter` (which wraps `RecursiveCharacterTextSplitter`) to break them into smaller chunks.

2.  **Indexing**:
    * The `VectorDB` class in `vectorstore.py` checks if a persistent vector store already exists in `src/data_source/vector_db/`.
    * If not, it uses `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`) to convert all text chunks into vectors.
    * These embeddings are stored in a persistent **ChromaDB** (or **FAISS**) database. If the database *does* exist, this step is skipped, and the existing DB is loaded instantly.

3.  **Retrieval & Generation (The RAG Chain)**:
    * The `build_rag_chain` function in `main.py` defines the core logic using LCEL.
    * When a user asks a question, the `VectorDB.get_retriever()` fetches the most relevant document chunks (the "context").
    * The `RAG` class in `rag_offline.py` formats the context and question into a detailed prompt.
    * This prompt is sent to the `GeminiLLM`, which generates the final, context-aware answer.

4.  **API Serving**:
    * `app.py` initializes the FastAPI app and builds the RAG chain on startup.
    * `add_routes` from `langserve` exposes this chain at the `/generative_ai` path, giving you API endpoints and a playground for free.
    * A custom endpoint `/rag/genai` is also provided for standard POST requests.

## 🛠️ Technologies Used

* **Core Framework**: LangChain, LangServe
* **API Server**: FastAPI, Uvicorn
* **Vector Database**: ChromaDB, FAISS
* **Embedding Model**: `sentence-transformers` (`langchain_huggingface`)
* **LLM**: Google Gemini (`langchain-google-genai`)
* **PDF Parsing**: `PyMuPDF`
* **Development**: Jupyter Notebook (for prototyping)

## 🚀 Getting Started

### Prerequisites

* Python 3.10+
* A Google API Key for Gemini

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Datyth/RAG-Langchain.git](https://github.com/Datyth/RAG-Langchain.git)
    cd RAG-Langchain
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
   

4.  **Set up your API Key:**
    Create a `.env` file in the root directory and add your Google Gemini API key:
    ```
    GEMINI_API_KEY="YOUR_GOOGLE_API_KEY"
    ```
   

### Usage

1.  **Add Your Documents:**
    Place your PDF documents inside the `src/data_source/gen_ai/` directory. The application will process them on its first run.

2.  **Run the Server:**
    Start the FastAPI application using Uvicorn:
    ```bash
    uvicorn src.app:app --host 0.0.0.0 --port 3000 --reload
    ```
    On the first launch, this will build and save the vector database. This may take a few minutes. Subsequent launches will be instant.

3.  **Interact with your RAG API:**
    * **Option 1: LangServe Playground (Recommended)**
        Open your browser and navigate to `http://localhost:8000/generative_ai/playground/`.

    * **Option 2: FastAPI Docs**
        Explore the self-documenting API at `http://localhost:8000/docs`.

    * **Option 3: cURL**
        Use the custom `/rag/genai` endpoint from another terminal:
        ```bash
        curl -X POST "http://localhost:8000/rag/genai" \
             -H "Content-Type: application/json" \
             -d '{"question": "How does a genetic algorithm work?"}'
        ```


### 📂Folder Structure
Here's the project layout to help you find your way around:

```
RAG-Langchain/
│
├── src/
│   ├── app.py             # Main FastAPI server (entry point)
│   ├── data_source/
│   │   ├── gen_ai/        # <-- Place your PDFs here (and download.py script)
│   │   └── vector_db/     # (Git-ignored) Persistent vector store is saved here
│   └── rag/
│       ├── base/
│       │   └── llm_models.py  # LLM abstraction (Gemini)
│       └── rag_system/
│           ├── file_loader.py # Logic for loading & chunking files
│           ├── main.py        # `build_rag_chain` function to wire everything up
│           ├── rag_offline.py # The core RAG (LCEL) chain logic
│           ├── rag_utils.py   # Utility functions (e.g., answer parsing)
│           └── vectorstore.py # Manages (saves/loads) ChromaDB or FAISS
│
├── notebooks/
│   ├── document.ipynb     # Notebook for exploring Langchain Documents
│   └── rag-pineline.ipynb # The original notebook for prototyping
│
├── data/                  # (Legacy data from notebook prototyping)
│   ├── pdf/               #
│   └── text/              #
│
├── .env                   # (You need to create this) For API keys
├── .gitignore             #
├── README.md              # (This file)
└── requirements.txt       #
```



