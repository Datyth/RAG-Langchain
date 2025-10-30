# Simple RAG Pipeline with LangChain

This project implements a complete Retrieval-Augmented Generation (RAG) pipeline using Python, LangChain, and ChromaDB. The system ingests PDF documents, processes them into a searchable vector store, and uses this store to provide contextually relevant answers to user queries through a Large Language Model (LLM) like Google's Gemini.

## 📜 Overview

The core idea of this RAG pipeline is to enhance the knowledge of an LLM with custom data. Instead of relying solely on its pre-trained knowledge, the model can "look up" relevant information from a provided set of documents before answering a question. This is particularly useful for building chatbots or Q&A systems over specific domains like internal documentation, research papers, or textbooks.

## ✨ Features

* **Document Ingestion**: Loads and processes multiple PDF files from a local directory.
* **Text Chunking**: Splits large documents into smaller, manageable chunks for effective embedding and retrieval.
* **Vector Embeddings**: Utilizes `sentence-transformers` (`all-MiniLM-L6-v2`) to convert text chunks into numerical vector representations.
* **Vector Storage**: Uses **ChromaDB** as a persistent vector store to save and index the document embeddings.
* **Efficient Retrieval**: A custom `Retriever` class fetches the most relevant document chunks for a given query based on cosine similarity.
* **LLM Integration**: Seamlessly integrates with LLMs like **Google Gemini** to generate answers based on the retrieved context.
* **Modular Code**: The pipeline is broken down into logical classes (`EmbeddingManager`, `Retriever`, `GeminiLLM`, `RAG`) for clarity and maintainability.

## ⚙️ How It Works: The RAG Pipeline

The process can be broken down into four main stages, as implemented in the `rag-pineline.ipynb` notebook.

1.  **Ingestion & Processing**:
    * The `process_all_documents` function scans the `data/pdf/` directory and loads all PDF files using `PyMuPDFLoader`.
    * The loaded documents are then passed to the `split_documents` function, which uses `RecursiveCharacterTextSplitter` to break them into smaller chunks.

2.  **Indexing**:
    * The `EmbeddingManager` class is initialized. It loads the `all-MiniLM-L6-v2` embedding model.
    * All text chunks are converted into vector embeddings.
    * These embeddings, along with their corresponding text and metadata, are stored in a persistent ChromaDB collection located in `data/vector_store/chroma_db/`.

3.  **Retrieval**:
    * When a user asks a question (e.g., "How does a genetic algorithm work?"), the `Retriever` class takes the query.
    * It converts the query into a vector embedding using the same model.
    * It then queries ChromaDB to find the most semantically similar text chunks from the database (the "context").

4.  **Generation**:
    * The `RAG` class takes the original query and the retrieved context chunks.
    * It formats this information into a detailed prompt.
    * Finally, it sends the prompt to the `GeminiLLM`, which generates a final, context-aware answer for the user.

## 🛠️ Technologies Used

* **Core Framework**: LangChain
* **Vector Database**: ChromaDB
* **Embedding Model**: `sentence-transformers`
* **LLM**: `langchain-google-genai` (Gemini)
* **PDF Parsing**: `PyMuPDF`
* **Development**: Jupyter Notebook

## 🚀 Getting Started

### Prerequisites

* Python 3.10+
* A Google API Key for Gemini

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Datyth/Simple-RAG-Langchain.git
    cd Simple-RAG-Langchain
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
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

The entire pipeline is demonstrated in the `notebooks/rag-pineline.ipynb` notebook.

1.  Place your PDF documents inside the `data/pdf/` directory.
2.  Open and run the cells in `notebooks/rag-pineline.ipynb` sequentially.
3.  The notebook will guide you through:
    * Loading and chunking the documents.
    * Embedding the chunks and storing them in ChromaDB.
    * Asking a question and seeing the retrieved context.
    * Getting the final answer from the LLM.


