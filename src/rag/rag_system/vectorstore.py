import os
from pathlib import Path
from typing import Union, Optional
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

project_root = Path(__file__).resolve().parent.parent.parent.parent

class VectorDB:
    def __init__(self, documents = None, 
                 vector_db: Union[Chroma, FAISS] = Chroma,
                 embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
                 persist_directory: Optional[str] = str(project_root / 'src/data_source/vector_db')):
        
        self.vector_db = vector_db
        self.embedding = embedding
        self.persist_directory = persist_directory
        self.db = self._load_or_build_db(documents)

    def _load_or_build_db(self, documents):
        if self.persist_directory and os.path.exists(self.persist_directory):
            print(f"Loading available VectorDB from: {self.persist_directory}")
            if self.vector_db == Chroma:
                return Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embedding
                )
            elif self.vector_db == FAISS:
                return FAISS.load_local(
                    self.persist_directory,
                    self.embedding,
                    index_name="index",
                    allow_dangerous_deserialization=True 
                )

        if documents is None:
            print(f"Warning: Cannot build DB. No persistent directory found and no documents provided.")
            return None
        
        print(f"Building new VectorDB in {self.persist_directory}")

        if self.vector_db == Chroma:
            db = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding,
                persist_directory=self.persist_directory 
            )
        
        elif self.vector_db == FAISS:
            db = FAISS.from_documents(
                documents=documents, 
                embedding=self.embedding
            )
            if self.persist_directory:
                print(f"Saving vectors into: {self.persist_directory}")
                db.save_local(self.persist_directory, index_name="index")
        
        return db
    
    def get_retriever(self, search_type: str= "similarity", search_kwargs: dict = {"k": 10}):
        if self.db is None:
            raise ValueError("Vector database is not initialized with documents.")
        return self.db.as_retriever(search_type = search_type, search_kwargs = search_kwargs)