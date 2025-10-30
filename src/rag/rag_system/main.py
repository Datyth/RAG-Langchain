from pydantic import BaseModel, Field
from .file_loader import Loader
from .vectorstore import VectorDB
from .rag_offline import RAG


class InputQA(BaseModel):
    question: str = Field(..., title="Question to ask the model")

class OutputQA(BaseModel):
    answer: str = Field(..., title="Answer from the model")
 
def build_rag_chain(llm, data_dir, data_type):
    doc_loaded = Loader(file_type=data_type).load_dir(data_dir, num_worker = 2)
    retriever = VectorDB(documents = doc_loaded).get_retriever()
    rag_chain = RAG(llm.model).get_chain(retriever)
    return rag_chain
    
# if __name__ == "__main__":
    
