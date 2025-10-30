import re 
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from operator import itemgetter

class StrOutputParserCustom(StrOutputParser):
    def __init__(self):
        super().__init__()

    def parse(self, text: str) -> str:
        return self.extract_answer(text)
    
    def extract_answer(self, text: str, pattern: str = r"Answer: \s*(.*)"):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return text.strip()
        

class RAG:
    def __init__(self, llm):
        self.llm = llm
        template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:"""
        self.prompt = PromptTemplate.from_template(template)
        
        self.output_parser = StrOutputParserCustom()

    def get_chain(self, retriever):
        input_data = {
            "question": itemgetter("question"), 
            "context": itemgetter("question") | retriever | self.format_docs 
        }
        
        rag_chain = (
            input_data
            | self.prompt
            | self.llm
            | self.output_parser
        )
        return rag_chain
    
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)


