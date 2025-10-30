from typing import Union, List, Literal
import glob
from pathlib import Path
from tqdm import tqdm
import multiprocessing

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

project_root = Path(__file__).resolve().parent.parent.parent.parent

def remove_non_utf8(text: str):
    return text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')

def load_pdf(file_path: str):
    docs = PyPDFLoader(file_path = file_path, extract_images = False).load()
    for doc in docs:
        doc.page_content = remove_non_utf8(doc.page_content)
    return docs

def get_num_cpu():
    return multiprocessing.cpu_count()

class BaseLoader:
    def __init__(self):
        self.num_processes = get_num_cpu()
    
    def __call__(self, file: List[str], **kwargs):
        pass

class PDFLoader(BaseLoader):
    def __init__(self):
        super().__init__()

    def __call__(self, pdf_files: List[str], **kwargs):
        result = []
        num_processes = min(self.num_processes, kwargs["num_workers"])
        with multiprocessing.Pool(processes=num_processes) as pool:
            total_files = len(pdf_files)
            with tqdm(total = total_files, desc = "Loading pdf files", unit = "file") as pbar:
                results = []
                for docs in pool.imap_unordered(load_pdf, pdf_files):
                    results.extend(docs)
                    pbar.update(1)
        return results

class PDFTextSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, separators: List[str] = ["\n\n", "\n", " ", ""]):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            separators = separators
        )

    def __call__(self, documents: List[Document]) -> List[Document]:
        return self.split_documents(documents)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        split_docs = []
        for doc in documents:
            splits = self.text_splitter.split_text(doc.page_content)
            for i, split in enumerate(splits):
                split_doc = Document(
                    page_content=split,
                    metadata={**doc.metadata, "chunk": i}
                )
                split_docs.append(split_doc)
        return split_docs
    
class Loader:
    def __init__(self,
                file_type: str = Literal["pdf"],
                split_kwargs: dict = {
                    "chunk_size": 300,
                    "chunk_overlap": 0
                }):
        assert file_type in ['pdf'], "file_type must be pdf"
        self.file_type = file_type
        if (file_type == 'pdf'):
            self.loader = PDFLoader()
        else:
            raise ValueError("file_type must be pdf")
        
        self.splitter = PDFTextSplitter(**split_kwargs)

    def load(self, pdf_files: Union[str, List[str]], num_worker : int = 1) -> List[Document]:
        if isinstance(pdf_files, str):
            pdf_files = [pdf_files]
        
        documents = self.loader(pdf_files, num_workers=num_worker)
        split_documents = self.splitter(documents)
        return split_documents
    
    def load_dir(self, dir_path: str, num_worker = 1) -> List[Document]:
        if self.file_type == 'pdf':
            search_path = project_root / dir_path
            file_pattern = f"*.{self.file_type}"
            files_list = [str(p) for p in search_path.glob(file_pattern)]
            
            if not files_list:
                print(f"Warning: Không tìm thấy file {file_pattern} nào trong {search_path}")
                return []
            
            return self.load(files_list, num_worker=num_worker)
        
        return []
        

    

if __name__ == "__main__":
    dir_path = "src/data_source/gen_ai"
    loader = Loader(file_type = 'pdf')
    docs = loader.load_dir(dir_path = dir_path)
    print(docs[0])



