import os
from pathlib import Path
from typing import List, Any


try:
    from langchain_core.documents import Document
except ImportError:
    Document = Any

try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader
except ImportError:
    PyPDFLoader = TextLoader = CSVLoader = Docx2txtLoader = None

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        RecursiveCharacterTextSplitter = None


LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".csv": CSVLoader,
    ".docx": Docx2txtLoader,
}

def load_all_documents(data_dir: str = "data") -> List[Document]:
    """Load supported documents (PDF, TXT, MD, CSV, DOCX) from a directory."""
    path = Path(data_dir).resolve()
    if not path.exists():
        print(f"[WARN] Data directory '{data_dir}' does not exist.")
        return []

    documents = []
    for file_path in path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in LOADERS:
            loader_cls = LOADERS[file_path.suffix.lower()]
            try:
                loader = loader_cls(str(file_path))
                docs = loader.load()
                documents.extend(docs)
                print(f"[INFO] Loaded {len(docs)} document page(s) from {file_path.name}")
            except Exception as e:
                print(f"[ERROR] Failed to load {file_path.name}: {e}")

    print(f"[INFO] Total documents loaded: {len(documents)}")
    return documents


def load_and_split_documents(data_dir: str = "data", chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """Load documents from data_dir and split into chunks."""
    docs = load_all_documents(data_dir)
    if not docs:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Split {len(docs)} documents into {len(chunks)} chunks.")
    return chunks


if __name__ == "__main__":
    docs = load_all_documents("data")
    print(f"Loaded {len(docs)} documents.")