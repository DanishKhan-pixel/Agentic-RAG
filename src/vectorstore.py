import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from src.embedding import get_embedding_model

class FaissVectorStore:
    """FAISS VectorStore manager using LangChain FAISS abstractions."""

    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2"):
        self.persist_dir = persist_dir
        self.embeddings = get_embedding_model(embedding_model)
        self.vectorstore: Optional[FAISS] = None

    def build_from_documents(self, documents: List[Document]) -> FAISS:
        """Create FAISS index from document chunks and save locally."""
        if not documents:
            raise ValueError("No documents provided to build vector store.")
        print(f"[INFO] Building FAISS vector store from {len(documents)} chunks...")
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        self.save()
        return self.vectorstore

    def save(self):
        """Save FAISS index to persist_dir."""
        if self.vectorstore:
            os.makedirs(self.persist_dir, exist_ok=True)
            self.vectorstore.save_local(self.persist_dir)
            print(f"[INFO] Saved FAISS vector store to '{self.persist_dir}'")

    def load(self) -> bool:
        """Load FAISS index from persist_dir if it exists."""
        if os.path.exists(self.persist_dir):
            try:
                self.vectorstore = FAISS.load_local(
                    self.persist_dir,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"[INFO] Loaded FAISS vector store from '{self.persist_dir}'")
                return True
            except Exception as e:
                print(f"[WARN] Could not load vector store from '{self.persist_dir}': {e}")
        return False

    def search(self, query: str, top_k: int = 4) -> List[Document]:
        """Perform similarity search for a query string."""
        if not self.vectorstore:
            print("[WARN] Vector store is not loaded.")
            return []
        return self.vectorstore.similarity_search(query, k=top_k)


if __name__ == "__main__":
    from src.data_loader import load_and_split_documents
    chunks = load_and_split_documents("data")
    if chunks:
        store = FaissVectorStore("faiss_store")
        store.build_from_documents(chunks)
        results = store.search("What is attention mechanism?", top_k=2)
        print(f"[INFO] Found {len(results)} search results.")

