from typing import List

try:
    from langchain_core.embeddings import Embeddings
except ImportError:
    Embeddings = object

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class LocalSentenceTransformerEmbeddings(Embeddings):
    """Clean SentenceTransformer embedding wrapper for LangChain vector stores."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise ImportError("Package 'sentence-transformers' is not installed. Please run: pip install sentence-transformers")
        print(f"[INFO] Initializing SentenceTransformer model: '{model_name}'...")
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(text, show_progress_bar=False)
        return embedding.tolist()


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> Embeddings:
    """Return a standard LangChain-compatible embedding model instance."""
    return LocalSentenceTransformerEmbeddings(model_name=model_name)


if __name__ == "__main__":
    embeddings = get_embedding_model()
    query_vector = embeddings.embed_query("What is attention mechanism?")
    print(f"[INFO] Generated query vector of dimension {len(query_vector)}")


