from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """Return a standard LangChain HuggingFace embedding model instance."""
    print(f"[INFO] Initializing embedding model: '{model_name}'...")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

if __name__ == "__main__":
    embeddings = get_embedding_model()
    query_vector = embeddings.embed_query("What is attention mechanism?")
    print(f"[INFO] Generated query vector of dimension {len(query_vector)}")

