import os
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from langchain_groq import ChatGroq

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

from src.vectorstore import FaissVectorStore
from src.data_loader import load_and_split_documents

load_dotenv()

SYSTEM_PROMPT = """You are a helpful and knowledgeable RAG Assistant.
Answer the user's question accurately using ONLY the provided context documents.
If the answer cannot be found in the context, state clearly: "I don't have enough information from the context documents to answer that."

Context:
{context}
"""

class RAGChatbot:
    """RAG Chatbot supporting multi-turn conversation and source context retrieval."""

    def __init__(
        self,
        data_dir: str = "data",
        persist_dir: str = "faiss_store",
        llm_model: str = "llama-3.3-70b-versatile"
    ):
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        self.vectorstore = FaissVectorStore(persist_dir=persist_dir)
        self.history: List[Tuple[str, str]] = []

        # Load existing index or build from data_dir
        if not self.vectorstore.load():
            print("[INFO] Index not found. Building FAISS index from data folder...")
            chunks = load_and_split_documents(data_dir=self.data_dir)
            if chunks:
                self.vectorstore.build_from_documents(chunks)

        # Initialize LLM safely using environment keys
        groq_key = os.getenv("GROQ_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if groq_key and groq_key.strip():
            self.llm = ChatGroq(model_name=llm_model, groq_api_key=groq_key, temperature=0.2)
            print(f"[INFO] ChatGroq LLM initialized with model '{llm_model}'")
        elif openai_key and openai_key.strip() and ChatOpenAI is not None:
            self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_key, temperature=0.2)
            print("[INFO] ChatOpenAI LLM initialized with model 'gpt-3.5-turbo'")
        else:
            self.llm = None
            print("[WARN] No GROQ_API_KEY or OPENAI_API_KEY found in .env file. Running in retrieval-only mode.")


    def chat(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        """Query the vectorstore and generate an LLM response with chat history."""
        retrieved_docs = self.vectorstore.search(query, top_k=top_k)
        
        if not retrieved_docs:
            return {
                "answer": "No relevant documents found. Make sure documents are placed in the 'data/' folder.",
                "sources": []
            }

        context_str = "\n\n".join([doc.page_content for doc in retrieved_docs])
        sources = [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "snippet": doc.page_content[:200]
            }
            for doc in retrieved_docs
        ]

        if not self.llm:
            return {
                "answer": f"[No LLM API Key Set - Retrieved Context]:\n\n{context_str}",
                "sources": sources
            }

        # Build prompt with system message + recent history
        messages = [("system", SYSTEM_PROMPT.format(context=context_str))]
        for user_msg, ai_msg in self.history[-3:]:
            messages.append(("human", user_msg))
            messages.append(("ai", ai_msg))
        messages.append(("human", query))

        response = self.llm.invoke(messages)
        answer = str(response.content)

        # Store turn in history
        self.history.append((query, answer))

        return {
            "answer": answer,
            "sources": sources
        }

    def clear_history(self):
        """Clear conversation history."""
        self.history.clear()

    # Legacy compatibility method
    def search_and_summarize(self, query: str, top_k: int = 4) -> str:
        res = self.chat(query, top_k=top_k)
        return res["answer"]


# Legacy class alias
RAGSearch = RAGChatbot


if __name__ == "__main__":
    bot = RAGChatbot()
    res = bot.chat("What is attention mechanism?")
    print("Answer:", res["answer"])

