import sys
from src.search import RAGChatbot
from src.data_loader import load_and_split_documents

def main():
    print("=" * 60)
    print(" 🤖 Simple RAG Chatbot")
    print("------------------------------------------------------------")
    print(" Commands:")
    print("   /clear   - Reset conversation history")
    print("   /reindex - Rebuild FAISS index from 'data/' directory")
    print("   exit     - Quit chatbot")
    print("=" * 60)

    chatbot = RAGChatbot(data_dir="data", persist_dir="faiss_store")

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            if user_input.lower() == "/clear":
                chatbot.clear_history()
                print("Chat history cleared.")
                continue

            if user_input.lower() == "/reindex":
                print("Rebuilding vector index...")
                chunks = load_and_split_documents("data")
                if chunks:
                    chatbot.vectorstore.build_from_documents(chunks)
                    print("Vector index rebuilt successfully.")
                else:
                    print("No documents found in 'data/' to index.")
                continue

            res = chatbot.chat(user_input)
            print(f"\nAI: {res['answer']}")

            if res.get("sources"):
                print("\n📌 Sources:")
                for i, src in enumerate(res["sources"], 1):
                    src_name = src["source"]
                    print(f"  [{i}] {src_name}")

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()

