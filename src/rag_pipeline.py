from embedder import configure_api, embed_query
from vector_store import NaiveVectorStore
from generator import generate_answer
from vector_store import ChromaVectorStore

class RAGPipeline:
    def __init__(self, top_k: int = 5):
        configure_api()
        self.store = ChromaVectorStore()
        self.top_k = top_k

    def answer(self, question: str) -> dict:
        query_vector = embed_query(question)
        retrieved_chunks = self.store.search(query_vector, top_k=self.top_k)
        answer_text = generate_answer(question, retrieved_chunks)

        return {
            "question": question,
            "answer": answer_text,
            "sources": [
                {"section": c["section"], "similarity": c["similarity"]}
                for c in retrieved_chunks
            ],
        }


if __name__ == "__main__":
    pipeline = RAGPipeline(top_k=5)

    print("Silmarillion RAG - terminal test mode. Type 'quit' to exit.\n")
    while True:
        question = input("Ask a question: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue

        result = pipeline.answer(question)

        print(f"\nAnswer:\n{result['answer']}\n")
        print("Sources used:")
        for s in result["sources"]:
            print(f"  - {s['section']} (similarity: {s['similarity']:.3f})")
        print()