import json
from pathlib import Path

import numpy as np

CHUNKS_PATH = Path("data/chunks.json")
EMBEDDINGS_PATH = Path("data/embeddings.npy")


class ChromaVectorStore:
    def __init__(self, db_path: str = "chroma_db", collection_name: str = "silmarillion"):
        import chromadb
        client = chromadb.PersistentClient(path=db_path)
        self.collection = client.get_collection(collection_name)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
        )

        output = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for text, metadata, distance in zip(documents, metadatas, distances):
            similarity = 1 - distance
            output.append({
                "section": metadata["section"],
                "text": text,
                "similarity": float(similarity),
            })
        return output

class NaiveVectorStore:
    def __init__(self):
        self.chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        self.embeddings = np.load(EMBEDDINGS_PATH)

        if len(self.chunks) != self.embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(self.chunks)} chunks but {self.embeddings.shape[0]} "
                "embeddings. Did chunks.json change after you ran embedder.py? "
                "Re-run embedder.py to regenerate embeddings.npy."
            )
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.normalized_embeddings = self.embeddings / norms

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        query_vector = np.array(query_vector, dtype=np.float32)
        query_norm = query_vector / np.linalg.norm(query_vector)

        similarities = self.normalized_embeddings @ query_norm  # shape: (num_chunks,)

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append({
                "section": chunk["section"],
                "text": chunk["text"],
                "similarity": float(similarities[idx]),
            })
        return results


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    import google.generativeai as genai

    load_dotenv()
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    store = NaiveVectorStore()
    print(f"Loaded {len(store.chunks)} chunks with embeddings shape {store.embeddings.shape}\n")

    test_query = "Who forged the Silmarils?"
    print(f"Test query: {test_query!r}\n")

    query_embed_result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=test_query,
        task_type="retrieval_query",
        output_dimensionality=768,
    )
    query_vector = query_embed_result["embedding"]

    results = store.search(query_vector, top_k=3)

    for rank, r in enumerate(results, start=1):
        print(f"#{rank} | similarity={r['similarity']:.4f} | section={r['section']}")
        print(f"    {r['text'][:200]}...\n")