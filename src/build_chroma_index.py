import json
from pathlib import Path

import numpy as np
import chromadb

CHUNKS_PATH = Path("data/chunks.json")
EMBEDDINGS_PATH = Path("data/embeddings.npy")
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "silmarillion"


def build_index():
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    embeddings = np.load(EMBEDDINGS_PATH)

    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks but {embeddings.shape[0]} embeddings."
        )

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [{"section": chunk["section"]} for chunk in chunks]
    embeddings_list = embeddings.tolist()

    BATCH_SIZE = 100
    for start in range(0, len(ids), BATCH_SIZE):
        end = start + BATCH_SIZE
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings_list[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  Added {min(end, len(ids))}/{len(ids)} chunks to Chroma")

    print(f"\nChroma collection '{COLLECTION_NAME}' built at ./{CHROMA_DB_PATH}")
    print(f"Total vectors stored: {collection.count()}")


if __name__ == "__main__":
    build_index()