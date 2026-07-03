import json
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai

CHUNKS_PATH = Path("data/chunks.json")
EMBEDDINGS_PATH = Path("data/embeddings.npy")

EMBEDDING_MODEL = "models/gemini-embedding-001"
OUTPUT_DIMENSIONALITY = 768
MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 5


def configure_api():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Check your .env file.")
    genai.configure(api_key=api_key)


def embed_one(text: str, title: str = None, task_type: str = "retrieval_document") -> list[float]:
    """
    Embed a single piece of text, with retry/backoff for transient
    errors (rate limits, temporary server issues).

    task_type matters: Gemini's embedding model produces slightly
    different vectors depending on which side of the retrieval pair
    the text is playing - 'retrieval_document' for chunks being
    indexed, 'retrieval_query' for the question being asked.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            kwargs = {
                "model": EMBEDDING_MODEL,
                "content": text,
                "task_type": task_type,
                "output_dimensionality": OUTPUT_DIMENSIONALITY,
            }
            if title:
                kwargs["title"] = title
            result = genai.embed_content(**kwargs)
            return result["embedding"]
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            wait = BASE_BACKOFF_SECONDS * attempt
            print(f"    Retry {attempt}/{MAX_RETRIES} after error: {e}. Waiting {wait}s...")
            time.sleep(wait)


def embed_query(question: str) -> list[float]:
    """Convenience wrapper for embedding a user's question at query time."""
    return embed_one(question, task_type="retrieval_query")


def embed_all_chunks():
    configure_api()
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

    print(f"Embedding {len(chunks)} chunks. This will take a few minutes...")
    vectors = []

    for i, chunk in enumerate(chunks):
        vector = embed_one(chunk["text"], title=chunk["section"])
        vectors.append(vector)

        if (i + 1) % 25 == 0 or (i + 1) == len(chunks):
            print(f"  {i + 1}/{len(chunks)} embedded")

        time.sleep(0.2)  # small buffer to stay comfortably under free-tier rate limits

    embeddings_array = np.array(vectors, dtype=np.float32)
    np.save(EMBEDDINGS_PATH, embeddings_array)

    print(f"\nSaved embeddings with shape {embeddings_array.shape} to {EMBEDDINGS_PATH}")
    print("Shape should be (num_chunks, 768) - row i corresponds to chunks[i] in chunks.json")


if __name__ == "__main__":
    embed_all_chunks()