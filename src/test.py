"""
Milestone 1: Sanity check.
Confirms your Gemini API key works for BOTH embedding and generation
before we build any RAG logic on top of it.

"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load GEMINI_API_KEY from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found. Did you create a .env file "
        "(copy .env.example to .env and paste your key in)?"
    )

genai.configure(api_key=api_key)

print("Testing embedding call...")
embed_result = genai.embed_content(
    model="models/gemini-embedding-001",
    content="Who forged the Silmarils?",
    task_type="retrieval_query",
    output_dimensionality=768,
)
embedding_vector = embed_result["embedding"]
print(f"  Success. Embedding dimension: {len(embedding_vector)}")

print("\nTesting generation call...")
model = genai.GenerativeModel("gemini-2.5-flash")
response = model.generate_content("Say 'RAG pipeline ready' and nothing else.")
print(f"  Success. Model said: {response.text.strip()}")

print("\nBoth calls worked. You're ready for Milestone 2 (chunking).")