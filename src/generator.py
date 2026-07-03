import os

from dotenv import load_dotenv
import google.generativeai as genai

GENERATION_MODEL = "gemini-2.5-flash"

PROMPT_TEMPLATE = """You are answering questions about J.R.R. Tolkien's The Silmarillion, using ONLY the context provided below.

Rules:
- Answer using only the information in the context.
- If the context doesn't contain enough information to answer, say so clearly instead of guessing.
- Cite which section(s) of the book the information came from when relevant.
- Keep the answer focused and well-written, matching the tone of someone knowledgeable about the lore.

Context:
{context}

Question: {question}

Answer:"""


def configure_api():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Check your .env file.")
    genai.configure(api_key=api_key)


def build_context(retrieved_chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        parts.append(f"[Source {i} - {chunk['section']}]\n{chunk['text']}")
    return "\n\n".join(parts)


def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
    context = build_context(retrieved_chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    model = genai.GenerativeModel(GENERATION_MODEL)
    response = model.generate_content(prompt)
    return response.text.strip()