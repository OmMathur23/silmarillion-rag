# The Silmarillion RAG

A Retrieval-Augmented Generation (RAG) system that answers questions about J.R.R. Tolkien's *The Silmarillion*, grounded in the actual text — built from scratch (no LangChain/LlamaIndex) to understand every layer of the pipeline before using higher-level frameworks.

**Ask it things like:**
- "Who forged the Silmarils?"
- "Who gave the name of Morgoth to Melkor?"
- "What are the names of all the major Valar?"

It retrieves the actual relevant passages from the book and generates an answer grounded in them — with source chapters cited, and an honest "not enough information" when the answer genuinely isn't in the text (tested against out-of-scope questions like "What is Fingolfin's favorite color?").

## Architecture

```
User question
    ↓
1. EMBED the question           (Gemini gemini-embedding-001, 768-dim)
    ↓
2. RETRIEVE top-k similar chunks (Chroma vector DB, cosine similarity)
    ↓
3. AUGMENT the prompt            (retrieved chunks + grounding instructions)
    ↓
4. GENERATE the answer           (Gemini gemini-2.5-flash)
```

| Stage | File | What it does |
|---|---|---|
| Chunking | `src/chunker.py` | Splits the book into ~400-word, sentence-safe chunks, respecting chapter boundaries so no chunk straddles two unrelated chapters |
| Embedding | `src/embedder.py` | Embeds each chunk (and later, each query) via Gemini's embedding API, with retry/backoff for rate limits |
| Vector store | `src/vector_store.py` | Two implementations: a naive numpy cosine-similarity store (built first, to understand the mechanics) and a Chroma-backed store (production pattern) — both share the same `.search()` interface |
| Indexing | `src/build_chroma_index.py` | One-time script loading precomputed embeddings into a persistent Chroma collection |
| Generation | `src/generator.py` | Builds the grounded prompt and calls Gemini's chat model |
| Pipeline | `src/rag_pipeline.py` | Wires the above into one callable: question → answer + sources |
| UI | `app.py` | Streamlit chat interface over the pipeline |

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env          # then paste your Gemini API key into .env
```

```bash
# One-time data pipeline (run in order)
python src/chunker.py              # book -> data/chunks.json
python src/embedder.py             # chunks -> data/embeddings.npy (calls Gemini API)
python src/build_chroma_index.py   # embeddings -> chroma_db/ (persistent index)

# Test in the terminal
python src/rag_pipeline.py

# Or launch the chat UI
streamlit run app.py
```

## Design decisions worth knowing (and the bugs that shaped them)

**Sentence-aware chunking.** The first version split chunks by raw word count, which cut sentences in half mid-thought — a chunk could end on "...and Fëanor was filled with a new" and the rest would land in the next chunk. Fixed by splitting into whole sentences first, then building chunks from complete sentences only, with sentence-based (not word-based) overlap. Verified: 343/345 chunks now end on proper sentence punctuation.

**Chapter-aware section boundaries.** Chunks never span two unrelated chapters — the chunker detects heading lines (`Chapter N ...` and ALL-CAPS section titles) and groups paragraphs under their actual section first, before chunking within each section.

**Naive vector store before Chroma.** Retrieval was first built by hand with numpy (`NaiveVectorStore`) — normalize vectors, dot product, sort — before swapping in Chroma. At this dataset's scale (345 vectors), Chroma provides no functional/performance improvement over the numpy version (a linear scan over 345 rows is sub-millisecond either way); it was added deliberately for the production-pattern experience — persistence, metadata handling, and a path to scale if more books were added later. Both implementations share the same `.search()` interface, so the swap was a two-line change in `rag_pipeline.py`.

**Deprecated embedding model.** The original plan (`text-embedding-004`) returned a 404 — the model had been shut down by Google on January 14, 2026, in favor of `gemini-embedding-001`. Caught and fixed by checking Google's current docs rather than assuming a remembered model name was still valid; a reminder that model names in fast-moving APIs need verifying at build time, not assumed from memory.

**Rate limit handling.** Gemini's free tier enforces a per-minute request cap, which the embedding script (345 sequential API calls) hit several times mid-run. Handled with retry/exponential backoff rather than failing the whole run — all 345 chunks embedded successfully despite several 429 errors along the way.

## Tech stack

- **Language:** Python
- **Embeddings & generation:** Google Gemini API (`gemini-embedding-001`, `gemini-2.5-flash`)
- **Vector store:** Chroma (persistent, cosine similarity), with a hand-built numpy version for comparison
- **UI:** Streamlit
- **No LangChain/LlamaIndex** — every layer (chunking, embedding calls, similarity search, prompt construction) was built directly, to understand the mechanics before relying on framework abstractions

## Possible extensions

- Multi-query retrieval (rewrite the question multiple ways, merge results) for better coverage on ambiguous questions
- Re-ranking retrieved chunks with a second, more precise model
- Formal eval set (retrieval precision/recall, answer faithfulness) beyond manual spot-checking
- Extend to more of Tolkien's legendarium (*The Lord of the Rings*, *Unfinished Tales*) — this is the scale point where Chroma's indexing would start to matter functionally