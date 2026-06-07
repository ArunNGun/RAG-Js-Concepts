# RAG-Js-Concepts

A production-style **Retrieval-Augmented Generation (RAG)** pipeline built from scratch in Node.js — no LangChain, no magic. Just the raw pipeline so you understand exactly what's happening at each step.

Built as a personal AI-powered study assistant that answers questions about programming concepts from my own markdown notes.

---

## What is RAG?

LLMs are frozen in time — they don't know your private docs, your codebase, or last week's notes. RAG fixes this by retrieving relevant context from your own knowledge base and injecting it into the prompt before the LLM answers.

```
Your docs → chunk → embed → store in vector DB
                                    ↓
User question → embed → similarity search → top-K chunks → LLM prompt → answer
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Embeddings | `@xenova/transformers` — `all-MiniLM-L6-v2` (runs fully locally, no API cost) |
| Vector Store | `faiss-node` — Facebook's FAISS, IndexFlatIP (inner product on normalized vectors = cosine similarity) |
| LLM | Fuelix AI (OpenAI-compatible API) — `claude-sonnet-4-6` |
| Runtime | Node.js (CommonJS) |

No paid embedding API. No cloud vector DB. Embeddings run 100% local.

---

## Project Structure

```
RAG-Js-concepts/
├── docs/                        # Your markdown knowledge base (drop .md files here)
├── src/
│   ├── embedder/index.js        # Loads all-MiniLM-L6-v2, converts text → 384-dim vectors
│   ├── ingester/index.js        # Chunks docs, embeds, saves FAISS index + chunks.json
│   └── query/index.js           # Embeds question, retrieves chunks, calls LLM
├── vector-store/
│   ├── index.faiss              # FAISS binary index
│   └── chunks.json              # Chunk text + source metadata
├── .env                         # FUELIX_API_KEY, FUELIX_BASE_URL
└── package.json
```

---

## How It Works

### 1. Ingest (one-time)

```
node src/ingester/index.js
```

- Reads all `.md` files from `docs/`
- Splits each file into overlapping chunks (512 words, 50-word overlap)
- Embeds each chunk using `all-MiniLM-L6-v2` locally
- Saves vectors to FAISS index + chunk text to `chunks.json`

### 2. Query (every time)

```
node src/query/index.js "your question here"
```

- Embeds the question using the same local model
- Runs cosine similarity search on FAISS index → top-2 chunks
- Builds a grounded prompt: `Answer using ONLY this context: [chunks]`
- Calls Fuelix LLM (`claude-sonnet-4-6`) and returns the answer

---

## Pipeline Deep Dive

### Chunking
Split docs into overlapping pieces so no context gets cut off at boundaries. 50-word overlap means if an important concept spans a chunk boundary, it still appears in full in one of the chunks.

### Embedding
`all-MiniLM-L6-v2` converts each chunk into a 384-dimensional float array. Semantically similar text produces numerically similar vectors. Runs locally — no API calls, no cost.

### Vector Store (FAISS)
`IndexFlatIP` stores all vectors. At query time, it computes inner product (= cosine similarity for normalized vectors) between the query vector and all chunk vectors, returning the top-K closest matches.

### Retrieval + Grounding
The retrieved chunks are injected into the system prompt. The LLM is explicitly instructed to answer ONLY from the provided context — this prevents hallucination and keeps answers grounded in your actual notes.

---

## Demo

### Positive — context found, grounded answer

Query: *"what is chunking and why does chunk size matter?"*

![Positive scenario — RAG finds relevant context and answers correctly](assets/positive.png)

The system retrieved the relevant chunk about chunking from the knowledge base and generated a grounded, accurate answer.

---

### Negative — out of scope, honest refusal

Query: *"give me a recursive program in python for fibonacci series"*

![Negative scenario — RAG correctly refuses when context is missing](assets/negative.png)

No relevant chunks found. Instead of hallucinating an answer, the system correctly said **"I don't know"** — exactly what a production RAG system should do.

---

## Setup

```bash
git clone https://github.com/ArunNGun/RAG-Js-concepts.git
cd RAG-Js-concepts
npm install
```

Create `.env`:
```
FUELIX_API_KEY=your_key_here
FUELIX_BASE_URL=https://api.fuelix.ai
```

Add your `.md` files to `docs/`, then:

```bash
node src/ingester/index.js   # build the vector store
node src/query/index.js "your question"  # query it
```

---

## Key Concepts Demonstrated

- Full RAG pipeline from scratch (no LangChain abstraction)
- Local open-source embeddings (`all-MiniLM-L6-v2` via Xenova/transformers)
- FAISS vector index — inner product search on normalized vectors
- Chunk overlap strategy to preserve context at boundaries
- Grounded prompting — LLM answers ONLY from retrieved context
- Honest refusal — system says "I don't know" when context is missing (no hallucination)

---

## What's Next

- [ ] Markdown-aware chunking (split by headers, not word count)
- [ ] Similarity score threshold (filter low-confidence retrievals)
- [ ] Multi-doc knowledge base with source citations in answers
- [ ] Hybrid search (semantic + keyword BM25)
- [ ] Persistent query interface (REPL or simple web UI)

---

Built by [Arun Kumar](https://github.com/ArunNGun) — learning GenAI/RAG by building, not just reading.
