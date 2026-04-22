# RAG Research Assistant

A production-grade Retrieval-Augmented Generation (RAG) system that lets you ask questions about your own documents and get accurate, source-cited answers. Built with a hybrid search pipeline (dense vector search + BM25), cross-encoder re-ranking, FastAPI streaming backend, and a Next.js chat UI.

---

## Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────────────┐
│           Query Pipeline                │
│                                         │
│  Dense Search (embeddings) ─┐           │
│                              ├─ RRF Merge ─ Re-rank ─ LLM ─ Answer
│  BM25 Keyword Search ───────┘           │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│         Ingestion Pipeline              │
│  PDFs → Chunk → Embed → Chroma DB      │
└─────────────────────────────────────────┘
```

**Why this architecture beats a basic RAG demo:**
- **Hybrid search** combines semantic (embedding) search with BM25 keyword search using Reciprocal Rank Fusion — catches both meaning-level and exact keyword matches
- **Cross-encoder re-ranking** reads query + document together (not just vector distance) for true relevance scoring
- **Streaming API** returns tokens as they generate — no waiting for the full response
- **Eval harness** measures faithfulness, relevance, and latency — not just "does it work"

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | `all-MiniLM-L6-v2` (local, free) or `text-embedding-3-small` (OpenAI) |
| Vector DB | Chroma (local, persisted to disk) |
| Keyword search | BM25 (rank-bm25) |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) |
| LLM | GPT-4o-mini (OpenAI) |
| Backend | FastAPI + Uvicorn (Python) |
| Frontend | Next.js 14 + TypeScript + Tailwind CSS |

---

## Project Structure

```
rag-project/
├── backend/
│   ├── ingest.py       # Document loading, chunking, embedding, vector store
│   ├── retriever.py    # Hybrid search (dense + BM25) + cross-encoder re-ranking
│   ├── api.py          # FastAPI streaming endpoint
│   └── eval.py         # RAGAS evaluation harness
├── frontend/
│   └── app/
│       └── page.tsx    # Next.js chat UI with streaming
├── data/               # Drop your PDF/TXT files here
├── chroma_db/          # Auto-generated vector database (not committed)
├── .env                # API keys (not committed)
└── requirements.txt
```

---

## Setup

### Prerequisites
- Python 3.10+
- Node.js 18+
- OpenAI API key with billing enabled (for GPT-4o-mini answers)

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/rag-project.git
cd rag-project
```

### 2. Set up Python environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure environment variables
Create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-your-key-here
CHROMA_PERSIST_DIR=./chroma_db
```

### 4. Add your documents
Drop PDF or `.txt` files into the `data/` folder.

### 5. Build the vector store
```bash
python backend/ingest.py
```
This loads your documents, chunks them, embeds them locally, and saves the vector database to `./chroma_db`. Re-run this whenever you add new documents.

### 6. Start the backend
```bash
uvicorn backend.api:app --reload --port 8000
```

### 7. Start the frontend
```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000** in your browser.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server status + vector count |
| `POST` | `/ask` | Ask a question (streaming SSE response) |
| `GET` | `/sources` | List indexed documents |

### Example request
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the effects of rhythmic auditory stimulation on gait?"}'
```

---

## Engineering Decisions

**Chunk size: 512 characters with 64-character overlap**
Smaller chunks improve retrieval precision but lose context. 512 chars is a proven baseline for academic text. The 64-char overlap ensures sentences at chunk boundaries aren't lost.

**Hybrid search weights: 60% dense / 40% BM25**
Semantic search is generally more powerful for paragraph-level retrieval, but BM25 is critical for exact matches (author names, medical terms, acronyms like "RAS"). The 60/40 split was chosen after manual inspection of retrieval results.

**Re-ranking: 20 candidates → top 5**
The cross-encoder is ~50x slower than vector search, so we use a two-stage approach: fast retrieval fetches 20 candidates, slow re-ranking cuts to 5. This is the standard pattern used in production search systems at Google and Bing.

**Streaming via Server-Sent Events**
SSE is simpler than WebSockets for one-directional streaming (server → client). The frontend uses the browser's native `ReadableStream` API — no extra library needed.

---

## Eval Results

*To be updated after OpenAI billing is configured and RAGAS eval harness is run.*

| Metric | Score |
|---|---|
| Faithfulness | — |
| Answer relevancy | — |
| Context recall | — |
| Avg latency | — |

---

## Roadmap

- [ ] Add OpenAI billing and run full eval harness
- [ ] Add document upload UI (drag and drop PDFs)
- [ ] Add conversation memory (multi-turn Q&A)
- [ ] Fine-tune chunk filtering for bibliography/header removal
- [ ] Add streaming citations (show source as answer generates)
- [ ] Deploy backend to Railway, frontend to Vercel

---

## Local Development Startup

```bash
# Terminal 1 — Backend
cd rag-project
.venv\Scripts\activate        # Windows
uvicorn backend.api:app --reload --port 8000

# Terminal 2 — Frontend
cd rag-project/frontend
npm run dev

# Terminal 3 — Health check
curl http://localhost:8000/health
```