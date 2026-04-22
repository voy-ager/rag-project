# backend/api.py
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import os
import json

from dotenv import load_dotenv

# FastAPI is the web framework. It handles incoming HTTP requests,
# routes them to the right function, and sends back responses.
# It's the Python equivalent of Express.js in Node.
from fastapi import FastAPI, HTTPException

# CORSMiddleware solves the browser security rule called "same-origin policy."
# By default, browsers block a webpage from calling an API on a DIFFERENT domain.
# Our Next.js frontend runs on localhost:3000, our API runs on localhost:8000.
# Different ports = different "origins" = blocked by default.
# CORSMiddleware tells the browser "yes, requests from localhost:3000 are allowed."
from fastapi.middleware.cors import CORSMiddleware

# StreamingResponse lets us send data back to the client in chunks as it arrives,
# rather than waiting for the full answer to be generated first.
# This is how ChatGPT shows words appearing one by one — it's streaming.
from fastapi.responses import StreamingResponse

# BaseModel is Pydantic's base class for data validation.
# When a request comes in, FastAPI uses Pydantic to automatically
# check that the JSON body has the right fields and types.
# If it doesn't, FastAPI returns a clear error before your code even runs.
from pydantic import BaseModel

# We import our retrieval pipeline functions from retriever.py.
# Notice we don't re-implement anything — api.py just orchestrates.
from retriever import (
    load_vectorstore,
    get_all_chunks,
    build_hybrid_retriever,
    hybrid_search,
    rerank,
)

# We'll use the OpenAI client directly for the LLM call.
# If you have billing set up, this uses GPT-4o-mini.
# If not, we'll use a local fallback — handled below.
from openai import OpenAI


load_dotenv()


# ── App setup ────────────────────────────────────────────────────────────────

# FastAPI() creates the application instance.
# Everything else — routes, middleware, startup logic — attaches to this object.
app = FastAPI(title="RAG API", version="1.0.0")

# Add CORS middleware so our frontend can talk to this server.
# allow_origins=["*"] allows ALL origins during development.
# In production you'd lock this down to your actual frontend URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],   # allow GET, POST, PUT, DELETE etc.
    allow_headers=["*"],   # allow any headers the browser sends
)


# ── Startup: load models once ────────────────────────────────────────────────

# These are module-level variables — they're created once when the server starts,
# then reused on every request. This is critical for performance.
# If we loaded the vector store and BM25 index inside the /ask endpoint function,
# they'd reload on every single request — adding 5-10 seconds per query.

print("Loading RAG pipeline on startup...")
vectorstore  = load_vectorstore()
all_chunks   = get_all_chunks(vectorstore)
retrievers   = build_hybrid_retriever(vectorstore, all_chunks)
print("RAG pipeline ready. Server is up.\n")


# ── Request/Response models ───────────────────────────────────────────────────

class AskRequest(BaseModel):
    # Pydantic model defines what JSON body we expect in POST /ask.
    # FastAPI will automatically validate this and return HTTP 422
    # if "question" is missing or not a string.
    question: str

class HealthResponse(BaseModel):
    status: str
    vectors_loaded: int


# ── Helper: build the LLM answer ─────────────────────────────────────────────

def get_llm_client():
    """
    Return an OpenAI client if OPENAI_API_KEY is set,
    otherwise return None so we can fall back gracefully.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key and api_key != "your-key-here":
        return OpenAI(api_key=api_key)
    return None


SYSTEM_PROMPT = """You are a helpful research assistant.
Answer the user's question using ONLY the context provided below.
If the answer is not contained in the context, say:
"I don't have enough information in the provided documents to answer that."

Be concise, accurate, and cite which document your answer comes from."""


def build_prompt(question: str, context_docs: list) -> str:
    """
    Combine the retrieved document chunks into a single context string
    and format it into the prompt we'll send to the LLM.

    Each chunk is numbered so the LLM can reference them.
    We also include the source filename so the model can cite it.
    """
    context_parts = []
    for i, doc in enumerate(context_docs, 1):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        # Format each chunk as a numbered block with its source
        context_parts.append(f"[{i}] From {source}:\n{doc.page_content}")

    # Join all chunks with a clear separator
    context = "\n\n---\n\n".join(context_parts)

    return f"""Context documents:

{context}

Question: {question}

Answer:"""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Simple health check endpoint.
    The frontend can call GET /health to verify the server is up
    and the vector store is loaded before showing the chat UI.

    @app.get("/health") is a decorator — it tells FastAPI:
    "when a GET request comes in to /health, run this function."
    """
    return {
        "status": "ok",
        "vectors_loaded": vectorstore._collection.count(),
    }


@app.post("/ask")
async def ask(request: AskRequest):
    """
    Main endpoint — takes a question, runs the full RAG pipeline,
    and streams the answer back token by token.

    POST /ask
    Body: { "question": "What are the effects of RAS on gait?" }
    Response: text/event-stream (Server-Sent Events)

    'async def' means this function is asynchronous — it can pause
    while waiting for the LLM API response without blocking the server
    from handling other requests. This is important for streaming.
    """
    question = request.question.strip()

    # Guard against empty questions
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # ── Stage 1 & 2: Retrieve and re-rank ────────────────────────────────────
    # Run our full retrieval pipeline to get the 5 most relevant chunks
    raw_docs = hybrid_search(question, retrievers)
    top_docs = rerank(question, raw_docs, top_k=5)

    # Build the prompt by injecting retrieved context
    prompt = build_prompt(question, top_docs)

    # ── Stage 3: Generate answer ──────────────────────────────────────────────
    client = get_llm_client()

    if client:
        # OpenAI streaming path — used when billing is set up
        async def stream_openai():
            """
            Generator function that yields Server-Sent Events (SSE).
            SSE is a simple protocol: each event is "data: <payload>\n\n"
            The frontend reads these events and appends text as it arrives.
            """
            try:
                # stream=True tells OpenAI to send tokens as they're generated
                # instead of waiting for the full response.
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    stream=True,
                    temperature=0,      # 0 = deterministic, no creativity
                    max_tokens=1024,    # cap response length
                )

                # Iterate over chunks as they stream in from OpenAI
                for chunk in response:
                    # Each chunk may or may not contain a text delta
                    delta = chunk.choices[0].delta.content
                    if delta:
                        # json.dumps safely escapes any special characters
                        # in the text before sending it as JSON
                        yield f"data: {json.dumps({'text': delta})}\n\n"

                # Send a special [DONE] event so the frontend knows to stop
                yield "data: [DONE]\n\n"

            except Exception as e:
                error_msg = str(e)
                # If it's a billing/quota error, fall back to local context
                # instead of showing a raw API error to the user
                if "insufficient_quota" in error_msg or "429" in error_msg:
                    yield f"data: {json.dumps({'text': 'OpenAI quota exceeded — showing retrieved context directly:\n\n'})}\n\n"
                    for i, doc in enumerate(top_docs[:3], 1):
                        source = os.path.basename(doc.metadata.get("source", "unknown"))
                        yield f"data: {json.dumps({'text': f'[Source {i}: {source}]\n'})}\n\n"
                        snippet = doc.page_content[:400].replace("\n", " ").strip()
                        yield f"data: {json.dumps({'text': snippet + '\n\n'})}\n\n"
                    yield f"data: {json.dumps({'text': '(Add OpenAI billing to get full AI-generated answers.)'})}\n\n"
                else:
                    yield f"data: {json.dumps({'error': error_msg})}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(stream_openai(), media_type="text/event-stream")

    else:
        # ── Local fallback — no API key needed ───────────────────────────────
        # When OpenAI isn't available, we build a simple answer directly
        # from the retrieved chunks. Not as polished, but fully functional
        # for testing the pipeline end-to-end without any API costs.
        async def stream_local():
            # Build a simple answer from context without an LLM
            intro = f"Based on the documents, here is what I found about your question:\n\n"
            yield f"data: {json.dumps({'text': intro})}\n\n"

            for i, doc in enumerate(top_docs[:3], 1):
                source = os.path.basename(doc.metadata.get("source", "unknown"))
                # Send the source header
                yield f"data: {json.dumps({'text': f'[Source {i}: {source}]' + chr(10)})}\n\n"
                # Send a snippet of the chunk content
                snippet = doc.page_content[:400].replace("\n", " ").strip()
                yield f"data: {json.dumps({'text': snippet + chr(10) + chr(10)})}\n\n"

            note = "(Note: Add your OpenAI API key and billing to get full AI-generated answers.)"
            yield f"data: {json.dumps({'text': note})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_local(), media_type="text/event-stream")


@app.get("/sources")
def list_sources():
    """
    Returns the list of documents currently indexed.
    Useful for the frontend to show users what files are available.
    """
    raw = vectorstore.get()
    # Extract unique source filenames from metadata
    sources = list({
        os.path.basename(meta.get("source", "unknown"))
        for meta in raw["metadatas"]
    })
    return {"sources": sorted(sources), "total_chunks": len(raw["documents"])}