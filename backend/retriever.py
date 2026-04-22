# backend/retriever.py

from dotenv import load_dotenv
import os

# BM25Retriever is keyword-based (sparse) search.
# BM25 = "Best Match 25" — the algorithm that powered search engines before
# neural networks. Still excellent for exact matches: proper nouns, author names,
# medical terms, version numbers — things embeddings sometimes blur over.
from langchain_community.retrievers import BM25Retriever

# Same embedding model we used in ingest.py — MUST match.
# If we used a different model here, the vectors would live in a different
# mathematical space and similarity search would silently return garbage.
from langchain_huggingface import HuggingFaceEmbeddings

# Chroma — we're LOADING the database we already built, not creating a new one
from langchain_chroma import Chroma

# CrossEncoder is our re-ranker.
# Unlike the embedding model which encodes query and document SEPARATELY,
# the cross-encoder reads (query + document) TOGETHER as a single input.
# This is slower but far more accurate — it actually reasons about relevance
# instead of just measuring vector distance.
from sentence_transformers import CrossEncoder

# Document is LangChain's container object: page_content + metadata
from langchain_core.documents import Document
RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

load_dotenv()


def load_vectorstore() -> Chroma:
    """
    Load the Chroma vector store we built during ingestion.
    We pass the same embedding model so Chroma knows how to
    interpret new query vectors at search time.
    """
    print("Loading vector store from disk...")

    # Must be identical to ingest.py — same model name, same dimensions
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Chroma() with persist_directory LOADS an existing DB.
    # Contrast with Chroma.from_documents() in ingest.py which CREATES one.
    vectorstore = Chroma(
        persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
        embedding_function=embeddings,
    )

    count = vectorstore._collection.count()
    print(f"Loaded {count} vectors from ./chroma_db")
    return vectorstore


def get_all_chunks(vectorstore: Chroma) -> list:
    """
    Reconstruct all Document objects from the Chroma database.
    We need these for BM25 — it builds a keyword index from raw text,
    not from vectors. So we pull the stored text back out of Chroma.
    """
    raw = vectorstore.get()  # returns dict with keys: ids, documents, metadatas

    # Zip the text and metadata back together into Document objects
    chunks = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]
    print(f"Reconstructed {len(chunks)} chunks for BM25 index")
    return chunks


def build_hybrid_retriever(vectorstore: Chroma, all_chunks: list) -> dict:
    """
    Returns a dict containing both retrievers separately.
    We implement our own merge logic in hybrid_search() below
    instead of relying on LangChain's EnsembleRetriever,
    which moved packages in newer LangChain versions.
    """
    print("Building hybrid retriever (dense + BM25)...")

    # Dense retriever — vector similarity search.
    # k=20: fetch 20 candidates. We fetch more than we need because
    # the re-ranker will cut to top 5. More candidates = better final quality.
    dense_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 20},
    )

    # BM25 retriever — keyword frequency search.
    # Builds an in-memory index from raw text. Fast to build at our scale.
    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    bm25_retriever.k = 20  # also fetch 20 candidates

    print("Hybrid retriever ready.")
    return {"dense": dense_retriever, "bm25": bm25_retriever}


def hybrid_search(query: str, retrievers: dict) -> list:
    """
    Merge dense and BM25 results using Reciprocal Rank Fusion (RRF).

    RRF score formula: sum of 1/(rank + 60) for each list the doc appears in.
    The constant 60 dampens the impact of very high ranks — a document at
    rank 1 in one list shouldn't completely dominate over a doc that ranks
    #2 in BOTH lists. This constant is standard in the RRF literature.

    Why implement this ourselves? It's only 15 lines, and you can now
    explain exactly how hybrid search works in any interview.

    Dense search gets weight 0.6 (60%) — semantic meaning
    BM25 gets weight 0.4 (40%)         — exact keyword matches
    """
    # Get results from both retrievers independently
    dense_docs = retrievers["dense"].invoke(query)
    bm25_docs  = retrievers["bm25"].invoke(query)

    # Build a dict keyed by document content to track scores and avoid duplicates.
    # We use content as the key because the same chunk may appear in both lists.
    scores = {}

    # Score every document from the dense retriever.
    # enumerate starts at 1 so rank 1 = best, rank 20 = worst.
    for rank, doc in enumerate(dense_docs, start=1):
        key = doc.page_content
        # Initialize score and store the doc object if we haven't seen it yet
        if key not in scores:
            scores[key] = {"score": 0.0, "doc": doc}
        # Add this doc's RRF contribution — weighted by how much we trust dense search
        scores[key]["score"] += 0.6 * (1 / (rank + 60))

    # Score every document from the BM25 retriever
    for rank, doc in enumerate(bm25_docs, start=1):
        key = doc.page_content
        if key not in scores:
            scores[key] = {"score": 0.0, "doc": doc}
        # Add this doc's RRF contribution — weighted by how much we trust BM25
        scores[key]["score"] += 0.4 * (1 / (rank + 60))

    # Sort all seen documents by their combined RRF score, highest first
    ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)

    # Return just the Document objects, up to 20 candidates for the re-ranker
    return [item["doc"] for item in ranked[:20]]


def rerank(query: str, docs: list, top_k: int = 5) -> list:
    """
    Re-rank candidate documents using a cross-encoder model.

    Two-stage retrieval is the standard production pattern:
      Stage 1 (fast)  — hybrid search fetches 20 candidates from 400 chunks
      Stage 2 (smart) — cross-encoder re-ranks 20 candidates to top 5

    Why not just use the cross-encoder on all 400 chunks directly?
    Because cross-encoders are ~50x slower than vector search.
    Running it on 20 pre-filtered candidates gives you the accuracy
    of deep reasoning at a fraction of the cost.

    This exact pattern is used by Google, Bing, and every serious
    enterprise search system — worth understanding deeply.
    """
    if not docs:
        print("No documents to re-rank.")
        return []

    # ms-marco-MiniLM-L-6-v2 is trained on Microsoft's MARCO dataset —
    # millions of real search queries with human-labeled relevance scores.
    # It's 22MB, runs fast on CPU, and is purpose-built for passage ranking.
    # Downloads once to your HuggingFace cache, then runs fully offline.
    reranker = RERANKER

    # Build input pairs: each pair is (query, one_chunk_of_text).
    # The cross-encoder reads BOTH together — this is what makes it more
    # accurate than embedding search, which encodes them separately.
    pairs = [(query, doc.page_content) for doc in docs]

    # predict() scores all pairs in one batch.
    # Scores are raw logits — not probabilities, just relative rankings.
    # Higher = more relevant to the query.
    scores = reranker.predict(pairs)

    # Sort (score, document) pairs descending, take the top_k
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in ranked[:top_k]]

    # Show scores so you can see how confident the re-ranker is.
    # A big gap between score[0] and score[1] means a clear winner.
    top_scores = [round(float(s), 3) for s, _ in ranked[:top_k]]
    print(f"Re-ranked {len(docs)} → {top_k} docs  |  top scores: {top_scores}")

    return top_docs


def retrieve(query: str, retrievers: dict) -> list:
    """
    Full two-stage retrieval pipeline — this is what api.py will call
    on every user question.

      Stage 1: hybrid_search() — fast, gets 20 candidates
      Stage 2: rerank()        — slower, cuts to 5 best matches
    """
    print(f"\nQuery: '{query}'")
    print("Stage 1: Hybrid search...")
    raw_docs = hybrid_search(query, retrievers)
    print(f"  Got {len(raw_docs)} candidates")

    print("Stage 2: Re-ranking...")
    top_docs = rerank(query, raw_docs, top_k=5)

    return top_docs


# ── Test block ────────────────────────────────────────────────────────────────
# Only runs when you execute `python retriever.py` directly.
# api.py will import the functions above without triggering this block.
if __name__ == "__main__":
    print("=== Retriever Test ===\n")

    # Load vector store from disk
    vectorstore = load_vectorstore()

    # Reconstruct chunks for BM25
    all_chunks = get_all_chunks(vectorstore)

    # Build both retrievers
    retrievers = build_hybrid_retriever(vectorstore, all_chunks)

    # Test questions tuned to your stroke/neurology PDFs.
    # Change these to match whatever your documents are about.
    test_questions = [
        "What are the effects of rhythmic auditory stimulation on gait?",
        "How does pain affect stroke patients?",
        "What rehabilitation methods improve walking after stroke?",
    ]

    for question in test_questions:
        results = retrieve(question, retrievers)

        print(f"\nTop {len(results)} results:")
        print("─" * 60)
        for i, doc in enumerate(results, 1):
            # metadata["source"] is the file path set by PyPDFLoader
            source = os.path.basename(doc.metadata.get("source", "unknown"))
            page   = doc.metadata.get("page", "?")
            print(f"\n[{i}] {source}  (page {page})")
            # First 250 chars — enough to verify relevance
            print(doc.page_content[:250].replace("\n", " ") + "...")
        print()