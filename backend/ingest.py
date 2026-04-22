# backend/ingest.py

# python-dotenv lets us load our API key from the .env file
# os gives us access to environment variables (like OPENAI_API_KEY)
from dotenv import load_dotenv
import os

# LangChain document loaders — each one knows how to read a different file type
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# RecursiveCharacterTextSplitter is LangChain's best general-purpose chunker.
# It tries to split on paragraphs first, then sentences, then words, then characters.
# This keeps semantically related text together as much as possible.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# OpenAIEmbeddings calls the OpenAI API to convert text into vectors (lists of numbers).
# Each chunk of text becomes a 1536-dimensional vector that captures its meaning.
#from langchain_openai import OpenAIEmbeddings

# HuggingFaceEmbeddings runs a model locally on your machine using sentence-transformers.
# No API key needed. No cost. The model downloads once (~90MB) and is cached after that.
from langchain_huggingface import HuggingFaceEmbeddings

# Chroma is your local vector database.
# It stores your embeddings on disk so you don't have to re-embed every time you run the app.
from langchain_chroma import Chroma

from pathlib import Path  # pathlib makes file path handling clean and cross-platform


# Load the .env file so that os.environ["OPENAI_API_KEY"] works throughout the script.
# Without this line, the OpenAI client won't find your key and will throw an auth error.
load_dotenv()


def load_documents(folder: str) -> list:
    """
    Walk through every file in `folder` and load its contents
    as a list of LangChain Document objects.

    A Document is a simple container with two fields:
      - page_content: the raw text of that page/chunk
      - metadata: a dict with info like source file, page number, etc.
    """
    docs = []  # we'll accumulate all loaded pages here

    # Path(folder).rglob("*") recursively finds every file in the folder.
    # rglob means "recursive glob" — it descends into subdirectories too.
    for path in Path(folder).rglob("*"):

        if path.suffix == ".pdf":
            # PyPDFLoader reads PDFs page by page.
            # Each page becomes one Document object.
            # .load() returns a list, so we use += to extend our docs list.
            docs += PyPDFLoader(str(path)).load()
            print(f"  Loaded PDF: {path.name}")

        elif path.suffix == ".txt":
            # TextLoader reads the whole .txt file as a single Document.
            docs += TextLoader(str(path), encoding="utf-8").load()
            print(f"  Loaded TXT: {path.name}")

    print(f"\nTotal pages/documents loaded: {len(docs)}")
    return docs


def chunk_documents(docs: list) -> list:
    """
    Split each Document into smaller overlapping chunks.

    Why chunk at all? LLMs have context limits, and embedding a huge page
    as one unit makes retrieval less precise. Smaller chunks = more targeted matches.

    Why overlap? If a key sentence sits at the boundary between two chunks,
    overlap ensures neither chunk loses it entirely.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,       # each chunk is at most 512 characters long
        chunk_overlap=64,     # the last 64 characters of one chunk repeat at the start of the next
        # separators tells the splitter what to look for when deciding where to cut.
        # It tries each separator in order: first paragraph breaks, then line breaks,
        # then sentence ends, then spaces. Only uses the next one if the current one
        # would make the chunk too long.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    # Print a quick quality check so you can see what your chunks look like
    print(f"\nCreated {len(chunks)} chunks from {len(docs)} pages")
    print(f"Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} characters")
    print("\nSample chunk:")
    print("-" * 40)
    print(chunks[0].page_content[:300])  # show the first 300 chars of the first chunk
    print("-" * 40)

    return chunks

def filter_chunks(chunks: list) -> list:
    """
    Remove low-quality chunks that hurt retrieval:
    - Very short chunks (likely headers or page numbers)
    - Chunks that are mostly references/bibliography
    - Chunks with too many dots or special characters (PDF extraction artifacts)
    """
    filtered = []
    for chunk in chunks:
        text = chunk.page_content.strip()

        # Skip very short chunks — not enough content to be useful
        if len(text) < 100:
            continue

        # Skip chunks that look like bibliography entries
        # (lots of numbers, journal abbreviations, year patterns like "1992;6:185")
        semicolon_density = text.count(";") / len(text)
        if semicolon_density > 0.03:
            continue

        # Skip chunks that are mostly dots (PDF table of contents artifacts)
        dot_density = text.count(".") / len(text)
        if dot_density > 0.15:
            continue

        filtered.append(chunk)

    print(f"Filtered {len(chunks) - len(filtered)} low-quality chunks")
    print(f"Keeping {len(filtered)} high-quality chunks")
    return filtered


def build_vectorstore(chunks: list) -> Chroma:
    """
    Embed every chunk and store the vectors in a local Chroma database.

    This is the most expensive step — it calls the OpenAI embeddings API
    once per chunk. For 100 chunks of ~512 chars, this costs less than $0.001.
    """
    print("\nEmbedding chunks and building vector store...")
    #print("(This calls the OpenAI API — you'll see it in your usage dashboard)")
    print("(Running locally — no API calls, no cost)")
    # OpenAIEmbeddings automatically uses OPENAI_API_KEY from your environment.
    # text-embedding-3-small produces 1536-dimensional vectors.
    # It's OpenAI's cheapest embedding model and good enough for most RAG use cases.
    #embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    
    # all-MiniLM-L6-v2 is a small, fast, well-tested sentence embedding model.
    # It converts text into 384-dimensional vectors (vs OpenAI's 1536, but plenty for our use).
    # The first time this runs it downloads ~90MB from HuggingFace — subsequent runs are instant.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


    # Chroma.from_documents does three things in one call:
    #   1. Takes each chunk's page_content
    #   2. Calls the embeddings model to get a vector for each one
    #   3. Stores both the vector AND the original text in the database on disk
    # persist_directory tells Chroma where to save the database files.
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
    )

    # _collection.count() queries Chroma for how many vectors it stored.
    # This should match your chunk count — if it doesn't, something went wrong.
    count = vectorstore._collection.count()
    print(f"\nVector store built successfully. {count} vectors stored in ./chroma_db")

    return vectorstore


# This block only runs when you execute this file directly with `python ingest.py`.
# It won't run when other files import functions from this module.
if __name__ == "__main__":
    print("=== RAG Ingestion Pipeline ===\n")

    print("Step 1: Loading documents from ./data/")
    docs = load_documents("./data")

    if not docs:
        print("\nNo documents found in ./data/ — add some PDFs or .txt files first!")
    else:
        print("\nStep 2: Chunking documents")
        chunks = chunk_documents(docs)

        print("\nStep 2b: Filtering low-quality chunks")
        chunks = filter_chunks(chunks)

        print("\nStep 3: Building vector store")
        vectorstore = build_vectorstore(chunks)

        print("\n=== Ingestion complete! ===")
        print("Your documents are now indexed and ready to query.")
        print("Next step: run retriever.py to test the search pipeline.")