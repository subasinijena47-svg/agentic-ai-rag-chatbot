import os
import sys
import time
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

# Import project configs
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    PDF_PATH,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    GEMINI_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    EMBEDDING_METRIC,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def get_embeddings_instance():
    """
    Get the embedding model instance based on configured LLM_PROVIDER.
    """
    if LLM_PROVIDER == "gemini":
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set in .env file.")
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GEMINI_API_KEY
        )
    else:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in .env file.")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY
        )


def ensure_pinecone_index(index_name: str):
    """
    Ensure the Pinecone index exists with appropriate dimension.
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]

    if index_name not in existing_indexes:
        print(f"Creating Pinecone index '{index_name}' (dimension={EMBEDDING_DIMENSION}, metric={EMBEDDING_METRIC})...")
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric=EMBEDDING_METRIC,
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Index '{index_name}' successfully created.")
    else:
        print(f"Pinecone index '{index_name}' already exists.")


def run_ingestion(pdf_path: str = None, index_name: str = None):
    """
    1. Load document via PyPDFLoader
    2. Chunk document using RecursiveCharacterTextSplitter
    3. Generate embeddings and upsert into Pinecone with rate-limit handling
    """
    target_pdf = str(pdf_path or PDF_PATH)
    target_index = index_name or PINECONE_INDEX_NAME

    if not os.path.exists(target_pdf):
        raise FileNotFoundError(f"PDF file not found at: {target_pdf}")

    print(f"--- Running Ingestion using Provider: [{LLM_PROVIDER.upper()}] ---")
    print(f"1. Loading document from: {target_pdf}")
    loader = PyPDFLoader(target_pdf)
    docs = loader.load()
    print(f"   Loaded {len(docs)} pages.")

    print(f"2. Splitting text into chunks (chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(docs)
    print(f"   Generated {len(chunks)} chunks.")

    print(f"3. Initializing Pinecone index check for '{target_index}'...")
    ensure_pinecone_index(target_index)

    print(f"4. Generating embeddings with '{EMBEDDING_MODEL}' and upserting into Pinecone in batches...")
    embeddings = get_embeddings_instance()
    vector_store = PineconeVectorStore(index_name=target_index, embedding=embeddings)

    # Process in batches to strictly respect free tier rate limits (100 RPM)
    batch_size = 30
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        print(f"   -> Upserting batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        # Retry logic with backoff if rate limit is encountered
        for attempt in range(3):
            try:
                vector_store.add_documents(batch)
                break
            except Exception as err:
                if "429" in str(err) or "RESOURCE_EXHAUSTED" in str(err):
                    print("   [!] Rate limit reached. Waiting 35 seconds before retrying...")
                    time.sleep(35)
                else:
                    raise err

        if i + batch_size < len(chunks):
            print("   Waiting 25 seconds before next batch to respect Gemini Free Tier limits...")
            time.sleep(25)

    print("Ingestion pipeline completed successfully!")
    return vector_store


if __name__ == "__main__":
    run_ingestion()