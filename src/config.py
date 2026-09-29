import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_PATH = DATA_DIR / "Ebook-Agentic-AI.pdf"

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip(' "')
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "").strip(' "')
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")).strip(' "')

# Provider Selection: "gemini" (Default/Active) or "openai"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini" if GEMINI_API_KEY else "openai").lower()

# Model Configurations based on Provider
if LLM_PROVIDER == "gemini":
    EMBEDDING_MODEL = "models/gemini-embedding-001"
    EMBEDDING_DIMENSION = 3072
    LLM_MODEL = "gemini-2.5-flash"
    DEFAULT_INDEX_NAME = "agentic-ai-gemini-index"
else:
    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_DIMENSION = 1536
    LLM_MODEL = "gpt-4o-mini"
    DEFAULT_INDEX_NAME = "agentic-ai-index"

# Pinecone Configuration
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", DEFAULT_INDEX_NAME)
EMBEDDING_METRIC = "cosine"

# Hyperparameters
LLM_TEMPERATURE = 0.0
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3
