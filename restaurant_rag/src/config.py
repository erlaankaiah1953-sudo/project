
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Project root: C:\restaurant_rag
BASE_DIR = Path(__file__).resolve().parent.parent

# Load the .env file from the project root
ENV_FILE = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_FILE, override=True)

# Project directories
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
CHROMA_DIR = BASE_DIR / "chroma_db"

# ChromaDB settings
COLLECTION_NAME = "restaurant_food_collection"

# API key (accept both naming conventions used across projects)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
GEMINI_API_KEY = GOOGLE_API_KEY

# Model settings
LLM_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"

# Text splitting
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Retrieval
RETRIEVER_K = 4

# Validate configuration
if not GOOGLE_API_KEY:
    raise ValueError(
        "No API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY in "
        f"{ENV_FILE}."
    )
