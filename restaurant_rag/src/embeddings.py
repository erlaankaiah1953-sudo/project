
import sys
from functools import lru_cache
from pathlib import Path

from langchain_google_genai import GoogleGenerativeAIEmbeddings

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import EMBEDDING_MODEL, GOOGLE_API_KEY


@lru_cache(maxsize=1)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Create and reuse the Gemini embedding model."""

    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )


if __name__ == "__main__":
    embeddings = get_embeddings()

    sample_text = "Chicken biryani costs Rs. 250."

    vector = embeddings.embed_query(sample_text)

    print("Embedding generated successfully.")
    print("Vector length:", len(vector))
    print("First five values:", vector[:5])
