
import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CHROMA_DIR, COLLECTION_NAME
from src.embeddings import get_embeddings


def create_vectorstore(
    chunks: list[Document],
) -> Chroma:
    """Create a Chroma vector store from document chunks."""

    if not chunks:
        raise ValueError("No document chunks provided.")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB.")
    print(f"Database location: {CHROMA_DIR}")

    return vectorstore


def load_vectorstore() -> Chroma:
    """Load an existing Chroma collection."""

    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            "ChromaDB does not exist yet. Run ingest.py first."
        )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


if __name__ == "__main__":
    from src.load_documents import load_documents
    from src.text_splitter import split_documents

    documents = load_documents()
    chunks = split_documents(documents)
    create_vectorstore(chunks)
