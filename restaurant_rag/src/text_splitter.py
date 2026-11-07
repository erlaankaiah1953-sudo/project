
import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(
    documents: list[Document],
) -> list[Document]:
    """Split documents into smaller text chunks."""

    if not documents:
        raise ValueError("No documents provided for splitting.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Document splitting produced no chunks.")

    print(f"Created {len(chunks)} chunks.")

    return chunks


if __name__ == "__main__":
    from src.load_documents import load_documents

    documents = load_documents()
    chunks = split_documents(documents)

    for index, chunk in enumerate(chunks[:3], start=1):
        print(f"\nChunk {index}:")
        print(chunk.page_content)
        print("Metadata:", chunk.metadata)
