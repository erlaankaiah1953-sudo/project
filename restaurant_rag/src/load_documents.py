
import sys
from pathlib import Path

from langchain_core.documents import Document

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import KNOWLEDGE_DIR


def load_documents() -> list[Document]:
    """Load all text files from the knowledge directory."""

    if not KNOWLEDGE_DIR.exists():
        raise FileNotFoundError(
            f"Knowledge directory not found: {KNOWLEDGE_DIR}"
        )

    documents = []

    for file_path in sorted(KNOWLEDGE_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()

        if not text:
            print(f"Skipping empty file: {file_path.name}")
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": file_path.name,
                    "file_path": str(file_path),
                },
            )
        )

    if not documents:
        raise ValueError(
            f"No non-empty .txt files found in {KNOWLEDGE_DIR}"
        )

    print(f"Loaded {len(documents)} documents.")

    return documents


if __name__ == "__main__":
    docs = load_documents()

    for doc in docs:
        print(f"\nSource: {doc.metadata['source']}")
        print(doc.page_content[:150])
