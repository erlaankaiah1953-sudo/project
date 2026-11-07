
import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RETRIEVER_K
from src.vectorstore import load_vectorstore


def get_retriever(vectorstore: Chroma | None = None):
    """Create a retriever from the Chroma vector store."""

    if vectorstore is None:
        vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )

    return retriever


def retrieve_documents(
    question: str,
    retriever=None,
) -> list[Document]:
    """Retrieve relevant documents for a question."""

    if not question or not question.strip():
        raise ValueError("Please provide a question.")

    if retriever is None:
        retriever = get_retriever()

    documents = retriever.invoke(question)

    return documents


if __name__ == "__main__":
    retriever = get_retriever()

    question = "What is the price of chicken biryani?"
    results = retrieve_documents(question, retriever)

    for index, document in enumerate(results, start=1):
        print(f"\nResult {index}")
        print("Source:", document.metadata.get("source", "Unknown"))
        print(document.page_content)
