
from src.load_documents import load_documents
from src.text_splitter import split_documents
from src.vectorstore import create_vectorstore


def main():
    print("Loading restaurant documents...")
    documents = load_documents()

    print("\nSplitting documents...")
    chunks = split_documents(documents)

    print("\nCreating vector database...")
    create_vectorstore(chunks)

    print("\nIngestion completed successfully!")


if __name__ == "__main__":
    main()
