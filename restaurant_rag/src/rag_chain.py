
import sys
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import LLM_MODEL
from src.retriever import get_retriever


def format_documents(documents) -> str:
    """Format retrieved documents and preserve their sources."""

    formatted_documents = []

    for document in documents:
        source = document.metadata.get("source", "Unknown")
        formatted_documents.append(
            f"Source: {source}\n"
            f"Content: {document.page_content}"
        )

    return "\n\n".join(formatted_documents)


def build_rag_chain():
    """Build the restaurant question-answering RAG chain."""

    retriever = get_retriever()

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful restaurant food delivery assistant.

Answer the customer's question using only the provided context.

Rules:
1. Do not invent menu items, prices, delivery times, or policies.
2. If the answer is not in the context, say:
   "Sorry, I could not find that information in the restaurant
   knowledge base. Please contact restaurant support."
3. If prices are provided, include the currency.
4. For allergy questions, do not guarantee a dish is safe.
   Recommend confirming ingredients and cross-contamination
   risks directly with the restaurant.
5. Give clear and simple answers.
6. Do not claim that an order has been placed, cancelled,
   refunded, or tracked because this chatbot only answers questions.

Context:
{context}

Customer question:
{question}

Answer:
"""
    )

    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda inputs: format_documents(
                retriever.invoke(inputs["question"])
            )
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


if __name__ == "__main__":
    chain = build_rag_chain()

    while True:
        question = input(
            "\nAsk about our restaurant (type 'exit' to quit): "
        ).strip()

        if question.lower() in {"exit", "quit"}:
            print("Thank you for visiting Tasty Bites!")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            answer = chain.invoke({"question": question})
            print("\nAssistant:", answer)

        except Exception as error:
            print(f"\nAn error occurred: {error}")
