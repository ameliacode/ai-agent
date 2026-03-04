import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP
from indexer import search
from llm import ask

logging.basicConfig(level=logging.INFO)

mcp = FastMCP("RAG")


@mcp.tool()
def retrieve_docs(query: str, limit: int = 5) -> str:
    """Retrieve the most relevant document chunks for a query."""
    logging.info(f"Retrieve: {query}")
    try:
        results = search(query, limit=limit)
        if results.empty:
            return "No relevant documents found."
        return "\n\n".join(f"[{r['source']}]\n{r['text'][:400]}" for _, r in results.iterrows())
    except Exception as e:
        logging.error(f"retrieve_docs error: {e}")
        return f"An error occurred: {e}"


@mcp.tool()
def summarize_docs(query: str) -> str:
    """Retrieve relevant document chunks and return a concise summary."""
    logging.info(f"Summarize: {query}")
    try:
        results = search(query, limit=5)
        if results.empty:
            return "No relevant documents found to summarize."
        return ask(f"Summarize the following document excerpts concisely:\n\n" + "\n\n".join(results["text"].astype(str).tolist()))
    except Exception as e:
        logging.error(f"summarize_docs error: {e}")
        return f"An error occurred: {e}"


@mcp.tool()
def ask_docs(query: str) -> str:
    """Answer a question using relevant document context."""
    logging.info(f"Query: {query}")
    try:
        results = search(query, limit=5)
        context = (
            "\n\n".join(results["text"].astype(str).tolist())
            if not results.empty
            else "No relevant documents found. Answer based on general knowledge."
        )
        return ask(f"Answer the following question using the provided document context.\n\nContext:\n{context}\n\nQuestion: {query}")
    except Exception as e:
        logging.error(f"ask_docs error: {e}")
        return f"An error occurred: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
