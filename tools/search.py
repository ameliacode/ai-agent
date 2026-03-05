from langchain_core.tools import tool

from core.index import search
from core.llm import ask


@tool
def retrieve_docs(query: str) -> str:
    """Find the most relevant document chunks for a topic or keyword."""
    try:
        results = search(query)
        if results.empty:
            return "No relevant documents found."
        return "\n\n".join(f"[{r['source']}]\n{r['text'][:400]}" for _, r in results.iterrows())
    except Exception as e:
        return f"Retrieval error: {e}"


@tool
def summarize_docs(query: str) -> str:
    """Retrieve relevant documents and return a concise summary."""
    try:
        results = search(query)
        if results.empty:
            return "No relevant documents found."
        context = "\n\n".join(results["text"].tolist())
        return ask(f"Summarize concisely:\n\n{context}")
    except Exception as e:
        return f"Summarization error: {e}"


@tool
def ask_docs(query: str) -> str:
    """Answer a research question using indexed document context."""
    try:
        results = search(query)
        context = (
            "\n\n".join(results["text"].tolist())
            if not results.empty
            else "No documents found. Use general knowledge."
        )
        return ask(f"Answer using context.\n\nContext:\n{context}\n\nQuestion: {query}")
    except Exception as e:
        return f"QA error: {e}"
