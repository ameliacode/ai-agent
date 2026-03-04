from datetime import datetime, timedelta

import pandas as pd
from langchain_core.tools import tool

from indexer import search
from llm import ask


@tool
def classify_papers(query: str) -> str:
    """Classify retrieved papers as SOTA, cutting-edge, established, or outdated."""
    try:
        results = search(query, limit=8, recency_weight=1.0)
        if results.empty:
            return "No documents found to classify."

        # Batch all documents into a single LLM call instead of N+1
        excerpts = "\n\n---\n\n".join(
            f"Source: {r['source']}\n{r['text'][:1200]}"
            for _, r in results.iterrows()
        )
        return ask(
            "Classify each of the following research papers. "
            "For each paper output exactly:\n"
            "  Source: <id>\n"
            "  Status: SOTA | cutting-edge | established | outdated\n"
            "  Field: <research field>\n"
            "  Reason: <one sentence citing specific claims, metrics, or publication date>\n\n"
            "Use the published date and abstract claims to judge recency and impact.\n\n"
            + excerpts,
            max_tokens=1200,
        )
    except Exception as e:
        return f"Classification error: {e}"


@tool
def list_by_field_and_time(query: str, since_days: int = 365) -> str:
    """List documents grouped by research field, filtered to the last N days."""
    try:
        results = search(query, limit=20, recency_weight=1.0)
        if results.empty:
            return "No documents found."

        cutoff = datetime.now() - timedelta(days=since_days)
        if "created" in results.columns:
            results = results[pd.to_datetime(results["created"]) >= cutoff]

        if results.empty:
            return f"No documents found within the last {since_days} days."

        context = "\n\n".join(
            f"Source: {r['source']}\n{r['text'][:300]}"
            for _, r in results.iterrows()
        )
        return ask(
            f"Group these documents by research field. "
            f"For each group, list the sources and a one-line description. "
            f"Topic: {query}\n\n{context}",
            max_tokens=800,
        )
    except Exception as e:
        return f"Listing error: {e}"
