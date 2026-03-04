from langchain_core.tools import tool

from arxiv_loader import fetch_by_category, fetch_by_id, search_arxiv
from indexer import add_records


@tool
def fetch_arxiv_papers(query: str, max_results: int = 20) -> str:
    """
    Search arxiv for papers matching a query, add them to the index,
    and return a formatted list with title, authors, date, and categories.
    """
    try:
        records = search_arxiv(query, max_results=max_results)
        if not records:
            return f"No arxiv papers found for: {query}"
        added = add_records(records)
        lines = [f"Fetched and indexed {added} papers for '{query}':\n"]
        for r in records:
            # Pull metadata back out of the embedded text
            lines.append(_format_record(r))
        return "\n".join(lines)
    except Exception as e:
        return f"arxiv fetch error: {e}"


@tool
def fetch_arxiv_by_id(arxiv_id: str) -> str:
    """
    Fetch a single arxiv paper by its ID (e.g. '2401.12345'),
    add it to the index, and return its metadata.
    """
    try:
        record = fetch_by_id(arxiv_id)
        if not record:
            return f"Paper not found: {arxiv_id}"
        add_records([record])
        return f"Indexed paper:\n\n{_format_record(record)}"
    except Exception as e:
        return f"arxiv fetch error: {e}"


@tool
def fetch_arxiv_by_category(category: str, max_results: int = 20) -> str:
    """
    Fetch the latest arxiv papers in a category and add them to the index.
    Common AI categories: cs.AI, cs.LG, cs.CL, cs.CV, stat.ML
    """
    try:
        records = fetch_by_category(category, max_results=max_results)
        if not records:
            return f"No papers found in category: {category}"
        added = add_records(records)
        lines = [f"Fetched and indexed {added} papers from {category}:\n"]
        for r in records:
            lines.append(_format_record(r))
        return "\n".join(lines)
    except Exception as e:
        return f"arxiv fetch error: {e}"


def _format_record(r: dict) -> str:
    """Extract display fields from the embedded text block."""
    lines = r["text"].split("\n")
    title = next((l.removeprefix("Title: ") for l in lines if l.startswith("Title:")), r["source"])
    authors = next((l.removeprefix("Authors: ") for l in lines if l.startswith("Authors:")), "")
    categories = next((l.removeprefix("Categories: ") for l in lines if l.startswith("Categories:")), "")
    published = r["created"].strftime("%Y-%m-%d") if hasattr(r["created"], "strftime") else str(r["created"])
    return f"• [{r['source']}] {title}\n  {authors}\n  {categories} | {published}"
