from langchain_core.tools import tool

from core import arxiv as _arxiv
from core.index import add_records


@tool
def fetch_arxiv_papers(query: str, max_results: int = 20) -> str:
    """Search arxiv for papers, index them, and return a formatted list."""
    try:
        records = _arxiv.search(query, max_results=max_results)
        if not records:
            return f"No papers found for: {query}"
        add_records(records)
        lines = [f"Fetched and indexed {len(records)} papers for '{query}':\n"]
        lines += [_fmt(r) for r in records]
        return "\n".join(lines)
    except Exception as e:
        return f"Fetch error: {e}"


@tool
def fetch_arxiv_by_id(arxiv_id: str) -> str:
    """Fetch a single arxiv paper by ID (e.g. '2401.12345') and index it."""
    try:
        record = _arxiv.fetch_by_id(arxiv_id)
        if not record:
            return f"Paper not found: {arxiv_id}"
        add_records([record])
        return f"Indexed:\n\n{_fmt(record)}"
    except Exception as e:
        return f"Fetch error: {e}"


@tool
def fetch_arxiv_by_category(category: str, max_results: int = 20) -> str:
    """Fetch latest arxiv papers in a category (e.g. cs.CV, cs.LG) and index them."""
    try:
        records = _arxiv.fetch_by_category(category, max_results=max_results)
        if not records:
            return f"No papers found in: {category}"
        add_records(records)
        lines = [f"Fetched {len(records)} papers from {category}:\n"]
        lines += [_fmt(r) for r in records]
        return "\n".join(lines)
    except Exception as e:
        return f"Fetch error: {e}"


def _fmt(r: dict) -> str:
    lines = r["text"].split("\n")
    title    = next((l.removeprefix("Title: ")    for l in lines if l.startswith("Title:")),    r["source"])
    authors  = next((l.removeprefix("Authors: ")  for l in lines if l.startswith("Authors:")),  "")
    cats     = next((l.removeprefix("Categories: ") for l in lines if l.startswith("Categories:")), "")
    pub      = r["created"].strftime("%Y-%m-%d") if hasattr(r["created"], "strftime") else str(r["created"])
    return f"• [{r['source']}] {title}\n  {authors}\n  {cats} | {pub}"
