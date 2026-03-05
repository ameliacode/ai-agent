import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

from config import OBSIDIAN_DIR
from core.index import search
from core.llm import ask


def _safe_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", name)


@tool
def export_to_markdown(query: str) -> str:
    """Export retrieved documents as Obsidian-compatible markdown notes."""
    try:
        results = search(query)
        if results.empty:
            return "No relevant documents found."
        excerpts = "\n\n---\n\n".join(
            f"[{r['source']}]\n{r['text'][:500]}" for _, r in results.iterrows()
        )
        summary = ask(
            "Write a 2-3 sentence summary for each excerpt. Label with source name.\n\n" + excerpts,
            max_tokens=800,
        )
        Path(OBSIDIAN_DIR).mkdir(parents=True, exist_ok=True)
        saved = []
        for _, r in results.iterrows():
            date_str = pd.to_datetime(r.get("created", datetime.now())).strftime("%Y-%m-%d")
            tag  = query.split()[0].lower() if query.strip() else "research"
            note = (
                f"---\nsource: {r['source']}\ndate: {date_str}\ntags: [research, {tag}]\n---\n\n"
                f"# {r['source']}\n\n## Excerpt\n> {r['text'][:400]}\n"
            )
            Path(OBSIDIAN_DIR).joinpath(f"{_safe_filename(r['source'])}.md").write_text(note, encoding="utf-8")
            saved.append(r["source"])
        Path(OBSIDIAN_DIR).joinpath(f"_index_{_safe_filename(query)}.md").write_text(
            f"# Research Summary: {query}\n\n{summary}\n", encoding="utf-8"
        )
        return f"Saved {len(saved)} notes to `{OBSIDIAN_DIR}`:\n" + "\n".join(f"- {s}" for s in saved)
    except Exception as e:
        return f"Export error: {e}"


@tool
def generate_todo(query: str) -> str:
    """Generate a prioritized research todo list from relevant documents."""
    try:
        results = search(query, limit=8)
        if results.empty:
            return "No documents found."
        context = "\n\n".join(f"[{r['source']}]\n{r['text'][:400]}" for _, r in results.iterrows())
        return ask(
            f"Prioritized research todo list for: {query}\n\n"
            "Include:\n- [ ] Papers to read\n- [ ] Concepts to explore\n"
            "- [ ] Experiments to try\n- [ ] Follow-up questions\n\n"
            f"Documents:\n{context}",
            max_tokens=800,
        )
    except Exception as e:
        return f"Todo error: {e}"


@tool
def generate_readme(query: str, project_name: str = "Research Project") -> str:
    """Generate a README.md for a research project based on indexed documents."""
    try:
        results = search(query)
        context = "\n\n".join(results["text"].tolist()) if not results.empty else "No documents found."
        return ask(
            f"README.md for '{project_name}' on: {query}\n"
            "Sections: Overview, Background, Key Papers, Methods, Getting Started, References.\n\n"
            f"Base it on:\n{context}",
            max_tokens=1000,
        )
    except Exception as e:
        return f"README error: {e}"
