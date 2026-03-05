from langchain_core.tools import tool

from core import research as dr


@tool
def deep_research(query: str, breadth: int = 4, depth: int = 2) -> str:
    """Run iterative deep research on a topic using live arxiv papers.

    No graph persistence — use research_project for project-scoped research.

    Args:
        query: Research topic or question.
        breadth: Arxiv queries per iteration (default 4).
        depth: Number of depth levels (default 2).
    """
    result      = dr.run(query=query, breadth=breadth, depth=depth)
    report_path = dr.write_report(query=query, learnings=result.learnings)

    n  = len(result.learnings)
    summary = "\n".join(f"- {l}" for l in result.learnings[:5])
    more    = f"\n...and {n - 5} more." if n > 5 else ""

    return (
        f"Deep research complete.\n"
        f"Papers: {len(result.visited_ids)} | Learnings: {n}\n\n"
        f"Top learnings:\n{summary}{more}\n\n"
        f"Report: {report_path}"
    )
