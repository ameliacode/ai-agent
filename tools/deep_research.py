from langchain_core.tools import tool

import deep_research as dr


@tool
def deep_research(query: str, breadth: int = 4, depth: int = 2) -> str:
    """Run iterative deep research on a topic using arxiv papers.

    Recursively searches arxiv, extracts key learnings, generates follow-up questions,
    and synthesizes everything into a markdown report saved to report.md.

    Args:
        query: The research topic or question to investigate.
        breadth: Number of search queries per iteration (default 4).
        depth: Number of recursive research levels (default 2).

    Returns:
        Summary of findings and path to the saved report.
    """
    result = dr.run(query=query, breadth=breadth, depth=depth)
    report_path = dr.write_report(query=query, learnings=result.learnings)

    n_learnings = len(result.learnings)
    n_papers = len(result.visited_ids)
    summary = "\n".join(f"- {l}" for l in result.learnings[:5])
    more = f"\n...and {n_learnings - 5} more." if n_learnings > 5 else ""

    return (
        f"Deep research complete.\n"
        f"Papers indexed: {n_papers} | Learnings extracted: {n_learnings}\n\n"
        f"Top learnings:\n{summary}{more}\n\n"
        f"Full report saved to: {report_path}"
    )
