import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import arxiv_loader
import indexer
import llm
from config import BASE_DIR

_CONCURRENCY = int(os.getenv("RESEARCH_CONCURRENCY", "2"))


@dataclass
class ResearchResult:
    learnings: list[str] = field(default_factory=list)
    visited_ids: list[str] = field(default_factory=list)


@dataclass
class PaperSummary:
    arxiv_id: str
    title: str
    method: str
    result: str
    strength: str
    weakness: str
    related_works: list[str] = field(default_factory=list)
    key_facts: list[str] = field(default_factory=list)


@dataclass
class IterationResult:
    papers: list[PaperSummary] = field(default_factory=list)
    follow_ups: list[str] = field(default_factory=list)
    human_blockers: list[str] = field(default_factory=list)
    new_ids: list[str] = field(default_factory=list)


def _generate_queries(query: str, breadth: int, learnings: list[str]) -> list[dict]:
    """Return list of {query, goal} dicts."""
    learnings_text = "\n".join(f"- {l}" for l in learnings) if learnings else "None yet."
    prompt = (
        f"Given the research goal: {query}\n"
        f"Previous learnings:\n{learnings_text}\n\n"
        f"Generate {breadth} specific arxiv search queries to explore this topic further. "
        f"For each, provide a 'query' string and a 'goal' string explaining what research "
        f"aspect this query addresses and what insight it should uncover. "
        f'Return a JSON array of {breadth} objects. Example:\n'
        f'[{{"query": "vision transformer image classification", "goal": "Understand benchmark accuracy of ViTs vs CNNs"}}]'
    )
    response = llm.ask(prompt, max_tokens=600)
    try:
        start = response.index("[")
        end = response.rindex("]") + 1
        items = json.loads(response[start:end])
        result = []
        for item in items[:breadth]:
            if isinstance(item, dict):
                result.append({"query": str(item.get("query", query)), "goal": str(item.get("goal", query))})
            else:
                result.append({"query": str(item), "goal": query})
        return result
    except (ValueError, json.JSONDecodeError):
        logging.warning("Failed to parse query list from LLM; falling back to original query.")
        return [{"query": query, "goal": query}]


def _extract_paper_summaries_batch(
    papers: list[dict], goal: str
) -> tuple[list[PaperSummary], list[str], list[str]]:
    """
    Single LLM call: extract structured summaries for all papers + follow_ups + human_blockers.
    Papers are labelled [PAPER_0]..[PAPER_N] so IDs are matched by index.
    Returns (summaries, follow_up_questions, human_action_items).
    """
    if not papers:
        return [], [], []

    labeled = []
    for i, p in enumerate(papers[:10]):
        labeled.append(f"[PAPER_{i}] id={p['id']} title={p.get('title', 'Unknown')}\n{p['text'][:800]}")
    papers_text = "\n\n---\n\n".join(labeled)

    prompt = (
        f"Research goal: {goal}\n\n"
        f"Papers (labelled [PAPER_0] to [PAPER_{len(papers[:10])-1}]):\n\n"
        f"{papers_text}\n\n"
        "For each paper, extract a structured summary. Also extract follow-up questions and human blockers.\n\n"
        "Return ONLY a JSON object with this structure:\n"
        "{\n"
        '  "papers": [\n'
        '    {\n'
        '      "index": 0,\n'
        '      "method": "brief description of the method/approach",\n'
        '      "result": "key quantitative or qualitative results",\n'
        '      "strength": "main advantage or contribution",\n'
        '      "weakness": "limitation or gap",\n'
        '      "related_works": ["paper1", "paper2"],\n'
        '      "key_facts": ["specific fact with metrics", "another fact"]\n'
        "    }\n"
        "  ],\n"
        '  "follow_up_questions": ["question1", "question2"],\n'
        '  "human_action_items": ["action requiring human decision or data"]\n'
        "}\n"
        "No extra text. Match paper index to [PAPER_N] label."
    )
    response = llm.ask(prompt, max_tokens=2500)
    try:
        start = response.index("{")
        end = response.rindex("}") + 1
        data = json.loads(response[start:end])
    except (ValueError, json.JSONDecodeError):
        logging.warning("Failed to parse batch summaries JSON.")
        return [], [], []

    summaries = []
    for item in data.get("papers", []):
        idx = item.get("index", -1)
        if not isinstance(idx, int) or idx < 0 or idx >= len(papers):
            continue
        p = papers[idx]
        summaries.append(PaperSummary(
            arxiv_id=p["id"],
            title=p.get("title", p["id"]),
            method=str(item.get("method", "")),
            result=str(item.get("result", "")),
            strength=str(item.get("strength", "")),
            weakness=str(item.get("weakness", "")),
            related_works=[str(x) for x in item.get("related_works", [])],
            key_facts=[str(x) for x in item.get("key_facts", [])],
        ))

    follow_ups = [str(x) for x in data.get("follow_up_questions", [])]
    blockers = [str(x) for x in data.get("human_action_items", [])]
    return summaries, follow_ups, blockers


def _graph_context_records(query: str, project_id: str | None) -> list[dict]:
    """Return relevant knowledge graph nodes as paper-like records for LLM context.

    Searches the KG index for nodes relevant to `query` and formats them
    as pseudo-paper dicts so they can be mixed into _extract_paper_summaries_batch.
    Only called when a project_id is provided.
    """
    if not project_id:
        return []
    try:
        import kg_indexer
        import knowledge_graph as _kg
        graph = _kg.load()
        texts = kg_indexer.search_relevant(query, project_id, limit=5)
        records = []
        for i, text in enumerate(texts):
            records.append({
                "id": f"kg-context-{i}",
                "title": f"[From knowledge graph] {text[:80]}",
                "text": f"Title: [Knowledge Graph Context]\n\nAbstract: {text}",
            })
        return records
    except Exception as e:
        logging.warning(f"KG context lookup failed: {e}")
        return []


def _search_and_summarize(
    query_item: dict,
    visited_ids: list[str],
    breadth: int,
    project_id: str | None = None,
) -> tuple[list[PaperSummary], list[str], list[str], list[str]]:
    """Single concurrent unit: search arxiv + KG graph, then extract structured summaries."""
    q, goal = query_item["query"], query_item["goal"]

    # --- arxiv search ---
    papers = arxiv_loader.search_arxiv(q, max_results=breadth * 2)
    new_papers = [p for p in papers if p["id"] not in visited_ids]
    new_ids = [p["id"] for p in new_papers]
    if new_papers:
        indexer.add_records(new_papers)

    # --- knowledge graph search ---
    kg_records = _graph_context_records(q, project_id)

    # Merge: new arxiv papers first, then KG context as supplementary
    target = (new_papers if new_papers else papers) + kg_records

    summaries, follow_ups, blockers = _extract_paper_summaries_batch(target, goal=goal)

    # Strip out KG-context pseudo-summaries from the returned list
    # (they're already in the graph; we only needed them for LLM context)
    summaries = [s for s in summaries if not s.arxiv_id.startswith("kg-context-")]

    return summaries, follow_ups, blockers, new_ids


def run_iteration(
    query: str,
    breadth: int,
    learnings: list[str],
    visited_ids: list[str],
    project_id: str | None = None,
) -> IterationResult:
    """One depth level: generate queries → concurrent search+summarize → collect results.

    If project_id is given, each search also queries the knowledge graph via kg_indexer
    and includes relevant prior nodes as LLM context alongside fresh arxiv papers.
    """
    query_items = _generate_queries(query, breadth, learnings)
    all_summaries: list[PaperSummary] = []
    all_follow_ups: list[str] = []
    all_blockers: list[str] = []
    all_new_ids: list[str] = []

    with ThreadPoolExecutor(max_workers=_CONCURRENCY) as executor:
        futures = {
            executor.submit(_search_and_summarize, qi, list(visited_ids), breadth, project_id): qi
            for qi in query_items
        }
        for future in as_completed(futures):
            try:
                summaries, follow_ups, blockers, new_ids = future.result()
                all_summaries.extend(summaries)
                all_follow_ups.extend(follow_ups)
                all_blockers.extend(blockers)
                for nid in new_ids:
                    if nid not in all_new_ids and nid not in visited_ids:
                        all_new_ids.append(nid)
            except Exception as e:
                logging.error(f"Search/summarize failed: {e}")

    return IterationResult(
        papers=all_summaries,
        follow_ups=list(dict.fromkeys(all_follow_ups)),
        human_blockers=list(dict.fromkeys(all_blockers)),
        new_ids=all_new_ids,
    )


def run(
    query: str,
    breadth: int = 4,
    depth: int = 2,
    learnings: list[str] | None = None,
    visited_ids: list[str] | None = None,
) -> ResearchResult:
    if learnings is None:
        learnings = []
    if visited_ids is None:
        visited_ids = []

    logging.info(f"deep_research run | query={query!r} breadth={breadth} depth={depth}")
    current_query = query

    for i in range(depth):
        logging.info(f"deep_research iteration {i+1}/{depth}")
        iter_result = run_iteration(current_query, breadth, learnings, visited_ids)
        new_facts = [f for p in iter_result.papers for f in p.key_facts]
        learnings.extend(new_facts)
        learnings = list(dict.fromkeys(learnings))
        for nid in iter_result.new_ids:
            if nid not in visited_ids:
                visited_ids.append(nid)
        if i < depth - 1 and iter_result.follow_ups:
            current_query = (
                f"Follow-up for '{query}':\n"
                + "\n".join(iter_result.follow_ups[:breadth])
            )

    return ResearchResult(learnings=learnings, visited_ids=visited_ids)


def write_report(query: str, learnings: list[str]) -> str:
    learnings_text = "\n".join(f"- {l}" for l in learnings) if learnings else "No learnings collected."
    prompt = (
        f'You are a research analyst. Based on these learnings about "{query}":\n\n'
        f"{learnings_text}\n\n"
        "Write a comprehensive markdown research report with the following sections:\n"
        "# Research Report\n"
        "## Summary\n"
        "## Key Findings\n"
        "## Methodology Trends\n"
        "## Open Questions\n"
        "## References\n\n"
        "Be specific, cite paper titles and metrics where available."
    )
    report = llm.ask(prompt, max_tokens=2000)
    report_path = os.path.join(BASE_DIR, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logging.info(f"Report written to {report_path}")
    return report_path


def write_todo(query: str, learnings: list[str], paper_summaries: list[PaperSummary] | None = None) -> str:
    """Generate an actionable 8-10 item today's todo list from research results."""
    learnings_text = "\n".join(f"- {l}" for l in learnings[:20]) if learnings else "None."
    papers_text = ""
    if paper_summaries:
        lines = []
        for p in paper_summaries[:8]:
            lines.append(f"- [{p.arxiv_id}] {p.title}: {p.method} | Weakness: {p.weakness}")
        papers_text = "\nPapers found:\n" + "\n".join(lines)

    prompt = (
        f'Research topic: "{query}"\n\n'
        f"Key learnings:\n{learnings_text}"
        f"{papers_text}\n\n"
        "Generate an actionable today's todo list (8-10 items) for a researcher. "
        "Include: specific papers to read, concepts to explore, experiments to try, "
        "gaps to investigate. Be concrete and prioritized. "
        "Return as a numbered markdown list, no extra text."
    )
    return llm.ask(prompt, max_tokens=600)
