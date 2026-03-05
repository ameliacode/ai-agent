import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from core import arxiv as _arxiv
from core import index as _index
from core import llm as _llm
from config import BASE_DIR

# Default 1 to stay within Gemini free tier (15 req/min).
# Set RESEARCH_CONCURRENCY=2 in .env only if on a paid plan.
_CONCURRENCY = int(os.getenv("RESEARCH_CONCURRENCY", "1"))


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Result:
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
class IterResult:
    papers: list[PaperSummary] = field(default_factory=list)
    follow_ups: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    new_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gen_queries(query: str, breadth: int, learnings: list[str]) -> list[dict]:
    text = "\n".join(f"- {l}" for l in learnings) if learnings else "None yet."
    prompt = (
        f"Research goal: {query}\n"
        f"Previous learnings:\n{text}\n\n"
        f"Generate {breadth} specific arxiv search queries. "
        f"Return a JSON array of {breadth} objects with 'query' and 'goal' keys.\n"
        f'Example: [{{"query": "efficient vision transformer", "goal": "Find speed/accuracy trade-offs"}}]'
    )
    response = _llm.ask(prompt, max_tokens=600)
    try:
        start = response.index("[")
        end   = response.rindex("]") + 1
        items = json.loads(response[start:end])
        return [
            {"query": str(i.get("query", query)), "goal": str(i.get("goal", query))}
            if isinstance(i, dict) else {"query": str(i), "goal": query}
            for i in items[:breadth]
        ]
    except (ValueError, json.JSONDecodeError):
        logging.warning("Query generation parse failed; using original query.")
        return [{"query": query, "goal": query}]


def _summarize_batch(
    papers: list[dict], goal: str
) -> tuple[list[PaperSummary], list[str], list[str]]:
    """Single LLM call to extract structured summaries + follow-ups + blockers."""
    if not papers:
        return [], [], []

    labeled = [
        f"[PAPER_{i}] id={p['id']} title={p.get('title','Unknown')}\n{p['text'][:800]}"
        for i, p in enumerate(papers[:10])
    ]
    prompt = (
        f"Research goal: {goal}\n\n"
        f"Papers (labelled PAPER_0 to PAPER_{len(labeled)-1}):\n\n"
        + "\n\n---\n\n".join(labeled)
        + "\n\nReturn ONLY a JSON object:\n"
        '{"papers": [{"index": 0, "method": "", "result": "", "strength": "", '
        '"weakness": "", "related_works": [], "key_facts": []}], '
        '"follow_up_questions": [], "human_action_items": []}'
    )
    response = _llm.ask(prompt, max_tokens=2500)
    try:
        start = response.index("{")
        end   = response.rindex("}") + 1
        data  = json.loads(response[start:end])
    except (ValueError, json.JSONDecodeError):
        logging.warning("Batch summary parse failed.")
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

    return (
        summaries,
        [str(x) for x in data.get("follow_up_questions", [])],
        [str(x) for x in data.get("human_action_items", [])],
    )


def _kg_context(query: str, project_id: str | None) -> list[dict]:
    """Pull top-5 relevant graph nodes as supplementary LLM context."""
    if not project_id:
        return []
    try:
        from core import graph_index
        texts = graph_index.search(query, project_id, limit=5)
        return [
            {
                "id": f"kg-{i}",
                "title": f"[Graph context] {t[:80]}",
                "text": f"Title: [Knowledge Graph]\n\nAbstract: {t}",
            }
            for i, t in enumerate(texts)
        ]
    except Exception as e:
        logging.warning(f"KG context lookup failed: {e}")
        return []


def _search_one(
    query_item: dict,
    visited_ids: list[str],
    breadth: int,
    project_id: str | None,
) -> tuple[list[PaperSummary], list[str], list[str], list[str]]:
    """One concurrent unit: arxiv search + KG context → batch summarize."""
    q, goal = query_item["query"], query_item["goal"]

    papers     = _arxiv.search(q, max_results=breadth * 2)
    new_papers = [p for p in papers if p["id"] not in visited_ids]
    new_ids    = [p["id"] for p in new_papers]
    if new_papers:
        _index.add_records(new_papers)

    target    = (new_papers if new_papers else papers) + _kg_context(q, project_id)
    summaries, follow_ups, blockers = _summarize_batch(target, goal=goal)

    # Strip KG context pseudo-entries from returned summaries
    summaries = [s for s in summaries if not s.arxiv_id.startswith("kg-")]
    return summaries, follow_ups, blockers, new_ids


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def iterate(
    query: str,
    breadth: int,
    learnings: list[str],
    visited_ids: list[str],
    project_id: str | None = None,
) -> IterResult:
    """One research depth level: generate queries → concurrent search + summarize."""
    query_items   = _gen_queries(query, breadth, learnings)
    all_summaries: list[PaperSummary] = []
    all_follow_ups: list[str] = []
    all_blockers:   list[str] = []
    all_new_ids:    list[str] = []

    with ThreadPoolExecutor(max_workers=_CONCURRENCY) as executor:
        futures = {
            executor.submit(_search_one, qi, list(visited_ids), breadth, project_id): qi
            for qi in query_items
        }
        for future in as_completed(futures):
            try:
                s, f, b, ids = future.result()
                all_summaries.extend(s)
                all_follow_ups.extend(f)
                all_blockers.extend(b)
                for nid in ids:
                    if nid not in all_new_ids and nid not in visited_ids:
                        all_new_ids.append(nid)
            except Exception as e:
                logging.error(f"Search failed: {e}")

    return IterResult(
        papers=all_summaries,
        follow_ups=list(dict.fromkeys(all_follow_ups)),
        blockers=list(dict.fromkeys(all_blockers)),
        new_ids=all_new_ids,
    )


def run(
    query: str,
    breadth: int = 4,
    depth: int = 2,
    learnings: list[str] | None = None,
    visited_ids: list[str] | None = None,
) -> Result:
    """Iterative deep research — no graph persistence (use tools/projects.py for that)."""
    learnings   = learnings or []
    visited_ids = visited_ids or []
    current     = query

    for i in range(depth):
        logging.info(f"research.run iteration {i+1}/{depth}")
        ir = iterate(current, breadth, learnings, visited_ids)
        facts = [f for p in ir.papers for f in p.key_facts]
        learnings.extend(facts)
        learnings = list(dict.fromkeys(learnings))
        for nid in ir.new_ids:
            if nid not in visited_ids:
                visited_ids.append(nid)
        if i < depth - 1 and ir.follow_ups:
            current = f"Follow-up for '{query}':\n" + "\n".join(ir.follow_ups[:breadth])

    return Result(learnings=learnings, visited_ids=visited_ids)


def write_report(query: str, learnings: list[str]) -> str:
    text = "\n".join(f"- {l}" for l in learnings) if learnings else "No learnings."
    prompt = (
        f'Research analyst report on "{query}":\n\n{text}\n\n'
        "Write a comprehensive markdown report:\n"
        "# Research Report\n## Summary\n## Key Findings\n"
        "## Methodology Trends\n## Open Questions\n## References\n\n"
        "Be specific — cite paper titles and metrics."
    )
    report = _llm.ask(prompt, max_tokens=2000)
    path   = os.path.join(BASE_DIR, "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    logging.info(f"Report written to {path}")
    return path


def write_todo(query: str, learnings: list[str], papers: list[PaperSummary] | None = None) -> str:
    text = "\n".join(f"- {l}" for l in learnings[:20]) if learnings else "None."
    papers_text = (
        "\nPapers:\n" + "\n".join(
            f"- [{p.arxiv_id}] {p.title}: {p.method} | Weakness: {p.weakness}"
            for p in (papers or [])[:8]
        )
    ) if papers else ""
    prompt = (
        f'Research topic: "{query}"\n\nKey learnings:\n{text}{papers_text}\n\n'
        "Generate an actionable today's todo list (8-10 items). "
        "Include: papers to read, concepts to explore, experiments to try. "
        "Return a numbered markdown list only."
    )
    return _llm.ask(prompt, max_tokens=600)


def write_workflow(
    project_id: str,
    project_name: str,
    query: str,
    learnings: list[str],
    papers: list[PaperSummary],
    follow_ups: list[str],
    open_tickets: list[str],
) -> str:
    """Update data/workflows/{project_id}.md — cumulative long-term blueprint."""
    workflow_dir  = os.path.join(BASE_DIR, "data", "workflows")
    os.makedirs(workflow_dir, exist_ok=True)
    path = os.path.join(workflow_dir, f"{project_id}.md")

    prior = ""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            prior = f.read()

    learnings_text = "\n".join(f"- {l}" for l in learnings[:30]) or "None yet."
    papers_text    = "\n".join(
        f"- [{p.arxiv_id}] {p.title}: {p.method} → {p.result}"
        for p in papers[:10]
    ) or "None."
    followups_text = "\n".join(f"- {q}" for q in follow_ups[:6]) or "None."
    tickets_text   = "\n".join(f"- {t}" for t in open_tickets) or "None."

    prompt = (
        (f"Prior workflow to update:\n{prior}\n\n" if prior else "No prior workflow.\n\n")
        + f"Project: {project_name} ({project_id})\n"
        + f"Latest query: {query}\n\n"
        + f"Accumulated learnings ({len(learnings)}):\n{learnings_text}\n\n"
        + f"Papers this session:\n{papers_text}\n\n"
        + f"Open follow-ups:\n{followups_text}\n\n"
        + f"Human blockers:\n{tickets_text}\n\n"
        + "Generate an updated workflow.md blueprint. Sections:\n"
        + "## Research Goal\n## Progress So Far\n## Key Findings\n"
        + "## Open Questions\n## Next Steps\n## Human Blockers\n## Methodology Notes\n\n"
        + "Be specific and cumulative — useful in 3 months. "
        + f"Start with: # Workflow: {project_name}"
    )
    workflow = _llm.ask(prompt, max_tokens=2000)
    with open(path, "w", encoding="utf-8") as f:
        f.write(workflow)
    logging.info(f"Workflow updated: {path}")
    return path
