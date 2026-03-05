"""LangGraph tools for project management — projects, research, and tickets."""

import logging

from langchain_core.tools import tool

from core import graph as kg
from core import research as dr


@tool
def list_projects(filter_status: str = "active") -> str:
    """Show all projects with status, concept count, paper count, and open ticket count.

    Args:
        filter_status: 'active', 'archived', or 'all'. Default 'active'.
    """
    g = kg.load()
    projects = [n for n in g.nodes.values() if n.type == "project"]
    if filter_status != "all":
        projects = [p for p in projects if p.data.get("status") == filter_status]
    if not projects:
        return f"No {filter_status} projects. Use add_project to create one."

    lines = [f"Projects ({filter_status}):\n"]
    for proj in projects:
        concepts = sum(1 for e in g.edges if e.from_id == proj.id and e.relation == "LEARNED")
        papers   = sum(1 for e in g.edges if e.from_id == proj.id and e.relation == "REFERENCES")
        tickets  = sum(
            1 for e in g.edges
            if e.from_id == proj.id and e.relation == "REQUIRES_HUMAN"
            and g.nodes.get(e.to_id, kg.Node("","","",{},"")).data.get("status") == "open"
        )
        lines.append(
            f"• [{proj.id}] {proj.label}\n"
            f"  {proj.data.get('status','active')} | "
            f"Concepts: {concepts} | Papers: {papers} | Tickets: {tickets}"
        )
        if proj.data.get("description"):
            lines.append(f"  {proj.data['description']}")
    return "\n".join(lines)


@tool
def add_project(name: str, description: str, tags: str = "") -> str:
    """Register a new research project.

    Args:
        name: Project name.
        description: Brief description of the research goal.
        tags: Comma-separated tags (optional).
    """
    g = kg.load()
    pid = kg.add_project(g, name=name, description=description, tags=tags)
    kg.save(g)
    return (
        f"Project created.\nID: {pid}\nName: {name}\n"
        f"Use research_project(project_id='{pid}', query='...') to start."
    )


@tool
def research_project(project_id: str, query: str, breadth: int = 4, depth: int = 2) -> str:
    """Run iterative deep research on a project, seeded from its existing graph knowledge.

    Saves papers and concepts to the graph after each iteration (crash-safe).
    Auto-creates tickets for human action items. Returns stats, todo list,
    workflow blueprint path, and report path.

    Args:
        project_id: Project ID from list_projects or add_project.
        query: Research question or topic.
        breadth: Arxiv queries per iteration (default 4).
        depth: Number of depth levels (default 2).
    """
    g = kg.load()
    project_node = kg.get_project(g, project_id)
    if project_node is None:
        return f"Project '{project_id}' not found. Use list_projects or add_project first."

    # Full prior state for accurate new-vs-old accounting
    all_prior, prior_visited = kg.project_seed(g, project_id)
    prior_set = set(all_prior)
    n_prior_learnings = len(all_prior)
    n_prior_papers    = len(prior_visited)

    # Top-K relevant learnings for seeding LLM context
    prior_learnings, _ = kg.relevant_seed(g, project_id, query, top_k=15)

    learnings   = list(prior_learnings)
    visited_ids = list(prior_visited)
    all_papers:   list[dr.PaperSummary] = []
    all_blockers: list[str] = []
    all_follow_ups: list[str] = []
    iter_stats: list[str] = []
    current_query = query

    for i in range(depth):
        logging.info(f"research_project iter {i+1}/{depth} project={project_id}")
        ir = dr.iterate(current_query, breadth, learnings, visited_ids, project_id=project_id)

        # Persist papers
        for paper in ir.papers:
            kg.add_paper(g, arxiv_id=paper.arxiv_id, title=paper.title,
                         project_id=project_id,
                         summary={"method": paper.method, "result": paper.result,
                                  "strength": paper.strength, "weakness": paper.weakness,
                                  "related_works": paper.related_works})

        # Persist key facts as concepts
        facts = [f for p in ir.papers for f in p.key_facts]
        for fact in facts:
            kg.add_concept(g, text=fact, project_id=project_id)

        learnings.extend(facts)
        learnings = list(dict.fromkeys(learnings))
        for nid in ir.new_ids:
            if nid not in visited_ids:
                visited_ids.append(nid)

        all_papers.extend(ir.papers)
        all_blockers.extend(ir.blockers)
        all_follow_ups.extend(ir.follow_ups)
        iter_stats.append(
            f"  Iter {i+1}: {len(ir.papers)} papers, {len(facts)} facts, {len(ir.blockers)} blockers"
        )

        kg.save(g)  # crash-safe

        if i < depth - 1 and ir.follow_ups:
            current_query = f"Follow-up for '{query}':\n" + "\n".join(ir.follow_ups[:breadth])

    # Auto-create tickets — deduplicated, capped at 5
    tickets_created = []
    seen: set[str] = set()
    for blocker in all_blockers:
        if blocker in seen or len(tickets_created) >= 5:
            continue
        seen.add(blocker)
        tid = kg.add_ticket(g, title=blocker[:80], description=blocker,
                            project_id=project_id, priority="medium",
                            context=f"Auto from research: {query}")
        tickets_created.append(tid)

    new_learnings = [l for l in learnings if l not in prior_set]
    new_visited   = [v for v in visited_ids if v not in prior_visited]

    # Write outputs
    todo          = dr.write_todo(query, new_learnings, all_papers)
    report_path   = dr.write_report(query=query, learnings=learnings)
    open_t_titles = [t.label for t in kg.open_tickets(g) if t.data.get("project_id") == project_id]
    workflow_path = dr.write_workflow(
        project_id=project_id,
        project_name=project_node.label,
        query=query,
        learnings=learnings,
        papers=all_papers,
        follow_ups=list(dict.fromkeys(all_follow_ups)),
        open_tickets=open_t_titles,
    )

    kg.save(g)

    tickets_line = (
        f"Tickets created ({len(tickets_created)}): {', '.join(tickets_created)}"
        if tickets_created else "No human blockers found."
    )

    return (
        f"Research complete: {project_node.label}\n\n"
        f"Prior: {n_prior_learnings} concepts, {n_prior_papers} papers\n"
        f"New: {len(new_learnings)} concepts, {len(new_visited)} papers\n\n"
        f"Iterations:\n" + "\n".join(iter_stats) + f"\n\n"
        f"{tickets_line}\n\n"
        f"Today's todo:\n{todo}\n\n"
        f"Workflow: {workflow_path}\n"
        f"Report: {report_path}\n"
        f"Graph: data/knowledge_graph.json"
    )


@tool
def create_ticket(title: str, description: str, project_id: str,
                  priority: str = "medium", context: str = "") -> str:
    """Create a human action item ticket.

    Args:
        title: Short title (e.g. 'Decide training compute budget').
        description: What needs to be done and why.
        project_id: Project this ticket belongs to.
        priority: 'low', 'medium', or 'high'.
        context: Additional context.
    """
    g = kg.load()
    if kg.get_project(g, project_id) is None:
        return f"Project '{project_id}' not found."
    tid = kg.add_ticket(g, title=title, description=description,
                        project_id=project_id, priority=priority, context=context)
    kg.save(g)
    return f"Ticket created: #{tid}\nTitle: {title}\nPriority: {priority.upper()}"


@tool
def get_my_todos(status: str = "open") -> str:
    """Return all open action item tickets across projects.

    Args:
        status: 'open', 'done', or 'all'.
    """
    g = kg.load()
    tickets = [n for n in g.nodes.values() if n.type == "ticket"]
    if status != "all":
        tickets = [t for t in tickets if t.data.get("status") == status]
    if not tickets:
        return "No open todos. All tickets are done or none have been created yet."

    lines = [f"Todos ({status}) — {len(tickets)} items:\n"]
    for t in sorted(tickets, key=lambda t: (t.data.get("priority","medium"), t.created_at), reverse=True):
        proj = g.nodes.get(t.data.get("project_id",""))
        lines.append(f"[{t.data.get('priority','medium').upper()}] #{t.id} — {t.label}")
        lines.append(f"  Project: {proj.label if proj else '?'} | Created: {t.created_at[:10]}")
        if t.data.get("description"):
            lines.append(f"  What to do: {t.data['description']}")
        if t.data.get("context"):
            lines.append(f"  Context: {t.data['context']}")
        lines.append("")
    return "\n".join(lines)


@tool
def close_ticket(ticket_id: str) -> str:
    """Mark a ticket as done.

    Args:
        ticket_id: The ticket ID (e.g. 'ticket-1234567').
    """
    g = kg.load()
    if not kg.close_ticket(g, ticket_id):
        return f"Ticket '{ticket_id}' not found."
    kg.save(g)
    return f"Ticket #{ticket_id} marked as done."
