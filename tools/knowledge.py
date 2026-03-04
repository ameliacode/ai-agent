"""LangGraph tools for persistent knowledge graph — projects, research, and tickets."""

from langchain_core.tools import tool

import deep_research as dr
import knowledge_graph as kg


@tool
def list_projects(filter_status: str = "active") -> str:
    """Show all projects with their status, concept count, paper count, and open ticket count.

    Args:
        filter_status: Filter by status ('active', 'archived', or 'all'). Default 'active'.
    """
    graph = kg.load()
    projects = [n for n in graph.nodes.values() if n.type == "project"]
    if filter_status != "all":
        projects = [p for p in projects if p.data.get("status") == filter_status]
    if not projects:
        return f"No {filter_status} projects found. Use add_project to create one."

    lines = [f"Projects ({filter_status}):\n"]
    for proj in projects:
        concept_count = sum(1 for e in graph.edges if e.from_id == proj.id and e.relation == "LEARNED")
        paper_count = sum(1 for e in graph.edges if e.from_id == proj.id and e.relation == "REFERENCES")
        open_tickets = sum(
            1 for e in graph.edges
            if e.from_id == proj.id and e.relation == "REQUIRES_HUMAN"
            and graph.nodes.get(e.to_id, kg.Node("","","",{},"")).data.get("status") == "open"
        )
        lines.append(
            f"• [{proj.id}] {proj.label}\n"
            f"  Status: {proj.data.get('status', 'active')} | "
            f"Concepts: {concept_count} | Papers: {paper_count} | Open tickets: {open_tickets}"
        )
        if proj.data.get("description"):
            lines.append(f"  {proj.data['description']}")
    return "\n".join(lines)


@tool
def add_project(name: str, description: str, tags: str = "") -> str:
    """Register a new research project in the knowledge graph.

    Args:
        name: Project name (e.g. 'Vision Transformers Research').
        description: Brief description of the project goal.
        tags: Comma-separated tags (optional).
    """
    graph = kg.load()
    project_id = kg.add_project(graph, name=name, description=description, tags=tags)
    kg.save(graph)
    return (
        f"Project created.\n"
        f"ID: {project_id}\n"
        f"Name: {name}\n"
        f"Use research_project(project_id='{project_id}', query='...') to start researching."
    )


@tool
def research_project(project_id: str, query: str, breadth: int = 4, depth: int = 2) -> str:
    """Run iterative deep research on a project, seeded from its existing graph knowledge.

    Saves papers and concepts to the graph after each iteration (crash-safe).
    Auto-creates tickets for any human action items found during research.
    Returns iteration stats, today's todo list, and report path.

    Args:
        project_id: The project ID to research (from list_projects or add_project).
        query: The research question or topic to explore.
        breadth: Number of search queries per iteration (default 4).
        depth: Number of research depth levels (default 2).
    """
    graph = kg.load()
    project_node = kg.get_project_by_id(graph, project_id)
    if project_node is None:
        return f"Project '{project_id}' not found. Use list_projects or add_project first."

    # Seed from existing graph knowledge — top-K semantically relevant to this query
    prior_learnings, prior_visited = kg.get_relevant_seed(graph, project_id, query, top_k=15)
    n_prior_learnings = len(prior_learnings)
    n_prior_papers = len(prior_visited)

    learnings = list(prior_learnings)
    visited_ids = list(prior_visited)
    all_papers: list = []
    all_human_blockers: list[str] = []
    iter_stats: list[str] = []
    current_query = query

    for i in range(depth):
        import logging
        logging.info(f"research_project iteration {i+1}/{depth} | project={project_id}")
        iter_result = dr.run_iteration(current_query, breadth, learnings, visited_ids, project_id=project_id)

        # Persist papers with structured summaries after each iteration
        for paper in iter_result.papers:
            kg.add_paper(
                graph,
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                project_id=project_id,
                summary={
                    "method": paper.method,
                    "result": paper.result,
                    "strength": paper.strength,
                    "weakness": paper.weakness,
                    "related_works": paper.related_works,
                },
            )

        # Persist key_facts as concepts
        new_facts = [f for p in iter_result.papers for f in p.key_facts]
        for fact in new_facts:
            kg.add_concept(graph, text=fact, project_id=project_id)

        # Update running state
        learnings.extend(new_facts)
        learnings = list(dict.fromkeys(learnings))
        for nid in iter_result.new_ids:
            if nid not in visited_ids:
                visited_ids.append(nid)

        all_papers.extend(iter_result.papers)
        all_human_blockers.extend(iter_result.human_blockers)
        iter_stats.append(
            f"  Iter {i+1}: {len(iter_result.papers)} papers, "
            f"{len(new_facts)} facts, {len(iter_result.human_blockers)} blockers"
        )

        kg.save(graph)  # crash-safe save after each iteration

        if i < depth - 1 and iter_result.follow_ups:
            current_query = (
                f"Follow-up for '{query}':\n"
                + "\n".join(iter_result.follow_ups[:breadth])
            )

    # Auto-create tickets from blockers (cap at 5)
    tickets_created = []
    for blocker in all_human_blockers[:5]:
        ticket_id = kg.add_ticket(
            graph,
            title=blocker[:80],
            description=blocker,
            project_id=project_id,
            priority="medium",
            context=f"Auto-generated from research on: {query}",
        )
        tickets_created.append(ticket_id)

    # Compute new items vs prior
    new_learnings = [l for l in learnings if l not in prior_learnings]
    new_visited = [v for v in visited_ids if v not in prior_visited]

    # Write todo and report
    todo = dr.write_todo(query, new_learnings, all_papers)
    report_path = dr.write_report(query=query, learnings=learnings)

    kg.save(graph)

    tickets_line = (
        f"Auto-tickets created ({len(tickets_created)}): {', '.join(tickets_created)}"
        if tickets_created else "No human blockers found."
    )
    iter_summary = "\n".join(iter_stats)

    return (
        f"Research complete for project: {project_node.label}\n\n"
        f"Prior knowledge: {n_prior_learnings} concepts, {n_prior_papers} papers\n"
        f"New this session: {len(new_learnings)} concepts, {len(new_visited)} papers\n\n"
        f"Iteration breakdown:\n{iter_summary}\n\n"
        f"{tickets_line}\n\n"
        f"Today's todo:\n{todo}\n\n"
        f"Full report: {report_path}\n"
        f"Graph updated: data/knowledge_graph.json"
    )


@tool
def create_ticket(
    title: str,
    description: str,
    project_id: str,
    priority: str = "medium",
    context: str = "",
) -> str:
    """Create a human action item when the agent hits a blocker requiring human decision.

    Tickets persist across sessions and appear in get_my_todos. Use this whenever
    you need the human to make a decision, provide data, or take an action before
    research can continue.

    Args:
        title: Short title for the ticket (e.g. 'Decide training compute budget').
        description: What needs to be done and why.
        project_id: The project this ticket belongs to.
        priority: 'low', 'medium', or 'high'. Default 'medium'.
        context: Additional context — e.g. what the agent was blocked on.
    """
    graph = kg.load()
    if kg.get_project_by_id(graph, project_id) is None:
        return f"Project '{project_id}' not found. Check list_projects for valid IDs."

    ticket_id = kg.add_ticket(
        graph,
        title=title,
        description=description,
        project_id=project_id,
        priority=priority,
        context=context,
    )
    kg.save(graph)
    return (
        f"Ticket created: #{ticket_id}\n"
        f"Title: {title}\n"
        f"Priority: {priority.upper()}\n"
        f"Project: {project_id}\n"
        f"The human will see this via get_my_todos."
    )


@tool
def get_my_todos(status: str = "open") -> str:
    """Return all tickets assigned to the human. Use this when asked '내가 할일이 뭐야?'

    Shows ticket title, project, priority, context, and creation date.

    Args:
        status: 'open' for pending items, 'done' for completed, 'all' for everything.
    """
    graph = kg.load()
    tickets = [n for n in graph.nodes.values() if n.type == "ticket"]
    if status != "all":
        tickets = [t for t in tickets if t.data.get("status") == status]

    if not tickets:
        if status == "open":
            return "할 일이 없습니다. 모든 티켓이 완료되었거나 아직 생성되지 않았습니다."
        return f"No tickets with status '{status}'."

    lines = [f"Your todos ({status}) — {len(tickets)} items:\n"]
    for ticket in sorted(tickets, key=lambda t: (t.data.get("priority", "medium"), t.created_at), reverse=True):
        proj_id = ticket.data.get("project_id", "")
        proj_node = graph.nodes.get(proj_id)
        proj_name = proj_node.label if proj_node else proj_id
        priority = ticket.data.get("priority", "medium").upper()
        lines.append(f"[{priority}] #{ticket.id} — {ticket.label}")
        lines.append(f"  Project: {proj_name} | Created: {ticket.created_at[:10]}")
        if ticket.data.get("description"):
            lines.append(f"  What to do: {ticket.data['description']}")
        if ticket.data.get("context"):
            lines.append(f"  Context: {ticket.data['context']}")
        lines.append("")
    return "\n".join(lines)


@tool
def close_ticket(ticket_id: str) -> str:
    """Mark a ticket as done after the human has completed the action.

    Args:
        ticket_id: The ticket ID (e.g. 'ticket-1234567').
    """
    graph = kg.load()
    found = kg.close_ticket(graph, ticket_id)
    if not found:
        return f"Ticket '{ticket_id}' not found. Check get_my_todos for valid IDs."
    kg.save(graph)
    return f"Ticket #{ticket_id} marked as done."
