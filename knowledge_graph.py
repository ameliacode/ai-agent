"""Persistent knowledge graph — stores projects, concepts, papers, and tickets."""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import BASE_DIR

_GRAPH_PATH = Path(BASE_DIR) / "data" / "knowledge_graph.json"
_MAP_PATH = Path(BASE_DIR) / "data" / "knowledge_map.md"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Node:
    id: str
    type: str          # project | concept | paper | ticket
    label: str
    data: dict
    created_at: str


@dataclass
class Edge:
    from_id: str
    to_id: str
    relation: str      # LEARNED | REFERENCES | REQUIRES_HUMAN | SUPPORTED_BY


@dataclass
class KnowledgeGraph:
    nodes: dict = field(default_factory=dict)   # id → Node (stored as dict)
    edges: list = field(default_factory=list)   # list of Edge (stored as dict)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load() -> KnowledgeGraph:
    """Read data/knowledge_graph.json; return empty graph if not found."""
    if not _GRAPH_PATH.exists():
        return KnowledgeGraph()
    try:
        raw = json.loads(_GRAPH_PATH.read_text(encoding="utf-8"))
        nodes = {k: Node(**v) for k, v in raw.get("nodes", {}).items()}
        edges = [Edge(**e) for e in raw.get("edges", [])]
        return KnowledgeGraph(nodes=nodes, edges=edges)
    except Exception:
        return KnowledgeGraph()


def save(graph: KnowledgeGraph) -> None:
    """Write graph to JSON and regenerate knowledge_map.md."""
    _GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "nodes": {k: asdict(v) for k, v in graph.nodes.items()},
        "edges": [asdict(e) for e in graph.edges],
    }
    _GRAPH_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _MAP_PATH.write_text(to_markdown(graph), encoding="utf-8")


# ---------------------------------------------------------------------------
# Node creation helpers
# ---------------------------------------------------------------------------

def _new_id(prefix: str) -> str:
    import time
    return f"{prefix}-{int(time.time() * 1000) % 10_000_000}"


def add_project(graph: KnowledgeGraph, name: str, description: str, tags: str = "") -> str:
    """Add a project node and return its id."""
    project_id = _new_id("proj")
    node = Node(
        id=project_id,
        type="project",
        label=name,
        data={"description": description, "tags": tags, "status": "active"},
        created_at=datetime.utcnow().isoformat(),
    )
    graph.nodes[project_id] = node
    return project_id


def add_concept(graph: KnowledgeGraph, text: str, project_id: str) -> str:
    """Add a concept node and a LEARNED edge to the project. Return concept id."""
    concept_id = _new_id("concept")
    node = Node(
        id=concept_id,
        type="concept",
        label=text[:120],
        data={"text": text},
        created_at=datetime.utcnow().isoformat(),
    )
    graph.nodes[concept_id] = node
    graph.edges.append(Edge(from_id=project_id, to_id=concept_id, relation="LEARNED"))
    return concept_id


def add_paper(
    graph: KnowledgeGraph,
    arxiv_id: str,
    title: str,
    project_id: str,
    summary: dict | None = None,
) -> str:
    """Add a paper node and a REFERENCES edge to the project. Return paper id.

    If summary is provided, merges {method, result, strength, weakness, related_works}
    into the paper node's data. If node already exists, updates the summary.
    """
    paper_node_id = f"paper-{arxiv_id}"
    if paper_node_id not in graph.nodes:
        node = Node(
            id=paper_node_id,
            type="paper",
            label=title[:200],
            data={"arxiv_id": arxiv_id, "title": title},
            created_at=datetime.utcnow().isoformat(),
        )
        graph.nodes[paper_node_id] = node
    if summary:
        node = graph.nodes[paper_node_id]
        for key in ("method", "result", "strength", "weakness", "related_works"):
            if key in summary:
                node.data[key] = summary[key]
    # Add edge if not already present
    existing = {(e.from_id, e.to_id, e.relation) for e in graph.edges}
    key = (project_id, paper_node_id, "REFERENCES")
    if key not in existing:
        graph.edges.append(Edge(from_id=project_id, to_id=paper_node_id, relation="REFERENCES"))
    return paper_node_id


def add_ticket(
    graph: KnowledgeGraph,
    title: str,
    description: str,
    project_id: str,
    priority: str = "medium",
    context: str = "",
) -> str:
    """Add a ticket node + REQUIRES_HUMAN edge from project. Return ticket id."""
    ticket_id = _new_id("ticket")
    node = Node(
        id=ticket_id,
        type="ticket",
        label=title,
        data={
            "description": description,
            "priority": priority,
            "context": context,
            "status": "open",
            "project_id": project_id,
        },
        created_at=datetime.utcnow().isoformat(),
    )
    graph.nodes[ticket_id] = node
    graph.edges.append(Edge(from_id=project_id, to_id=ticket_id, relation="REQUIRES_HUMAN"))
    return ticket_id


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_relevant_seed(
    graph: KnowledgeGraph,
    project_id: str,
    query: str,
    top_k: int = 15,
) -> tuple[list[str], list[str]]:
    """Return (top_k semantically relevant learnings for query, all visited paper ids).

    Syncs this project's concept and paper summary texts into the Superlinked KG index,
    then retrieves the top-K most relevant to `query`. Always returns all paper IDs so
    already-seen papers are skipped regardless of relevance.
    """
    import kg_indexer

    records: list[dict] = []

    # Index concept nodes
    concept_ids = {
        e.to_id for e in graph.edges
        if e.from_id == project_id and e.relation == "LEARNED"
    }
    for cid in concept_ids:
        node = graph.nodes.get(cid)
        if node:
            text = node.data.get("text", node.label)
            records.append({"id": cid, "text": text, "project_id": project_id, "node_type": "concept"})

    # Index paper nodes — combine method + result + strength into a rich text
    paper_node_ids = {
        e.to_id for e in graph.edges
        if e.from_id == project_id and e.relation == "REFERENCES"
    }
    visited_ids: list[str] = []
    for pid in paper_node_ids:
        node = graph.nodes.get(pid)
        if not node:
            continue
        arxiv_id = node.data.get("arxiv_id", "")
        if arxiv_id:
            visited_ids.append(arxiv_id)
        parts = [node.data.get("title", node.label)]
        for field in ("method", "result", "strength"):
            val = node.data.get(field)
            if val:
                parts.append(val)
        text = " | ".join(p for p in parts if p)
        records.append({"id": pid, "text": text, "project_id": project_id, "node_type": "paper"})

    if not records:
        return [], [v for v in visited_ids if v]

    kg_indexer.upsert(records)
    relevant_texts = kg_indexer.search_relevant(query, project_id, limit=top_k)
    return relevant_texts, [v for v in visited_ids if v]


def get_project_seed(graph: KnowledgeGraph, project_id: str) -> tuple[list[str], list[str]]:
    """Return (learnings, visited_ids) for seeding deep_research from a project's graph."""
    # Collect concept ids linked from this project via LEARNED edges
    concept_ids = {
        e.to_id for e in graph.edges
        if e.from_id == project_id and e.relation == "LEARNED"
    }
    learnings = [
        graph.nodes[cid].data.get("text", graph.nodes[cid].label)
        for cid in concept_ids
        if cid in graph.nodes
    ]

    # Collect arxiv ids from paper nodes linked via REFERENCES edges
    paper_node_ids = {
        e.to_id for e in graph.edges
        if e.from_id == project_id and e.relation == "REFERENCES"
    }
    visited_ids = [
        graph.nodes[pid].data.get("arxiv_id", "")
        for pid in paper_node_ids
        if pid in graph.nodes
    ]
    visited_ids = [v for v in visited_ids if v]  # drop empty strings
    return learnings, visited_ids


def get_open_tickets(graph: KnowledgeGraph) -> list[Node]:
    """Return all ticket nodes with status=open."""
    return [
        node for node in graph.nodes.values()
        if node.type == "ticket" and node.data.get("status") == "open"
    ]


def close_ticket(graph: KnowledgeGraph, ticket_id: str) -> bool:
    """Set ticket data.status = done. Returns True if found."""
    node = graph.nodes.get(ticket_id)
    if node and node.type == "ticket":
        node.data["status"] = "done"
        return True
    return False


def get_project_by_id(graph: KnowledgeGraph, project_id: str) -> Optional[Node]:
    node = graph.nodes.get(project_id)
    if node and node.type == "project":
        return node
    return None


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def to_markdown(graph: KnowledgeGraph) -> str:
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    projects = [n for n in graph.nodes.values() if n.type == "project"]
    open_tickets = get_open_tickets(graph)

    lines = [f"# Knowledge Map — {date_str}\n"]

    # Projects section
    active = [p for p in projects if p.data.get("status") == "active"]
    lines.append(f"## Projects ({len(active)} active)\n")
    for proj in projects:
        # Count concepts and papers for this project
        concept_count = sum(
            1 for e in graph.edges
            if e.from_id == proj.id and e.relation == "LEARNED"
        )
        paper_count = sum(
            1 for e in graph.edges
            if e.from_id == proj.id and e.relation == "REFERENCES"
        )
        ticket_count = sum(
            1 for e in graph.edges
            if e.from_id == proj.id and e.relation == "REQUIRES_HUMAN"
            and graph.nodes.get(e.to_id, Node("","","",{},"")).data.get("status") == "open"
        )
        lines.append(f"### {proj.id}: {proj.label}")
        lines.append(
            f"Status: {proj.data.get('status', 'active')} | "
            f"Concepts: {concept_count} | Papers: {paper_count} | Open tickets: {ticket_count}"
        )
        if proj.data.get("description"):
            lines.append(f"Description: {proj.data['description']}")
        # Recent learnings (up to 5)
        concept_ids = [
            e.to_id for e in graph.edges
            if e.from_id == proj.id and e.relation == "LEARNED"
        ]
        if concept_ids:
            lines.append("Recent learnings:")
            for cid in concept_ids[-5:]:
                node = graph.nodes.get(cid)
                if node:
                    lines.append(f"- {node.label}")
        # Recent papers with structured summaries (up to 5)
        paper_node_ids = [
            e.to_id for e in graph.edges
            if e.from_id == proj.id and e.relation == "REFERENCES"
        ]
        recent_papers = [graph.nodes[pid] for pid in paper_node_ids[-5:] if pid in graph.nodes]
        if recent_papers:
            lines.append("Recent papers:")
            for pnode in recent_papers:
                arxiv_id = pnode.data.get("arxiv_id", pnode.id)
                title = pnode.data.get("title", pnode.label)
                lines.append(f"  **{arxiv_id}** — {title}")
                if pnode.data.get("method"):
                    lines.append(f"  - Method: {pnode.data['method']}")
                if pnode.data.get("result"):
                    lines.append(f"  - Result: {pnode.data['result']}")
                if pnode.data.get("weakness"):
                    lines.append(f"  - Weakness: {pnode.data['weakness']}")
        lines.append("")

    # Open tickets section
    lines.append(f"## Open Tickets ({len(open_tickets)})\n")
    for ticket in open_tickets:
        proj_id = ticket.data.get("project_id", "unknown")
        proj_node = graph.nodes.get(proj_id)
        proj_name = proj_node.label if proj_node else proj_id
        priority = ticket.data.get("priority", "medium").upper()
        lines.append(f"### [{priority}] #{ticket.id} — {ticket.label}")
        lines.append(f"Project: {proj_name} | Created: {ticket.created_at[:10]}")
        if ticket.data.get("context"):
            lines.append(f"Context: {ticket.data['context']}")
        if ticket.data.get("description"):
            lines.append(f"Description: {ticket.data['description']}")
        lines.append("")

    return "\n".join(lines)
