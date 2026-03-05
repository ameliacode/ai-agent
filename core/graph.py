"""Persistent knowledge graph — projects, concepts, papers, tickets."""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import BASE_DIR

_GRAPH_PATH = Path(BASE_DIR) / "data" / "knowledge_graph.json"
_MAP_PATH   = Path(BASE_DIR) / "data" / "knowledge_map.md"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Node:
    id: str
    type: str       # project | concept | paper | ticket
    label: str
    data: dict
    created_at: str


@dataclass
class Edge:
    from_id: str
    to_id: str
    relation: str   # LEARNED | REFERENCES | REQUIRES_HUMAN


@dataclass
class Graph:
    nodes: dict = field(default_factory=dict)   # id → Node
    edges: list = field(default_factory=list)   # list of Edge


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load() -> Graph:
    if not _GRAPH_PATH.exists():
        return Graph()
    try:
        raw   = json.loads(_GRAPH_PATH.read_text(encoding="utf-8"))
        nodes = {k: Node(**v) for k, v in raw.get("nodes", {}).items()}
        edges = [Edge(**e) for e in raw.get("edges", [])]
        return Graph(nodes=nodes, edges=edges)
    except Exception:
        return Graph()


def save(graph: Graph) -> None:
    _GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "nodes": {k: asdict(v) for k, v in graph.nodes.items()},
        "edges": [asdict(e) for e in graph.edges],
    }
    _GRAPH_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _MAP_PATH.write_text(to_markdown(graph), encoding="utf-8")


# ---------------------------------------------------------------------------
# Node helpers
# ---------------------------------------------------------------------------

def _new_id(prefix: str) -> str:
    import time, random
    return f"{prefix}-{int(time.time() * 1000) % 10_000_000:07d}{random.randint(0, 99):02d}"


def add_project(graph: Graph, name: str, description: str, tags: str = "") -> str:
    pid = _new_id("proj")
    graph.nodes[pid] = Node(
        id=pid, type="project", label=name,
        data={"description": description, "tags": tags, "status": "active"},
        created_at=datetime.utcnow().isoformat(),
    )
    return pid


def add_concept(graph: Graph, text: str, project_id: str) -> str:
    cid = _new_id("concept")
    graph.nodes[cid] = Node(
        id=cid, type="concept", label=text[:120],
        data={"text": text},
        created_at=datetime.utcnow().isoformat(),
    )
    graph.edges.append(Edge(from_id=project_id, to_id=cid, relation="LEARNED"))
    return cid


def add_paper(
    graph: Graph,
    arxiv_id: str,
    title: str,
    project_id: str,
    summary: dict | None = None,
) -> str:
    """Add paper node + REFERENCES edge. Merges summary fields if provided."""
    nid = f"paper-{arxiv_id}"
    if nid not in graph.nodes:
        graph.nodes[nid] = Node(
            id=nid, type="paper", label=title[:200],
            data={"arxiv_id": arxiv_id, "title": title},
            created_at=datetime.utcnow().isoformat(),
        )
    if summary:
        for key in ("method", "result", "strength", "weakness", "related_works"):
            if key in summary:
                graph.nodes[nid].data[key] = summary[key]
    existing = {(e.from_id, e.to_id, e.relation) for e in graph.edges}
    if (project_id, nid, "REFERENCES") not in existing:
        graph.edges.append(Edge(from_id=project_id, to_id=nid, relation="REFERENCES"))
    return nid


def add_ticket(
    graph: Graph,
    title: str,
    description: str,
    project_id: str,
    priority: str = "medium",
    context: str = "",
) -> str:
    tid = _new_id("ticket")
    graph.nodes[tid] = Node(
        id=tid, type="ticket", label=title,
        data={"description": description, "priority": priority,
              "context": context, "status": "open", "project_id": project_id},
        created_at=datetime.utcnow().isoformat(),
    )
    graph.edges.append(Edge(from_id=project_id, to_id=tid, relation="REQUIRES_HUMAN"))
    return tid


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_project(graph: Graph, project_id: str) -> Optional[Node]:
    node = graph.nodes.get(project_id)
    return node if node and node.type == "project" else None


def open_tickets(graph: Graph) -> list[Node]:
    return [n for n in graph.nodes.values()
            if n.type == "ticket" and n.data.get("status") == "open"]


def close_ticket(graph: Graph, ticket_id: str) -> bool:
    node = graph.nodes.get(ticket_id)
    if node and node.type == "ticket":
        node.data["status"] = "done"
        return True
    return False


def project_seed(graph: Graph, project_id: str) -> tuple[list[str], list[str]]:
    """Return (all concept texts, all paper arxiv_ids) for a project."""
    concept_ids = {e.to_id for e in graph.edges
                   if e.from_id == project_id and e.relation == "LEARNED"}
    learnings = [graph.nodes[cid].data.get("text", graph.nodes[cid].label)
                 for cid in concept_ids if cid in graph.nodes]

    paper_ids = {e.to_id for e in graph.edges
                 if e.from_id == project_id and e.relation == "REFERENCES"}
    visited = [graph.nodes[pid].data.get("arxiv_id", "")
               for pid in paper_ids if pid in graph.nodes]
    return learnings, [v for v in visited if v]


def relevant_seed(
    graph: Graph,
    project_id: str,
    query: str,
    top_k: int = 15,
) -> tuple[list[str], list[str]]:
    """Return (top_k semantically relevant learnings, all visited paper ids).

    Syncs project nodes to graph_index, queries for top-K relevant to query.
    Always returns all paper IDs for dedup regardless of relevance.
    """
    from core import graph_index

    records: list[dict] = []

    concept_ids = {e.to_id for e in graph.edges
                   if e.from_id == project_id and e.relation == "LEARNED"}
    for cid in concept_ids:
        node = graph.nodes.get(cid)
        if node:
            records.append({
                "id": cid,
                "text": node.data.get("text", node.label),
                "project_id": project_id,
                "node_type": "concept",
            })

    paper_ids = {e.to_id for e in graph.edges
                 if e.from_id == project_id and e.relation == "REFERENCES"}
    visited: list[str] = []
    for pid in paper_ids:
        node = graph.nodes.get(pid)
        if not node:
            continue
        arxiv_id = node.data.get("arxiv_id", "")
        if arxiv_id:
            visited.append(arxiv_id)
        parts = [node.data.get("title", node.label)]
        for key in ("method", "result", "strength"):
            val = node.data.get(key)
            if val:
                parts.append(val)
        records.append({
            "id": pid,
            "text": " | ".join(p for p in parts if p),
            "project_id": project_id,
            "node_type": "paper",
        })

    if not records:
        return [], [v for v in visited if v]

    graph_index.upsert(records)
    relevant = graph_index.search(query, project_id, limit=top_k)
    return relevant, [v for v in visited if v]


# ---------------------------------------------------------------------------
# Markdown export
# ---------------------------------------------------------------------------

def to_markdown(graph: Graph) -> str:
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    projects     = [n for n in graph.nodes.values() if n.type == "project"]
    tickets_open = open_tickets(graph)

    lines = [f"# Knowledge Map — {date_str}\n"]
    lines.append(f"## Projects ({sum(1 for p in projects if p.data.get('status') == 'active')} active)\n")

    for proj in projects:
        concepts = sum(1 for e in graph.edges if e.from_id == proj.id and e.relation == "LEARNED")
        papers   = sum(1 for e in graph.edges if e.from_id == proj.id and e.relation == "REFERENCES")
        tickets  = sum(
            1 for e in graph.edges
            if e.from_id == proj.id and e.relation == "REQUIRES_HUMAN"
            and graph.nodes.get(e.to_id, Node("","","",{},"")).data.get("status") == "open"
        )
        lines.append(f"### {proj.id}: {proj.label}")
        lines.append(f"Status: {proj.data.get('status','active')} | Concepts: {concepts} | Papers: {papers} | Open tickets: {tickets}")
        if proj.data.get("description"):
            lines.append(f"Description: {proj.data['description']}")

        concept_ids = [e.to_id for e in graph.edges if e.from_id == proj.id and e.relation == "LEARNED"]
        if concept_ids:
            lines.append("Recent learnings:")
            for cid in concept_ids[-5:]:
                node = graph.nodes.get(cid)
                if node:
                    lines.append(f"- {node.label}")

        paper_ids = [e.to_id for e in graph.edges if e.from_id == proj.id and e.relation == "REFERENCES"]
        recent = [graph.nodes[pid] for pid in paper_ids[-5:] if pid in graph.nodes]
        if recent:
            lines.append("Recent papers:")
            for pnode in recent:
                lines.append(f"  **{pnode.data.get('arxiv_id', pnode.id)}** — {pnode.data.get('title', pnode.label)}")
                if pnode.data.get("method"):
                    lines.append(f"  - Method: {pnode.data['method']}")
                if pnode.data.get("result"):
                    lines.append(f"  - Result: {pnode.data['result']}")
                if pnode.data.get("weakness"):
                    lines.append(f"  - Weakness: {pnode.data['weakness']}")
        lines.append("")

    lines.append(f"## Open Tickets ({len(tickets_open)})\n")
    for ticket in tickets_open:
        proj_id  = ticket.data.get("project_id", "unknown")
        proj_node = graph.nodes.get(proj_id)
        proj_name = proj_node.label if proj_node else proj_id
        priority  = ticket.data.get("priority", "medium").upper()
        lines.append(f"### [{priority}] #{ticket.id} — {ticket.label}")
        lines.append(f"Project: {proj_name} | Created: {ticket.created_at[:10]}")
        if ticket.data.get("context"):
            lines.append(f"Context: {ticket.data['context']}")
        if ticket.data.get("description"):
            lines.append(f"Description: {ticket.data['description']}")
        lines.append("")

    return "\n".join(lines)
