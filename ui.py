"""Violit web UI for the AI Research Agent.

Run with:
    python ui.py

Opens at http://localhost:8000
"""

import threading

import violit as vl

import knowledge_graph as kg
from main import run as agent_run

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = vl.App(title="AI Research Agent", theme="ocean")

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

conversation = app.state("")   # full chat history as markdown
is_running   = app.state(False)
graph_info   = app.state("")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_graph_info() -> str:
    graph = kg.load()
    projects      = [n for n in graph.nodes.values() if n.type == "project"]
    open_tickets  = [n for n in graph.nodes.values()
                     if n.type == "ticket" and n.data.get("status") == "open"]
    papers_total  = sum(1 for n in graph.nodes.values() if n.type == "paper")
    concepts_total = sum(1 for n in graph.nodes.values() if n.type == "concept")

    lines = [
        f"**Projects:** {len(projects)} | "
        f"**Papers:** {papers_total} | "
        f"**Concepts:** {concepts_total} | "
        f"**Open tickets:** {len(open_tickets)}\n"
    ]

    for proj in projects:
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
            and graph.nodes.get(e.to_id, kg.Node("", "", "", {}, "")).data.get("status") == "open"
        )
        lines.append(
            f"**[{proj.id}]** {proj.label}  \n"
            f"&nbsp;&nbsp;{proj.data.get('status','active')} · "
            f"{concept_count} concepts · {paper_count} papers · {ticket_count} tickets"
        )

    if open_tickets:
        lines.append("\n---\n**Open Tickets**")
        for ticket in open_tickets[:5]:
            priority = ticket.data.get("priority", "medium").upper()
            lines.append(f"- `[{priority}]` {ticket.label[:60]}")
        if len(open_tickets) > 5:
            lines.append(f"- *...and {len(open_tickets) - 5} more*")

    return "\n".join(lines)


def _send(query) -> None:
    """Called when the user clicks Send. query is the text_input state."""
    q = query.value.strip()
    if not q or is_running.value:
        return

    is_running.set(True)
    query.set("")   # clear input field

    conversation.set(
        conversation.value
        + f"\n\n---\n\n**You:** {q}\n\n*Agent is thinking...*"
    )

    def run_agent():
        try:
            response = agent_run(q)
        except Exception as e:
            response = f"**Error:** {e}"

        updated = conversation.value.replace(
            "*Agent is thinking...*",
            f"**Agent:**\n\n{response}",
        )
        conversation.set(updated)
        graph_info.set(_build_graph_info())
        is_running.set(False)

    threading.Thread(target=run_agent, daemon=True).start()


# ---------------------------------------------------------------------------
# Layout — Sidebar
# ---------------------------------------------------------------------------

with app.sidebar:
    app.title("Knowledge Graph")
    graph_info.set(_build_graph_info())
    app.write(graph_info)
    app.button("Refresh", on_click=lambda: graph_info.set(_build_graph_info()))
    app.divider()
    app.markdown(
        "*Auto-refreshes after each agent response.*\n\n"
        "Stored in `data/knowledge_graph.json`"
    )


# ---------------------------------------------------------------------------
# Layout — Main panel
# ---------------------------------------------------------------------------

app.title("AI Research Agent")
app.markdown(
    "Chat with the agent. Examples:\n"
    "- `add project Vision Transformers`\n"
    "- `research project proj-xxx on efficient attention mechanisms`\n"
    "- `내가 할일이 뭐야?`\n"
    "- `list projects`"
)
app.divider()

# Conversation history — reactive, updates in place
app.write(conversation)

app.divider()

# Input row: text box + send button
col_input, col_btn = app.columns(2)
with col_input:
    query = app.text_input("Ask the agent anything...", value="")
with col_btn:
    app.button("Send", on_click=lambda: _send(query))


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run()
