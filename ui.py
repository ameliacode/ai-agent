"""Violit web UI for the Research Agent.

Run:  python ui.py
Opens: http://localhost:8000
"""

import threading

import violit as vl

from core import graph as kg
from main import run as agent_run

app = vl.App(title="Research Agent", theme="light")

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

conversation = app.state("")
is_running   = app.state(False)
graph_info   = app.state("")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _graph_summary() -> str:
    g        = kg.load()
    projects = [n for n in g.nodes.values() if n.type == "project"]
    tickets  = [n for n in g.nodes.values() if n.type == "ticket" and n.data.get("status") == "open"]
    papers   = sum(1 for n in g.nodes.values() if n.type == "paper")
    concepts = sum(1 for n in g.nodes.values() if n.type == "concept")

    lines = [f"{len(projects)} projects · {papers} papers · {concepts} concepts · {len(tickets)} open tickets\n"]

    for proj in projects:
        p = sum(1 for e in g.edges if e.from_id == proj.id and e.relation == "REFERENCES")
        c = sum(1 for e in g.edges if e.from_id == proj.id and e.relation == "LEARNED")
        lines.append(f"**{proj.label}**  \n`{proj.id}` · {p} papers · {c} concepts")

    if tickets:
        lines.append("\n**Open tickets**")
        for t in tickets[:5]:
            lines.append(f"- {t.label[:55]}")
        if len(tickets) > 5:
            lines.append(f"- …and {len(tickets) - 5} more")

    return "\n".join(lines)


def _send(query) -> None:
    q = query.value.strip()
    if not q or is_running.value:
        return
    is_running.set(True)
    query.set("")
    conversation.set(conversation.value + f"\n\n---\n\n**You**\n\n{q}\n\n*thinking…*")

    def _run():
        try:
            response = agent_run(q)
        except Exception as e:
            response = f"Error: {e}"
        conversation.set(
            conversation.value.replace("*thinking…*", f"**Agent**\n\n{response}")
        )
        graph_info.set(_graph_summary())
        is_running.set(False)

    threading.Thread(target=_run, daemon=True).start()


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

with app.sidebar:
    app.title("Graph")
    graph_info.set(_graph_summary())
    app.write(graph_info)
    app.button("Refresh", on_click=lambda: graph_info.set(_graph_summary()))
    app.divider()
    app.markdown("`data/knowledge_graph.json`")

app.title("Research Agent")
app.markdown("Examples: `add project X` · `research project proj-xxx on Y` · `what are my todos?`")
app.divider()

app.write(conversation)
app.divider()

col_input, col_btn = app.columns(2)
with col_input:
    query = app.text_input("Query", value="")
with col_btn:
    app.button("Send", on_click=lambda: _send(query))

if __name__ == "__main__":
    app.run()
