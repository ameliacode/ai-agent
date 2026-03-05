# AI Research Agent

A LangGraph-based autonomous research agent that searches arxiv via live API, builds a persistent knowledge graph of every paper it finds, and manages project-scoped research with structured summaries, crash-safe iterative saves, and human-in-the-loop ticketing.

> **Paper source:** Only the live [arxiv API](https://arxiv.org/search/) is used. No local CSV files are involved — every paper is fetched fresh and indexed in real time.

---

## What It Does

```
input: query + project_id (+ optional depth, breadth)
         ↓
[iteration 1]  arxiv API search
               → each paper found → saved as a node in the knowledge graph
                 node stores: method / result / strength / weakness / key_facts
               → save papers + concepts to graph   ← after every iteration
               → collect human blockers → auto-tickets
               → generate follow-up questions
[iteration 2]  seeded from iter 1 key_facts + visited_ids
               → Superlinked KG index ranks prior knowledge by relevance to new query
               → repeat until depth reached
         ↓
output: knowledge graph (JSON) + backlog tickets + today's todo list + report.md
        all scoped to a project
```

---

## Architecture

```
main.py                    LangGraph ReAct agent (REPL)
ui.py                      Violit web UI (chat + knowledge graph sidebar)
├── llm.py                 ChatOpenAI wrapper (gpt-4o-mini default)
├── config.py              Env vars: MODEL, BASE_DIR, DOCS_DIR
│
├── arxiv_loader.py        Live arxiv API → {id, title, text, source, created}
│                          NO CSV files — all data fetched fresh via API
│
├── indexer.py             Superlinked document index
│                            TextSimilaritySpace (all-mpnet-base-v2, chunk 200)
│                            RecencySpace (1/2/3yr periods, −0.25 old-paper penalty)
│                            Indexes: local docs (DOCS_DIR) + every arxiv paper fetched
│
├── kg_indexer.py          Superlinked knowledge graph index
│                            TextSimilaritySpace over concept + paper summary texts
│                            search_relevant(query, project_id, top_k)
│                            → returns top-K semantically relevant priors for seeding
│
├── deep_research.py       Core research engine
│                            PaperSummary: method/result/strength/weakness/key_facts
│                            IterationResult: papers/follow_ups/human_blockers/new_ids
│                            _extract_paper_summaries_batch() — single LLM call per iter
│                            run_iteration() → concurrent search + batch LLM summarize
│                            run() → iterative depth loop
│                            write_report() → report.md
│                            write_todo() → 8–10 item actionable list
│
├── knowledge_graph.py     Persistent graph — saved as JSON metadata
│                            data/knowledge_graph.json  ← all nodes + edges
│                            data/knowledge_map.md      ← auto-generated human view
│                            Nodes: project | concept | paper | ticket
│                            Edges: LEARNED | REFERENCES | REQUIRES_HUMAN
│                            get_relevant_seed() → top-K semantically relevant priors
│                            add_paper(..., summary=dict) → structured data in node
│
└── tools/
    ├── knowledge.py       list_projects, add_project, research_project,
    │                      create_ticket, get_my_todos, close_ticket
    ├── deep_research.py   deep_research (stateless, no graph)
    ├── fetch.py           fetch_arxiv_papers, fetch_arxiv_by_id, fetch_arxiv_by_category
    ├── search.py          retrieve_docs, summarize_docs, ask_docs
    ├── analyze.py         classify_papers, list_by_field_and_time
    └── output.py          export_to_markdown, generate_readme, generate_todo
```

---

## Paper → Knowledge Graph Node Flow

Every paper the agent finds during research is immediately persisted as a structured node:

```
arxiv API
  └─ search_arxiv(query)
       └─ returns {id, title, text, source, created}
            │
            ├─ indexer.add_records()          → indexed in Superlinked doc index
            │                                   (for retrieve_docs / ask_docs)
            │
            └─ _extract_paper_summaries_batch()
                 LLM extracts per paper:
                   method, result, strength, weakness,
                   related_works, key_facts
                 │
                 └─ knowledge_graph.add_paper(arxiv_id, title, project_id,
                                              summary={method, result,
                                                       strength, weakness,
                                                       related_works})
                      └─ saved to data/knowledge_graph.json
                           paper node data:
                             {arxiv_id, title,
                              method, result,
                              strength, weakness,
                              related_works: [...]}
                         + REFERENCES edge: project → paper

                    each key_fact → add_concept() → concept node
                                  + LEARNED edge: project → concept
```

The graph is saved to disk **after every iteration**, so partial results are never lost.

---

## Knowledge Graph — JSON Metadata Storage

The graph persists entirely in `data/knowledge_graph.json`:

```json
{
  "nodes": {
    "proj-1234567": {
      "id": "proj-1234567",
      "type": "project",
      "label": "Vision Transformers",
      "data": {"description": "...", "tags": "cv,transformers", "status": "active"},
      "created_at": "2026-03-04T10:00:00"
    },
    "paper-2305.07027": {
      "id": "paper-2305.07027",
      "type": "paper",
      "label": "EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction",
      "data": {
        "arxiv_id": "2305.07027",
        "title": "EfficientViT: Multi-Scale Linear Attention...",
        "method": "Multi-scale linear attention with ReLU kernel",
        "result": "3.4× faster than SegFormer on ADE20K at same accuracy",
        "strength": "Hardware-efficient — no softmax, runs well on mobile",
        "weakness": "Evaluated on dense prediction only, not classification",
        "related_works": ["SegFormer", "Swin Transformer"]
      },
      "created_at": "2026-03-04T10:05:00"
    },
    "concept-9876543": {
      "id": "concept-9876543",
      "type": "concept",
      "label": "Linear attention replaces softmax to reduce O(n²) to O(n)",
      "data": {"text": "Linear attention replaces softmax..."},
      "created_at": "2026-03-04T10:05:00"
    },
    "ticket-1111111": {
      "id": "ticket-1111111",
      "type": "ticket",
      "label": "Decide training compute budget for ViT fine-tuning",
      "data": {
        "description": "...",
        "priority": "medium",
        "status": "open",
        "project_id": "proj-1234567",
        "context": "Auto-generated from research on: vision transformer efficiency"
      },
      "created_at": "2026-03-04T10:06:00"
    }
  },
  "edges": [
    {"from_id": "proj-1234567", "to_id": "paper-2305.07027", "relation": "REFERENCES"},
    {"from_id": "proj-1234567", "to_id": "concept-9876543",  "relation": "LEARNED"},
    {"from_id": "proj-1234567", "to_id": "ticket-1111111",   "relation": "REQUIRES_HUMAN"}
  ]
}
```

`data/knowledge_map.md` is auto-regenerated on every save — a human-readable view with paper method/result/weakness per project.

---

## Superlinked — Searching the Knowledge Graph

Two independent Superlinked in-memory indices power this agent:

### 1. Document Index (`indexer.py`)
Used for general document search (`retrieve_docs`, `ask_docs`).
- **TextSimilaritySpace** — semantic embedding, 200-token chunks
- **RecencySpace** — boosts recent papers, penalizes old ones
- Sources: local docs on startup + every arxiv paper fetched during research

### 2. Knowledge Graph Index (`kg_indexer.py`)
Used to seed each research iteration with relevant prior knowledge — not all of it.

```
On each new research_project() call:
  all concept nodes + paper nodes for this project
    → each converted to a text string
       concept: "Linear attention replaces softmax to reduce O(n²) to O(n)"
       paper:   "EfficientViT | Multi-scale linear attention | 3.4× faster | Hardware-efficient"
    → kg_indexer.upsert(records)

  search_relevant(new_query, project_id, top_k=15)
    → TextSimilaritySpace ranks all texts by cosine similarity to query
    → returns top-15 most relevant
    → used as prior learnings to seed _generate_queries() and LLM context

  all paper arxiv_ids always returned (for visited_ids dedup)
```

**Why not use all prior learnings?**
A project after 10 sessions may have 500+ concept nodes. Sending all 500 as LLM context wastes tokens and dilutes signal. The KG index keeps seeding focused and token-efficient regardless of graph size.

---

## Tools Reference

### Project & Knowledge Graph

| Tool | Description |
|---|---|
| `add_project` | Register a new research project with name, description, tags |
| `list_projects` | Show all projects with concept / paper / ticket counts |
| `research_project` | Full iterative deep research scoped to a project |
| `create_ticket` | Manually create a human action item |
| `get_my_todos` | List all open tickets and action items |
| `close_ticket` | Mark a ticket as done |

### arxiv & Document Search

| Tool | Description |
|---|---|
| `fetch_arxiv_papers` | Search arxiv API, index results |
| `fetch_arxiv_by_id` | Fetch and index a single paper by ID |
| `fetch_arxiv_by_category` | Fetch latest from a category (e.g. `cs.CV`, `cs.LG`) |
| `retrieve_docs` | Semantic search over all indexed documents |
| `summarize_docs` | Retrieve + LLM summarize |
| `ask_docs` | Retrieve + LLM answer a question with context |

### Standalone Research & Output

| Tool | Description |
|---|---|
| `deep_research` | Iterative arxiv research without graph persistence |
| `classify_papers` | LLM-classify indexed papers into research fields |
| `list_by_field_and_time` | Group papers by field and publication date |
| `export_to_markdown` | Export indexed content to markdown |
| `generate_readme` | LLM-write a README from indexed content |
| `generate_todo` | LLM-generate a todo list from indexed content |

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env and add your OpenAI key
```

`.env`:
```env
GOOGLE_API_KEY=AIza...
MODEL=gemini-2.0-flash-lite   # optional, default: gemini-2.0-flash-lite
DOCS_DIR=./docs               # optional: local docs indexed on startup
```

Get a free API key at [aistudio.google.com](https://aistudio.google.com) — free tier: 15 req/min · 500 req/day · 250k tokens/min.

---

## Usage

### Option A — Web UI (recommended)

```bash
python ui.py
```

Opens at `http://localhost:8000`. Chat with the agent, view the knowledge graph sidebar, and track open tickets — all in the browser.

### Option B — Terminal REPL

```bash
python main.py
```

---

## Usage Guide

### 1. Create a project

```
add project Vision Transformers
```
Returns a project ID like `proj-1234567`. All research, papers, and tickets are scoped to this ID.

---

### 2. Run research on the project

```
research project proj-1234567 on vision transformer efficiency
```

What happens:
- Generates 4 arxiv search queries (breadth=4)
- Fetches papers via live arxiv API (no CSV)
- Extracts per-paper: method / result / strength / weakness / key_facts
- Saves each paper as a structured node in `data/knowledge_graph.json`
- Runs a second iteration seeded from iteration 1's key facts
- Auto-creates tickets for any human decisions needed
- Writes `report.md` and a today's todo list

Output:
```
Research complete for project: Vision Transformers

Prior knowledge: 0 concepts, 0 papers
New this session: 32 concepts, 11 papers

Iteration breakdown:
  Iter 1: 6 papers, 18 facts, 1 blocker
  Iter 2: 5 papers, 14 facts, 0 blockers

Auto-tickets created (1): ticket-9876543

Today's todo:
1. Read EfficientViT (2305.07027) — focus on mobile benchmark section
2. Compare DeiT-III vs ViT-H on ImageNet-21K pretraining setup
...

Full report: /path/to/report.md
```

---

### 3. Check your todo list

```
what are my todos?
```
or
```
what are my todos?
```

Shows all open tickets from all projects, sorted by priority.

---

### 4. Return to the same project later

```
research project proj-1234567 on attention mechanism efficiency
```

The agent:
- Loads the existing graph (32 concepts, 11 papers from last session)
- Runs `get_relevant_seed()` — Superlinked ranks the 32 prior concepts by relevance to "attention mechanism efficiency" → returns top-15
- Skips all 11 previously seen papers (dedup via visited_ids)
- Builds on what it already knows

---

### 5. Browse and manage the graph

```
list projects                          → show all projects
fetch arxiv papers on ViT pruning      → manual fetch + index
ask docs what is token merging?        → QA over indexed papers
close ticket ticket-9876543            → mark a blocker as done
```

---

### 6. Adjust research depth and breadth

| Parameter | Default | Effect |
|---|---|---|
| `breadth` | 4 | arxiv queries per iteration — wider coverage |
| `depth` | 2 | number of iterations — deeper follow-up |

```
research project proj-xxx on diffusion models breadth=6 depth=3
```

Higher breadth/depth = more papers, more LLM calls, more cost.

---

## Data Flow Summary

```
User query
    │
    ▼
LangGraph ReAct agent (main.py / ui.py)
    │  picks tools, reasons, calls them in sequence
    │
    ▼
research_project(project_id, query, breadth=4, depth=2)
    │
    ├─ get_relevant_seed()  →  kg_indexer ranks prior knowledge  →  top-15 learnings
    │
    ├─ [ITER 1]  run_iteration()
    │    ├─ _generate_queries()     LLM → 4 search queries
    │    ├─ [parallel × 2]
    │    │   ├─ search_arxiv()      live arxiv API
    │    │   ├─ indexer.add_records()  → Superlinked doc index
    │    │   └─ _extract_paper_summaries_batch()  LLM (2500 tok)
    │    │        → method, result, strength, weakness, key_facts per paper
    │    ├─ add_paper(graph, summary)  → paper node in JSON graph
    │    ├─ add_concept(graph, fact)   → concept node in JSON graph
    │    └─ kg.save()                  → crash-safe write
    │
    ├─ [ITER 2]  query = "Follow-up: " + iter1.follow_ups
    │    └─ (same as iter 1)
    │
    ├─ auto-create tickets from human_blockers (cap 5)
    ├─ write_todo()   → actionable today list
    └─ write_report() → report.md
```

---

## Outputs

| File | Description |
|---|---|
| `data/knowledge_graph.json` | All nodes (projects, papers, concepts, tickets) and edges as JSON |
| `data/knowledge_map.md` | Auto-generated human-readable view with paper method/result/weakness |
| `data/workflows/{project_id}.md` | Cumulative research blueprint — updated every session (long-term memory) |
| `report.md` | Markdown research report (Summary, Key Findings, Open Questions, References) |

---

## Design Notes

**No CSV files.** Papers come only from the live arxiv API via `arxiv_loader.py`. Every fetched paper is indexed immediately.

**Every paper becomes a node.** `add_paper()` is called for every paper the agent processes — not just papers with high relevance scores. The structured summary (method/result/strength/weakness) is stored directly in the node's `data` field.

**Crash-safe saves.** The original implementation saved to the graph only after all depth levels completed. The rewrite saves after every iteration, so partial results survive crashes.

**Batch LLM summarization.** One LLM call per iteration (not one per paper) extracts structured summaries for all papers simultaneously. Papers are labelled `[PAPER_0]..[PAPER_N]` to prevent the model from confusing which fact belongs to which paper.

**Superlinked for KG retrieval.** The KG Superlinked index is kept separate from the document index — different schema (no recency), different purpose (project-scoped concept ranking vs. general doc search). As the graph grows, seeding stays at top-15 relevant items rather than growing unboundedly.
