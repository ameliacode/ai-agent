from tools.projects import list_projects, add_project, research_project, create_ticket, get_my_todos, close_ticket
from tools.research import deep_research
from tools.search import ask_docs, retrieve_docs, summarize_docs
from tools.analyze import classify_papers, list_by_field_and_time
from tools.output import export_to_markdown, generate_readme, generate_todo
from tools.fetch import fetch_arxiv_papers, fetch_arxiv_by_id, fetch_arxiv_by_category

TOOLS = [
    # Project management + knowledge graph
    list_projects,
    add_project,
    research_project,
    create_ticket,
    get_my_todos,
    close_ticket,
    # Standalone deep research
    deep_research,
    # Arxiv fetch
    fetch_arxiv_papers,
    fetch_arxiv_by_id,
    fetch_arxiv_by_category,
    # Document search & QA
    retrieve_docs,
    summarize_docs,
    ask_docs,
    # Analysis
    classify_papers,
    list_by_field_and_time,
    # Output
    export_to_markdown,
    generate_todo,
    generate_readme,
]
