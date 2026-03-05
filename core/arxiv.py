import logging
from datetime import datetime

import arxiv

_client = arxiv.Client()


def _to_record(paper: arxiv.Result) -> dict:
    arxiv_id = paper.entry_id.split("/")[-1]
    text = (
        f"Title: {paper.title}\n\n"
        f"Authors: {', '.join(str(a) for a in paper.authors)}\n"
        f"Categories: {', '.join(paper.categories)}\n"
        f"Published: {paper.published.strftime('%Y-%m-%d')}\n\n"
        f"Abstract: {paper.summary}"
    )
    return {
        "id": arxiv_id,
        "title": paper.title,
        "text": text,
        "source": f"arxiv:{arxiv_id}",
        "created": paper.published.replace(tzinfo=None),
    }


def search(query: str, max_results: int = 20, categories: list[str] | None = None) -> list[dict]:
    s = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
    records = []
    for paper in _client.results(s):
        if categories and not any(c in paper.categories for c in categories):
            continue
        records.append(_to_record(paper))
    logging.info(f"arxiv search '{query}': {len(records)} papers fetched.")
    return records


def fetch_by_id(arxiv_id: str) -> dict | None:
    results = list(_client.results(arxiv.Search(id_list=[arxiv_id])))
    if not results:
        logging.warning(f"arxiv ID not found: {arxiv_id}")
        return None
    return _to_record(results[0])


def fetch_by_category(category: str, max_results: int = 20) -> list[dict]:
    return search(f"cat:{category}", max_results=max_results)
