"""Superlinked semantic index for knowledge graph nodes (concepts + papers).

Keeps a persistent in-memory index of concept texts and paper summary texts,
enabling top-K relevance-ranked seeding for deep_research iterations.
"""

import logging

import pandas as pd
import superlinked.framework as sl


class KgSchema(sl.Schema):
    id: sl.IdField
    text: sl.String
    project_id: sl.String
    node_type: sl.String  # concept | paper


kg_schema = KgSchema()

text_space = sl.TextSimilaritySpace(
    text=kg_schema.text,
    model="sentence-transformers/all-mpnet-base-v2",
)

kg_index = sl.Index([text_space])

kg_query = (
    sl.Query(
        kg_index,
        weights={text_space: sl.Param("weight")},
    )
    .find(kg_schema)
    .similar(text_space, sl.Param("search_query"))
    .select(kg_schema.id, kg_schema.text, kg_schema.project_id, kg_schema.node_type)
    .limit(sl.Param("limit"))
)

_parser = sl.DataFrameParser(
    kg_schema,
    mapping={
        kg_schema.id: "id",
        kg_schema.text: "text",
        kg_schema.project_id: "project_id",
        kg_schema.node_type: "node_type",
    },
)
_source = sl.InMemorySource(kg_schema, parser=_parser)
_executor = sl.InMemoryExecutor(sources=[_source], indices=[kg_index])
app = _executor.run()


def upsert(records: list[dict]) -> None:
    """Add or update records in the KG index.

    Each record must have: {id, text, project_id, node_type}.
    """
    if not records:
        return
    for batch in [records[i:i + 20] for i in range(0, len(records), 20)]:
        _source.put([pd.DataFrame(batch)])
    logging.info(f"kg_indexer: upserted {len(records)} records.")


def search_relevant(query: str, project_id: str, limit: int = 15) -> list[str]:
    """Return top-`limit` text strings from this project most relevant to query."""
    # Fetch more than needed so post-filtering still yields `limit` results
    results = app.query(
        kg_query,
        search_query=query,
        weight=1.0,
        limit=limit * 4,
    )
    if results.empty:
        return []
    # Filter to this project only
    filtered = results[results["project_id"] == project_id]
    return filtered["text"].head(limit).tolist()
