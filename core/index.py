import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import superlinked.framework as sl
from docx import Document as WordDocument
from langchain_community.document_loaders import PyPDFLoader
from tqdm import tqdm

from config import DOCS_DIR


class DocSchema(sl.Schema):
    text: sl.String
    source: sl.String
    created: sl.Timestamp
    id: sl.IdField


doc = DocSchema()

text_space = sl.TextSimilaritySpace(
    text=sl.chunk(doc.text, chunk_size=200, chunk_overlap=50),
    model="sentence-transformers/all-mpnet-base-v2",
)

recency_space = sl.RecencySpace(
    timestamp=doc.created,
    period_time_list=[
        sl.PeriodTime(timedelta(days=365)),
        sl.PeriodTime(timedelta(days=2 * 365)),
        sl.PeriodTime(timedelta(days=3 * 365)),
    ],
    negative_filter=-0.25,
)

doc_index = sl.Index([text_space, recency_space])

doc_query = (
    sl.Query(
        doc_index,
        weights={
            text_space: sl.Param("relevance_weight"),
            recency_space: sl.Param("recency_weight"),
        },
    )
    .find(doc)
    .similar(text_space, sl.Param("search_query"))
    .select(doc.id, doc.source, doc.text, doc.created)
    .limit(sl.Param("limit"))
)


def _load_documents(folder_path: str) -> list[dict]:
    records, doc_id = [], 0
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            path = Path(dirpath) / filename
            created = datetime.fromtimestamp(path.stat().st_mtime)
            try:
                if path.suffix == ".pdf":
                    for page in PyPDFLoader(str(path)).load():
                        records.append({"id": str(doc_id), "text": page.page_content, "source": path.name, "created": created})
                        doc_id += 1
                elif path.suffix == ".docx":
                    word = WordDocument(str(path))
                    text = "\n".join(p.text for p in word.paragraphs if p.text.strip())
                    records.append({"id": str(doc_id), "text": text, "source": path.name, "created": created})
                    doc_id += 1
                elif path.suffix == ".xlsx":
                    for sheet_name, df in pd.read_excel(str(path), sheet_name=None).items():
                        records.append({"id": str(doc_id), "text": df.astype(str).to_string(index=False), "source": f"{path.name} - {sheet_name}", "created": created})
                        doc_id += 1
                elif path.suffix == ".md":
                    records.append({"id": str(doc_id), "text": path.read_text(encoding="utf-8"), "source": path.name, "created": created})
                    doc_id += 1
            except Exception as e:
                logging.error(f"File load error: {path.name} - {e}")
    return records


_parser = sl.DataFrameParser(
    doc,
    mapping={doc.id: "id", doc.text: "text", doc.source: "source", doc.created: "created"},
)
_source = sl.InMemorySource(doc, parser=_parser)
_executor = sl.InMemoryExecutor(sources=[_source], indices=[doc_index])
app = _executor.run()

_records = _load_documents(DOCS_DIR)
if _records:
    for batch in tqdm([_records[i:i + 10] for i in range(0, len(_records), 10)], desc="Indexing"):
        _source.put([pd.DataFrame(batch)])
    logging.info(f"Indexed {len(_records)} document chunks from {DOCS_DIR}")
else:
    logging.warning(f"No documents found in {DOCS_DIR}")


def add_records(records: list[dict]) -> int:
    if not records:
        return 0
    for batch in [records[i:i + 10] for i in range(0, len(records), 10)]:
        _source.put([pd.DataFrame(batch)])
    logging.info(f"Added {len(records)} records to index.")
    return len(records)


def search(query: str, limit: int = 5, recency_weight: float = 0.5) -> pd.DataFrame:
    results = app.query(
        doc_query,
        search_query=query,
        relevance_weight=1.0,
        recency_weight=recency_weight,
        limit=limit,
    )
    results["text"] = results["text"].astype(str)
    return results
