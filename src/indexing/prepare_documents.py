from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import os
from langchain_core.documents import Document

from src.indexing.chunking import build_text_splitter, chunk_event_document


INDEX_READY_PATH = Path("data/processed/events_index_ready.jsonl")


def load_index_ready(path: Path = INDEX_READY_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_json(path, lines=True)


def _clean_nan(x):
    return None if pd.isna(x) else x


def row_to_metadata(row: pd.Series) -> Dict:
    agenda_slug = os.getenv("OPENAGENDA_AGENDA_UID")  # ex: universite-paris-saclay
    agenda_url = f"https://openagenda.com/fr/{agenda_slug}"

    return {
        "uid": str(_clean_nan(row.get("uid"))),
        "origin_url": _clean_nan(row.get("origin_url")),
        "agenda_slug": agenda_slug,
        "agenda_url": agenda_url,
        "first_begin_dt": _clean_nan(row.get("first_begin_dt")),
        "first_end_dt": _clean_nan(row.get("first_end_dt")),
        "location_name": _clean_nan(row.get("location_name")),
        "location_address": _clean_nan(row.get("location_address")),
        "location_city": _clean_nan(row.get("location_city")),
        "location_postal": _clean_nan(row.get("location_postal")),
        "location_lat": _clean_nan(row.get("location_lat")),
        "location_lon": _clean_nan(row.get("location_lon")),
        "type_devenement": _clean_nan(row.get("type_devenement")),
    }


def build_documents(df: pd.DataFrame, chunk_size: int = 800, chunk_overlap: int = 120) -> List[Document]:
    splitter = build_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    docs: List[Document] = []
    for _, row in df.iterrows():
        doc_text = row.get("document", "") or ""
        md = row_to_metadata(row)

        for chunk_text, chunk_md in chunk_event_document(doc_text, md, splitter):
            docs.append(Document(page_content=chunk_text, metadata=chunk_md))

    return docs


def main():
    df = load_index_ready()
    docs = build_documents(df)
    print(f"Events: {len(df)}")
    print(f"Chunks/Documents: {len(docs)}")
    if docs:
        print("Sample doc metadata:", docs[0].metadata)
        print("Sample doc text:", docs[0].page_content[:200])


if __name__ == "__main__":
    main()
