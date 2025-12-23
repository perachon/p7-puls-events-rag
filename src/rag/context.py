from __future__ import annotations

from typing import List
from langchain_core.documents import Document


def format_docs_as_context(docs: List[Document]) -> str:
    chunks = []
    for i, d in enumerate(docs, start=1):
        md = d.metadata or {}
        chunks.append(
            f"[SOURCE {i}]\n"
            f"uid: {md.get('uid')}\n"
            f"date: {md.get('first_begin_dt')}\n"
            f"city: {md.get('location_city')}\n"
            f"location: {md.get('location_name')}\n"
            f"agenda_url: {md.get('agenda_url')}\n"
            f"text:\n{d.page_content}\n"
        )
    return "\n".join(chunks)
