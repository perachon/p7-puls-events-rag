from __future__ import annotations

from typing import Dict, List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_text_splitter(
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_text(
    text: str,
    splitter: RecursiveCharacterTextSplitter,
) -> List[str]:
    if not text:
        return []
    return splitter.split_text(text)


def chunk_event_document(
    event_doc: str,
    metadata: Dict,
    splitter: RecursiveCharacterTextSplitter,
) -> List[Tuple[str, Dict]]:
    """
    Retourne une liste de (chunk_text, chunk_metadata) pour un événement.
    Ajoute chunk_id et chunk_count aux métadonnées.
    """
    chunks = chunk_text(event_doc, splitter)
    out = []
    total = len(chunks)
    for i, ch in enumerate(chunks):
        md = dict(metadata)
        md["chunk_id"] = i
        md["chunk_count"] = total
        out.append((ch, md))
    return out
