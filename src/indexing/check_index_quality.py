from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.indexing.faiss_store import load_vectorstore


INDEX_READY = Path("data/processed/events_index_ready.jsonl")


def main():
    # 1) compter les événements "index-ready"
    df = pd.read_json(INDEX_READY, lines=True)
    n_events = len(df)

    # 2) charger FAISS
    vs = load_vectorstore()

    # 3) estimer le nombre de vecteurs dans l’index FAISS
    # langchain_community FAISS expose généralement l’index via vs.index
    n_vectors = vs.index.ntotal

    print(f"Events (index-ready): {n_events}")
    print(f"Vectors in FAISS index: {n_vectors}")

    # Sanity: n_vectors >= n_events (car chunking peut créer >=1 chunk)
    if n_vectors < n_events:
        raise RuntimeError("Index seems incomplete: fewer vectors than events.")

    # 4) test métadonnées sur une requête
    docs = vs.similarity_search("conférence", k=3)
    for i, d in enumerate(docs, start=1):
        md = d.metadata
        print(f"\n--- Sample {i} ---")
        print("uid:", md.get("uid"))
        print("date:", md.get("first_begin_dt"))
        print("city:", md.get("location_city"))
        print("agenda_url:", md.get("agenda_url"))
        print(d.page_content[:160].replace("\n", " "))


if __name__ == "__main__":
    main()
