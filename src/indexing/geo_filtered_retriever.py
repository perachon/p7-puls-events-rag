from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Set

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever


@dataclass
class GeoFilteredRetriever:
    """Wrapper retriever: retrieves more docs, then filters by metadata (city, etc.)."""
    base_retriever: VectorStoreRetriever
    allowed_cities: Optional[Set[str]] = None
    k_final: int = 5

    def invoke(self, query: str) -> List[Document]:
        docs = self.base_retriever.invoke(query)

        if not self.allowed_cities:
            return docs[: self.k_final]

        kept: List[Document] = []
        for d in docs:
            city = (d.metadata.get("location_city") or "").strip()
            if city in self.allowed_cities:
                kept.append(d)
            if len(kept) >= self.k_final:
                break

        # fallback: si on n'a pas assez de docs après filtrage, on complète avec les meilleurs non filtrés
        if len(kept) < self.k_final:
            for d in docs:
                if d not in kept:
                    kept.append(d)
                if len(kept) >= self.k_final:
                    break

        return kept
