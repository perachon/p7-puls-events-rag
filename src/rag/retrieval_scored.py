from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Set, Tuple

import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


def parse_dt_utc(value) -> Optional[datetime]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return pd.to_datetime(value, utc=True).to_pydatetime()
    except Exception:
        return None


@dataclass
class ScoredFilteredRetriever:
    vectorstore: FAISS
    allowed_cities: Optional[Set[str]] = None
    k_fetch: int = 20
    k_final: int = 5
    max_distance: float = 1.0
    future_only: bool = True

    def retrieve(self, question: str) -> List[Tuple[Document, float]]:
        now = datetime.now(timezone.utc)

        def run(max_dist: float) -> List[Tuple[Document, float]]:
            results = self.vectorstore.similarity_search_with_score(question, k=self.k_fetch)
            kept = []
            for doc, dist in results:
                if dist is None or dist > max_dist:
                    continue

                md = doc.metadata or {}

                if self.allowed_cities:
                    city = (md.get("location_city") or "").strip()
                    if city and city not in self.allowed_cities:
                        continue

                dt = parse_dt_utc(md.get("first_begin_dt"))
                if self.future_only:
                    if dt is None or dt < now:
                        continue

                kept.append((doc, dist))
                if len(kept) >= self.k_final:
                    break
            return kept

        kept = run(self.max_distance)
        if len(kept) < min(3, self.k_final):
            kept = run(self.max_distance + 0.2)

        return kept
