from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Set

import pandas as pd
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever


RECO_HINTS = {
    "recommande", "recommander", "propose", "suggestion", "quoi faire", "sortie", "idée",
    "à venir", "ce week-end", "ce weekend", "ce soir", "demain", "prochain", "prochaine",
}
PAST_HINTS = {
    "récemment", "recent", "dernier", "dernière", "la semaine dernière", "le mois dernier",
    "passé", "avaient lieu",
}


def detect_intent(question: str) -> str:
    q = (question or "").lower()
    if any(h in q for h in PAST_HINTS):
        return "past_ok"
    if any(h in q for h in RECO_HINTS):
        return "upcoming_only"
    # par défaut, on traite comme une demande de reco (plus safe métier)
    return "upcoming_only"


def parse_dt_utc(value) -> Optional[datetime]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return pd.to_datetime(value, utc=True).to_pydatetime()
    except Exception:
        return None


@dataclass
class FilteredRetriever:
    base_retriever: VectorStoreRetriever
    allowed_cities: Optional[Set[str]] = None
    k_final: int = 5

    def invoke(self, question: str) -> List[Document]:
        intent = detect_intent(question)
        now = datetime.now(timezone.utc)

        candidates = self.base_retriever.invoke(question)  # ex: k_fetch=20
        kept: List[Document] = []

        for d in candidates:
            md = d.metadata or {}

            # --- filtre géo ---
            if self.allowed_cities:
                city = (md.get("location_city") or "").strip()
                if city and city not in self.allowed_cities:
                    continue

            # --- filtre temps ---
            dt = parse_dt_utc(md.get("first_begin_dt"))
            if intent == "upcoming_only":
                if dt is None or dt < now:
                    continue

            kept.append(d)
            if len(kept) >= self.k_final:
                break

        # fallback si trop strict
        if len(kept) < self.k_final:
            for d in candidates:
                if d not in kept:
                    kept.append(d)
                if len(kept) >= self.k_final:
                    break

        return kept
