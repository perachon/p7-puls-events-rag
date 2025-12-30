from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional, List, Set

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.rag.chain import answer_question, DEFAULT_ALLOWED_CITIES, RAGResult
from src.rag.index import rebuild_vectorstore, RebuildResult


app = FastAPI(
    title="RAG API",
    description="API REST locale pour interroger un système RAG (RAG + base vectorielle FAISS).",
    version="1.0.0",
)


# ---------
# Schemas
# ---------
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    allowed_cities: Optional[List[str]] = None
    future_only: bool = Field(
        default=True,
        description="Si true, ne renvoie que les événements à venir (par rapport à maintenant).",
    )


class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)


class RebuildResponse(BaseModel):
    status: str
    message: str
    duration_s: float
    details: Dict[str, Any] = Field(default_factory=dict)


# ----------------
# Utils (helpers)
# ----------------
def _dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    raise TypeError("Object is not a dataclass or dict")


# ---------
# Routes
# ---------
@app.get("/")
def health() -> Dict[str, str]:
    return {"status": "ok", "message": "RAG API is running"}


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> Dict[str, Any]:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question vide")

    # Validation allowed_cities si fourni
    allowed_cities: Optional[Set[str]] = None
    if payload.allowed_cities is not None:
        # on clean et on garde les non-vides
        cities = {c.strip() for c in payload.allowed_cities if c and c.strip()}
        if not cities:
            raise HTTPException(status_code=400, detail="allowed_cities fourni mais vide après nettoyage")

        # Optionnel mais utile: vérifier que les villes sont dans la whitelist
        unknown = sorted(cities - DEFAULT_ALLOWED_CITIES)
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"Villes non autorisées: {unknown}. Autorisées: {sorted(DEFAULT_ALLOWED_CITIES)}",
            )
        allowed_cities = cities

    try:
        res: RAGResult = answer_question(question, allowed_cities=allowed_cities, future_only=payload.future_only)
        # RAGResult est un dataclass -> asdict
        data = _dataclass_to_dict(res)
        return {"answer": data.get("answer", ""), "sources": data.get("sources", [])}
    except HTTPException:
        raise
    except Exception:
        # éviter d'exposer des infos sensibles (clé API etc.)
        raise HTTPException(status_code=500, detail="Erreur interne lors de la génération")


@app.post("/rebuild", response_model=RebuildResponse)
def rebuild() -> Dict[str, Any]:
    try:
        res: RebuildResult = rebuild_vectorstore()
        data = _dataclass_to_dict(res)
        return data
    except Exception:
        raise HTTPException(status_code=500, detail="Erreur interne lors du rebuild vectorstore")
