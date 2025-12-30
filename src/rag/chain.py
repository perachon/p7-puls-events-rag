from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src.indexing.faiss_store import load_vectorstore
from src.rag.context import format_docs_as_context
from src.rag.llm import get_llm
from src.rag.prompt import SYSTEM_PROMPT, HUMAN_PROMPT
from src.rag.retrieval_scored import ScoredFilteredRetriever

_COMPONENTS_CACHE = None
_SHARED_CACHE = None  # (vs, prompt, llm)

DEFAULT_ALLOWED_CITIES = {
    "Gif-sur-Yvette",
    "Orsay",
    "Évry",
    "Sceaux",
    "Paris",
    "Le Plessis-Robinson",
    "Bures-sur-Yvette",
}


@dataclass
class RAGResult:
    answer: str
    sources: List[Dict[str, Any]]


def docs_to_sources(docs: List[Document], max_excerpt_chars: int = 220) -> List[Dict[str, Any]]:
    out = []
    for d in docs:
        md = dict(d.metadata or {})
        excerpt = (d.page_content or "")[:max_excerpt_chars]
        out.append({"metadata": md, "excerpt": excerpt})
    return out


def get_shared_components():
    global _SHARED_CACHE
    if _SHARED_CACHE is None:
        vs = load_vectorstore()

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT.strip()),
                ("human", HUMAN_PROMPT.strip()),
            ]
        )

        llm = get_llm()
        _SHARED_CACHE = (vs, prompt, llm)
    return _SHARED_CACHE


def build_components(
    allowed_cities: Optional[Set[str]] = None,
    k_fetch: int = 30,
    k_final: int = 5,
    max_distance: float = 1.3,
    future_only: bool = True,
):
    vs, prompt, llm = get_shared_components()

    retriever = ScoredFilteredRetriever(
        vectorstore=vs,
        allowed_cities=allowed_cities or DEFAULT_ALLOWED_CITIES,
        k_fetch=k_fetch,
        k_final=k_final,
        max_distance=max_distance,
        future_only=future_only,
    )
    return retriever, prompt, llm



def format_sources_block(sources: List[Dict[str, Any]]) -> str:
    if not sources:
        return "\n\nSources :\n- Aucune source pertinente."
    uids = []
    for s in sources:
        uid = (s.get("metadata") or {}).get("uid")
        if uid:
            uids.append(str(uid))
    uids = sorted(set(uids))
    return "\n\nSources :\n- " + "\n- ".join(uids) if uids else "\n\nSources :\n- Aucune source pertinente."


def answer_question(question: str, allowed_cities: Optional[Set[str]] = None, llm_override=None, future_only: bool = True) -> RAGResult:
    retriever, prompt, llm = build_components(allowed_cities=allowed_cities, future_only=future_only)

    if llm_override is not None:
        llm = llm_override

    scored = retriever.retrieve(question)
    docs = [d for d, _ in scored]
    dists = [dist for _, dist in scored]

    # Pas de doc -> pas trouvé
    if not docs:
        return RAGResult(
            answer="Je n'ai pas trouvé d'événement correspondant à cette demande. Tu peux essayer d'élargir le thème, la période ou la zone géographique.",
            sources=[],
        )

    # Heuristique de confiance : si même le meilleur résultat est “limite”, on ne montre pas de sources
    best = min(dists) if dists else 999.0
    if best > 1.3: # au lieu de 0.95
        return RAGResult(
            answer="Je n'ai pas trouvé d'événement suffisamment pertinent...",
            sources=[],
        )

    sources = docs_to_sources(docs)
    context = format_docs_as_context(docs)
    messages = prompt.format_messages(context=context, question=question)
    res = llm.invoke(messages)
    
    answer_text = res.content.strip()

    # Heuristique: si le LLM indique "pas trouvé", on ne montre pas de sources
    not_found_markers = [
        "je n'ai pas trouvé",
        "aucun événement",
        "pas d'événement",
    ]
    
    # if any(m in answer_text.lower() for m in not_found_markers):
    #     return RAGResult(answer=answer_text + "\n\nSources :\n- Aucune source pertinente.", sources=[])

    answer = res.content.strip() + format_sources_block(sources)
    return RAGResult(answer=answer, sources=sources)
