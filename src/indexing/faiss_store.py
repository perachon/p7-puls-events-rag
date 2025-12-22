from __future__ import annotations

from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever


DEFAULT_INDEX_DIR = Path("data/index/faiss_events")
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings(model_name: str = DEFAULT_EMBED_MODEL) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model_name)


def load_vectorstore(
    index_dir: Path = DEFAULT_INDEX_DIR,
    model_name: str = DEFAULT_EMBED_MODEL,
) -> FAISS:
    if not index_dir.exists():
        raise FileNotFoundError(
            f"FAISS index directory not found: {index_dir}. "
            "Build it first with: python -m src.indexing.build_faiss_index"
        )

    embeddings = get_embeddings(model_name=model_name)
    # requis par LangChain/FAISS pour recharger le docstore local
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def get_retriever(
    vectorstore: FAISS,
    k: int = 5,
    score_threshold: Optional[float] = None,
) -> VectorStoreRetriever:
    search_kwargs = {"k": k}
    if score_threshold is not None:
        # optionnel: uniquement garder les résultats au-dessus d'un seuil
        search_kwargs["score_threshold"] = score_threshold

    return vectorstore.as_retriever(search_kwargs=search_kwargs)
