# src/rag/index.py
from __future__ import annotations

import time
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class RebuildResult:
    status: str
    message: str
    duration_s: float
    details: Dict[str, Any]


def _reset_rag_cache() -> None:
    """
    on vide le cache après rebuild pour que /ask recharge le nouvel index.
    """
    try:
        import src.rag.chain as chain  # import local pour éviter les cycles
        # au lieu de chain._COMPONENTS_CACHE = None
        if hasattr(chain, "_SHARED_CACHE"):
            chain._SHARED_CACHE = None
    except Exception:
        # On ne bloque pas le rebuild si le reset échoue
        pass


def rebuild_vectorstore() -> RebuildResult:
    """
    Reconstruit la base vectorielle en appelant le module:
    python -m src.indexing.build_faiss_index
    """
    start = time.time()
    try:
        # Lance le script de build exactement comme recommandé par ton faiss_store.py
        proc = subprocess.run(
            [sys.executable, "-m", "src.indexing.build_faiss_index"],
            capture_output=True,
            text=True,
            check=True,
        )

        _reset_rag_cache()

        duration = time.time() - start
        return RebuildResult(
            status="ok",
            message="Vectorstore rebuilt successfully",
            duration_s=duration,
            details={
                "stdout": (proc.stdout or "")[-4000:],  # on limite la taille
                "stderr": (proc.stderr or "")[-4000:],
            },
        )

    except subprocess.CalledProcessError as e:
        duration = time.time() - start
        return RebuildResult(
            status="error",
            message="Vectorstore rebuild failed",
            duration_s=duration,
            details={
                "returncode": e.returncode,
                "stdout": (e.stdout or "")[-4000:],
                "stderr": (e.stderr or "")[-4000:],
            },
        )
