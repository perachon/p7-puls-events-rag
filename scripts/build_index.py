from __future__ import annotations

import sys
from pathlib import Path

# Ajoute la racine du projet au PYTHONPATH pour que "import src..." marche
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.indexing.build_faiss_index import main


if __name__ == "__main__":
    main()
