from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.indexing.prepare_documents import load_index_ready, build_documents

import shutil

INDEX_DIR = Path("data/index/faiss_events")
INDEX_DIR.parent.mkdir(parents=True, exist_ok=True)


def main() -> dict:
    # 1) Load + chunk -> Documents
    df = load_index_ready()
    docs = build_documents(df, chunk_size=800, chunk_overlap=120)
    print(f"Documents to index: {len(docs)}")

    # 2) Embeddings model (local, reproducible)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 3) Build FAISS index
    vectorstore = FAISS.from_documents(docs, embeddings)

    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)


    # 4) Save locally
    vectorstore.save_local(str(INDEX_DIR))
    print(f"FAISS index saved to: {INDEX_DIR.resolve()}")

    return {"rows": len(df), "chunks": len(docs), "index_dir": str(INDEX_DIR)}

if __name__ == "__main__":
    main()
