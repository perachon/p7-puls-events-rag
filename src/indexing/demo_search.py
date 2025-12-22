from __future__ import annotations
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

INDEX_DIR = Path("data/index/faiss_events")

def main():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)

    query = "conférence sur la pollinisation des abeilles"
    results = vs.similarity_search(query, k=5)

    for i, doc in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print("uid:", doc.metadata.get("uid"))
        print("date:", doc.metadata.get("first_begin_dt"))
        print("city:", doc.metadata.get("location_city"))
        print(doc.page_content[:250])

if __name__ == "__main__":
    main()
