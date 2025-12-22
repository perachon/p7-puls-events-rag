from __future__ import annotations

from src.indexing.faiss_store import load_vectorstore, get_retriever


def main():
    vs = load_vectorstore()
    retriever = get_retriever(vs, k=5)

    query = "conférence sur les abeilles et la pollinisation"
    docs = retriever.invoke(query)

    for i, d in enumerate(docs, start=1):
        print(f"\n--- {i} ---")
        print("uid:", d.metadata.get("uid"))
        print("date:", d.metadata.get("first_begin_dt"))
        print("city:", d.metadata.get("location_city"))
        print(d.page_content[:250])


if __name__ == "__main__":
    main()
