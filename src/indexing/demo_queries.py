from __future__ import annotations

from src.indexing.faiss_store import load_vectorstore, get_retriever
from src.indexing.geo_filtered_retriever import GeoFilteredRetriever

QUERIES = [
    "conférence sur l’environnement et la biodiversité",
    "atelier de programmation python pour débutants",
    "exposition ou visite guidée sur le campus",
    "événement sur la santé mentale ou le bien-être",
    "événements étudiants, associations, vie de campus",
]


def main():
    vs = load_vectorstore()
    base = vs.as_retriever(search_kwargs={"k": 20})  # on récupère plus large, puis on filtre
    retriever = GeoFilteredRetriever(
        base_retriever=base,
        allowed_cities={
            "Gif-sur-Yvette",
            "Orsay",
            "Évry",
            "Sceaux",
            "Paris",
            "Le Plessis-Robinson",
        },
        k_final=5,
    )

    for q in QUERIES:
        print("\n" + "=" * 90)
        print("QUERY:", q)
        docs = retriever.invoke(q)

        for i, d in enumerate(docs, start=1):
            md = d.metadata
            print(f"\n--- Result {i} ---")
            print("uid:", md.get("uid"))
            print("date:", md.get("first_begin_dt"))
            print("city:", md.get("location_city"))
            print("url:", md.get("origin_url"))
            print(d.page_content[:220].replace("\n", " "))


if __name__ == "__main__":
    main()
