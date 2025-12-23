from __future__ import annotations

from src.rag.chain import answer_question


def main():
    q = "Je cherche une conférence sur l'environnement ou le climat à venir autour de Gif-sur-Yvette."
    result = answer_question(q)

    print("\n=== QUESTION ===\n")
    print(q)

    print("\n=== ANSWER ===\n")
    print(result.answer)

    print("\n=== SOURCES COUNT ===\n")
    # Si SOURCES COUNT > 0 → alors le seuil max_distance=1.1 est trop permissif et FAISS considère ces events “assez proches”.
    print(len(result.sources))

    print("\n=== SOURCES (debug) ===\n")
    if not result.sources:
        print("Aucune source pertinente.")
        return

    for i, s in enumerate(result.sources, start=1):
        md = s["metadata"]
        print(f"\n--- Source {i} ---")
        print("uid:", md.get("uid"))
        print("date:", md.get("first_begin_dt"))
        print("city:", md.get("location_city"))
        print("agenda_url:", md.get("agenda_url"))
        print("excerpt:", s["excerpt"])


if __name__ == "__main__":
    main()
