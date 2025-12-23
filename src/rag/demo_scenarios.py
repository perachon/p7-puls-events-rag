from __future__ import annotations

from src.rag.chain import answer_question


SCENARIOS = [
    "Je cherche une conférence scientifique à venir autour de Gif-sur-Yvette.",
    "Je veux une exposition ou une visite sur le campus (Orsay / Gif-sur-Yvette).",
    "Quels événements pour les étudiants / associations sont prévus prochainement ?",
    "Plutôt à Orsay : qu’est-ce que tu me recommandes ?",
    "Qu’est-ce qu’il y a ce mois-ci ?",
    "Je cherche un concert de métal à Gif-sur-Yvette ce soir.",
    "Je suis à Lyon, que me recommandes-tu ?",
    "Je veux sortir.",
]


def main():
    for q in SCENARIOS:
        print("\n" + "=" * 100)
        print("QUESTION:", q)
        res = answer_question(q)
        print("\nANSWER:\n", res.answer)
        print("\nSOURCES COUNT:", len(res.sources))


if __name__ == "__main__":
    main()
