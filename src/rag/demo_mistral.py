from __future__ import annotations

from src.rag.llm import get_llm


def main():
    llm = get_llm()
    msg = "Réponds en une phrase : quel est le but d'un système RAG ?"
    res = llm.invoke(msg)
    print(res.content)


if __name__ == "__main__":
    main()
