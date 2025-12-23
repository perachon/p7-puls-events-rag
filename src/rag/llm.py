from __future__ import annotations

import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI


DEFAULT_MODEL = "mistral-small-latest"


def get_llm(model: str = DEFAULT_MODEL, temperature: float = 0.2):
    load_dotenv()
    if not os.getenv("MISTRAL_API_KEY"):
        raise RuntimeError("MISTRAL_API_KEY manquante. Ajoute-la dans ton .env (non versionné).")

    # ChatMistralAI utilise la clé via env var MISTRAL_API_KEY
    return ChatMistralAI(
        model=model,
        temperature=temperature,
    )
