import os
from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()  # charge .env si présent

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")


app = FastAPI(title="Puls-Events RAG API", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query(payload: dict):
    # TODO: brancher ici le pipeline RAG à l'étape 3/4
    question = payload.get("question", "")
    return {
        "question": question,
        "answer": "TODO: RAG not wired yet",
        "sources": [],
    }
