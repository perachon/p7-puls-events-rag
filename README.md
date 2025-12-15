# Puls-Events — POC RAG (OpenAgenda)

## Objectif
Développer un POC de chatbot RAG capable de répondre à des questions sur des événements culturels récents (< 1 an),
en combinant :
- recherche vectorielle (FAISS)
- orchestrateur (LangChain)
- génération de réponse (Mistral)
- exposition via API (FastAPI)

## Structure du projet
- `src/` : ingestion, nettoyage, indexation, chaîne RAG
- `api/` : API FastAPI exposant l'assistant
- `tests/` : tests unitaires (imports, indexation, etc.)
- `data/` : données locales (raw/processed) + index vectoriel (index/)
- `docs/` : rapport + assets
- `notebooks/` : exploration (optionnel)

## Prérequis
- Python >= 3.8
- (Recommandé) `make` / terminal Bash ou PowerShell

## Installation (reproductible)
### 1) Cloner le dépôt
```bash
git clone git@github.com:perachon/p7-puls-events-rag.git
cd puls-events-rag
```

### 2) Créer un environnement virtuel
```bash
python -m venv .venv
```
Activer l'environnement :
macOS/Linux :
```bash
source .venv/bin/activate
```
Windows PowerShell :
```bash
.\.venv\Scripts\Activate.ps1
```

### 3) Installer les dépendances
```bash
pip install -U pip
pip install -r requirements.txt
```

## Configuration des variables d’environnement
Créer un fichier .env à la racine (ne jamais le commit) :
```bash
MISTRAL_API_KEY=your_key_here
```
Un exemple est fourni : .env.example

## Vérifier l’installation (imports)
Lancer les tests :
```bash
pytest -q
```

## Lancer l’API
```bash
uvicorn api.main:app --reload
```
Puis tester :
- Healthcheck : GET http://127.0.0.1:8000/health
- Docs swagger : http://127.0.0.1:8000/docs

## Notes

- Le dossier data/ contient les exports OpenAgenda, les données nettoyées, et l’index FAISS : il n’est pas versionné.
- Les clés API ne doivent jamais être commit : utiliser .env.