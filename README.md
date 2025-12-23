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

### 3) Activer l'environnement :
macOS/Linux :
```bash
source .venv/bin/activate
```
Windows PowerShell :
```bash
.\.venv\Scripts\Activate.ps1
```

### 4) Installer les dépendances
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

## Pré-processing des données OpenAgenda

### Source des données
Les données utilisées proviennent de l’API OpenAgenda.
L’agenda sélectionné pour ce POC est : **universite-paris-saclay**.

Les événements sont récupérés via l’endpoint `/v2/agendas/{agendaUID}/events`,
en utilisant une pagination par curseur (`after`) afin de collecter l’intégralité du corpus.

### Volume de données
- Événements bruts récupérés : ~3800
- Événements après nettoyage et validation : ~3800
- Événements récents (< 1 an) : ~600
- Événements à venir : ~50

### Nettoyage et validation
Les règles de nettoyage suivantes ont été appliquées :
- suppression des événements sans date (`first_begin_dt` manquant)
- suppression des titres trop courts
- suppression des descriptions trop pauvres, sauf si un lieu est renseigné
- dédoublonnage des événements sur l’identifiant unique `uid`

Ces règles visent à garantir une qualité minimale du contenu textuel avant indexation.

### Filtrage temporel
Conformément aux consignes du projet, les événements conservés respectent l’une des conditions suivantes :
- événements dont la date de début est comprise dans les 365 derniers jours
- événements dont la date de début est postérieure à la date courante

Le jeu final utilisé pour l’indexation est la concaténation des deux sous-ensembles
(`past_year` et `upcoming`).

### Structuration et préparation pour l’indexation
Un schéma de données stable est défini, incluant :
- identifiants
- métadonnées temporelles
- informations de localisation
- contenu textuel

Un champ texte unique `document` est construit pour chaque événement.
Il concatène le titre, la description, la date, le lieu et les mots-clés.
Ce champ constitue l’entrée principale pour la vectorisation.

Les dates sont exportées au format ISO afin d’assurer une compatibilité
et une interprétation correcte lors des phases ultérieures.

### Tests unitaires
Des tests unitaires ont été mis en place pour vérifier :
- l’existence des fichiers générés
- la présence des champs essentiels
- le respect des bornes temporelles
- la non-vacuité du champ `document`

Ces tests garantissent la reproductibilité et la robustesse du pipeline de pré-processing.

## Base de données vectorielle (FAISS)

Cette étape consiste à transformer les événements culturels nettoyés en vecteurs sémantiques
et à les indexer dans une base vectorielle FAISS afin de permettre une recherche rapide par similarité.

### Données en entrée
- Fichier : `data/processed/events_index_ready.jsonl`
- Nombre d’événements : ~661
- Champ textuel indexé : `document` (titre, description, date, lieu, mots-clés)
- Métadonnées conservées : identifiant, dates, localisation, type d’événement, lien vers l’agenda

### Découpage des textes (chunking)
Les textes sont découpés en chunks avant vectorisation afin d’améliorer la précision de la recherche.
- Taille des chunks : 800 caractères
- Overlap : 120 caractères
- Outil : `RecursiveCharacterTextSplitter` (LangChain)

Dans le cas présent, la majorité des événements tiennent dans un seul chunk.

### Vectorisation
- Modèle d’embeddings : `sentence-transformers/all-MiniLM-L6-v2`
- Calcul des embeddings en local (pas d’API externe)
- Un embedding est généré par chunk

### Indexation FAISS
- Type d’index : FAISS (via LangChain)
- Nombre de vecteurs indexés : ~661
- L’index est persisté localement dans : `data/index/faiss_events/`

Un script de reconstruction permet de régénérer l’index à partir des données brutes en une seule commande.

### Recherche sémantique
La recherche est réalisée via un `Retriever` LangChain :
- récupération des `k` documents les plus similaires
- filtrage géographique post-retrieval pour privilégier les événements locaux (Paris-Saclay)
- accès aux métadonnées pour enrichir les réponses (date, ville, lien)

### Qualité et performance
- Vérification de la complétude de l’index (nombre de vecteurs cohérent)
- Latence moyenne de recherche : ~27 ms (local)
- Tests unitaires automatisés validant :
  - le chargement de l’index
  - la présence de vecteurs
  - la capacité à retourner des résultats pertinents

Cette base vectorielle constitue le socle du système RAG et sera utilisée dans l’étape suivante
pour la génération de réponses augmentées via un modèle de langage (Mistral).

-------------------------------
--------------------------------
----------------------------------------

## Système RAG (Retrieval-Augmented Generation)

### Objectif
Cette étape vise à implémenter un **chatbot intelligent** capable de recommander des événements culturels à partir des données OpenAgenda, en combinant :
- une **recherche sémantique** dans une base vectorielle FAISS,
- et la **génération de réponses naturelles** à l’aide d’un LLM (Mistral).

Le système repose sur une architecture **RAG (Retrieval-Augmented Generation)** afin de garantir des réponses pertinentes, traçables et limitées aux données disponibles.

### Architecture du système

1. Question utilisateur  
2. Vectorisation de la requête  
3. Recherche sémantique dans FAISS  
   - filtrage géographique (zone Paris-Saclay)  
   - filtrage temporel (événements à venir)  
   - seuil de similarité pour limiter le bruit  
4. Construction du contexte  
   - descriptions des événements pertinents  
   - métadonnées (date, lieu, ville, uid)  
5. Génération de la réponse  
   - appel au LLM Mistral via LangChain  
   - prompt contraint (anti-hallucination)  
6. Réponse finale  
   - texte généré  
   - liste des sources (UID OpenAgenda)  

### Gestion des hallucinations
Le chatbot respecte les règles suivantes :
- les réponses sont générées **uniquement à partir du contexte récupéré**
- si aucun événement pertinent n’est trouvé, le bot répond explicitement qu’il n’y a pas de résultat
- les **sources ne sont jamais générées par le LLM**, mais ajoutées côté code à partir des documents réellement récupérés
- aucun événement, date ou lieu n’est inventé

### Format de sortie
Chaque réponse contient :
- une **réponse textuelle structurée**
- une section **Sources** listant les UID OpenAgenda utilisés  
  (ou *Aucune source pertinente* si aucun événement n’a été trouvé)

### Lancer une démonstration
Depuis la racine du projet :
```bash
python -m src.rag.demo_rag
```
Pour tester plusieurs scénarios utilisateurs :
```bash
python -m src.rag.demo_scenarios
```

### Jeu de test annoté (Gold Standard)
Un jeu de test manuel a été construit pour évaluer la qualité des réponses :
- Fichier : data/eval/qa_gold.jsonl
- 10 questions représentatives :
  - recommandations thématiques
  - cas sans résultat
  - questions vagues
  - hors périmètre géographique
- Chaque question est associée à :
  - une description de réponse attendue
  - une liste d’UID OpenAgenda attendus (expected_uids)
Ce jeu constitue la vérité terrain (gold standard) utilisée pour l’évaluation automatique.

### Évaluation automatique
Un script d’évaluation compare les réponses du chatbot au jeu de test annoté.
Commande :
```bash
python -m src.eval.evaluate_rag
```
Métriques calculées :
- Exact Match (UID)
- Précision
- Rappel
- F1-score
- Verdict par question : correct, partial, incorrect

Résultats obtenus (POC)
- 50 % réponses correctes
- 50 % partiellement correctes
- 0 % réponses incorrectes

Ces résultats montrent une bonne robustesse globale du système, avec des pistes d’amélioration possibles sur le classement des résultats.