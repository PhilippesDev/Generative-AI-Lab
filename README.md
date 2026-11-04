# GENAI LAB — TP 2 : CONSTRUIRE UNE IA GÉNÉRATIVE PAS À PAS
**Licence 3 Génie Logiciel — ISIG GOMA (Année académique 2026-2027)**  
*Enseignant : Blaise CIZA*  
*Variante : Génie Logiciel (FastAPI, Interface Web, 20 Tests Automatiques, Journal des requêtes)*

---

## 1. Vue d'ensemble du Projet

**GENAI LAB** est une application logicielle complète et modulaire conçue pour étudier, expérimenter et comparer les grandes approches de l'Intelligence Artificielle générative et de la recherche documentaire :

- **Partie A — Mini-modèle de langage (n-grammes)** : Implémentation complète sans bibliothèque externe (pur Python) d'un modèle bigramme et trigramme sur un corpus de 320 phrases en français orientées Génie Logiciel. Intègre la formule du cours pour l'échantillonnage par température $T \in \{0.5, 1.0, 1.5\}$ et le lissage de Laplace (Add-1 smoothing) pour éliminer les probabilités nulles.
- **Partie B — Modèles pré-entraînés** : Intégration de GPT-2 (`openai-community/gpt2`) et du LLM récent local via **Ollama** (`qwen2.5-coder:3b`), avec comparaison expérimentale réelle sur 5 questions techniques.
- **Partie C — Assistant RAG (Retrieval-Augmented Generation)** : Pipeline complet comprenant la gestion documentaire (10-20 pages équivalentes), le chunking avec métadonnées et chevauchement (300-500 mots), l'indexation unique en mémoire, la recherche comparée **TF-IDF vs Embeddings neuronaux** (Precision@3), le prompt augmented strict (« Réponds uniquement à partir des extraits... ») et la citation obligatoire des sources.
- **Partie D — Évaluation sur 20 questions** : 15 questions documentées + 5 questions hors-corpus pour éprouver le refus (« Je ne sais pas ») et la résistance aux hallucinations. Tests automatisés via `pytest`.
- **Partie E — Application Web & API REST** : API FastAPI documentée avec Swagger/OpenAPI (`/docs`) et interface Web interactive responsive (HTML5 / Tailwind CSS).
- **Journalisation persistante (C10)** : Enregistrement de chaque échange (horodatage, question, méthode, extraits, scores, réponse, citations, latence, statut de refus) dans `logs/interactions.jsonl`.

---

## 2. Structure du Code Source

```text
GENAI_LAB/
│
├── app/
│   ├── main.py                     # API FastAPI & serveur web
│   ├── core/
│   │   ├── config.py               # Paramètres de configuration
│   │   └── logger.py               # Journalisation persistante (interactions.jsonl)
│   ├── services/
│   │   ├── ngram_service.py        # Algorithmes n-grammes, température et lissage
│   │   ├── gpt2_service.py         # Service GPT-2 avec top_k, max_tokens
│   │   ├── ollama_service.py       # Client HTTP standard vers Ollama local
│   │   ├── document_service.py     # Gestion du corpus documentaire
│   │   ├── chunking_service.py     # Découpage 300-500 mots avec overlap
│   │   ├── tfidf_service.py        # Vectorisation et recherche TF-IDF (scikit-learn)
│   │   ├── embedding_service.py    # Recherche neuronale (sentence-transformers)
│   │   ├── rag_service.py          # Orchestration RAG & prompt strict
│   │   └── evaluation_service.py   # Métriques Precision@3, Recall et 20 questions
│   ├── schemas/                    # Validation des types Pydantic
│   └── utils/
│       ├── create_corpus.py        # Génération du corpus 320 phrases
│       └── create_rag_data.py      # Génération des documents & benchmark
│
├── frontend/
│   └── index.html                  # Interface Web moderne tout-en-un
│
├── data/
│   ├── corpus/                     # corpus_genie_logiciel.txt (320 phrases)
│   ├── documents/                  # Documents sources (Règlement, API, DevOps, Bonnes pratiques)
│   ├── questions/                  # benchmark_20_questions.json (vérité terrain)
│   └── evaluation/                 # Résultats JSON réels de comparaison
│
├── tests/
│   ├── test_ngram.py               # Tests unitaires de la Partie A
│   └── test_rag_20_questions.py    # Tests d'intégration et benchmark des 20 questions
│
├── logs/
│   └── interactions.jsonl          # Journal persistant (JSON Lines)
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 3. Installation et Lancement

### 3.1 Prérequis
- Windows 11
- Python 3.10 ou supérieur (testé avec succès sur Python 3.14 / 3.13)
- Ollama avec le modèle `qwen2.5-coder:3b` (ou `llama3.2`)

### 3.2 Installation des dépendances
```bash
pip install -r requirements.txt
```

### 3.3 Exécution des tests automatisés
```bash
# Lancer tous les tests avec pytest
pytest tests/ -v
```

### 3.4 Lancement du serveur Web FastAPI
```bash
# Depuis le dossier GENAI_LAB :
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
- **Interface Web Utilisateur** : [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Documentation Swagger / OpenAPI** : [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 4. Résultats Expérimentaux Clés

1. **Partie A (n-grammes)** :
   - Formule exacte de température du cours : $P'(mot) = \frac{P(mot)^{1/T}}{\sum P(w)^{1/T}}$.
   - $T=0.5$ : favorise les transitions majoritaires (réponses stables et prévisibles).
   - $T=1.5$ : aplatit les probabilités (diversité accrue mais risque de déviation syntaxique).
   - Lissage de Laplace : transforme toute probabilité nulle issue d'une transition inconnue en une valeur strictement positive $\frac{C(a,b)+1}{C(a)+|V|}$.

2. **Partie B (Comparaison 3 modèles)** :
   - n-gramme : cohérence locale mais amnésie globale au-delà de 2 mots.
   - GPT-2 (2019) : modèle en langue anglaise, produit des hallucinations sur le vocabulaire technique français.
   - Ollama (`qwen2.5-coder:3b`) : compréhension contextuelle et précision technique en français pour le Génie Logiciel.

3. **Partie C & D (RAG & 20 questions)** :
   - TF-IDF surpasse les embeddings sur les acronymes stricts et numéros d'articles précis.
   - Cadrage strict du prompt : 100 % de réussite sur le refus « Je ne sais pas » pour les 5 questions hors-corpus.
   - Journal persistant horodaté avec traçabilité intégrale dans `logs/interactions.jsonl`.
