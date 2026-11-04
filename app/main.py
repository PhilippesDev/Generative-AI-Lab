"""API REST FastAPI & Serveur Web Principal - GENAI LAB.
Variante Génie Logiciel : TP 2 - ISIG GOMA (2026-2027).
Conforme aux standards OpenAPI / Swagger (accessible sur /docs).
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from app.core.logger import interaction_logger
from app.services.document_service import document_service
from app.services.evaluation_service import evaluation_service
from app.services.gpt2_service import gpt2_service
from app.services.ngram_service import NGramService
from app.services.ollama_service import ollama_service
from app.services.rag_service import rag_service

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="GENAI LAB — TP 2 ISIG-GOMA",
    description="Application modulaire d'IA Générative : n-grammes, GPT-2, Ollama local et RAG avec TF-IDF / Embeddings.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialisation du service n-gramme sur le corpus
ngram_service = NGramService()
corpus_file = BASE_DIR / "data" / "corpus" / "corpus_genie_logiciel.txt"
if corpus_file.exists():
    ngram_service.charger_corpus(corpus_file)

# Indexation initiale du RAG
rag_service.initialiser_ou_reindexer()


# ---------------------------------------------------------------------------
# Schémas Pydantic
# ---------------------------------------------------------------------------
class NGramGenerateRequest(BaseModel):
    depart: str = Field(default="le", description="Mot de départ")
    temperature: float = Field(default=1.0, ge=0.0, le=2.5, description="Température d'échantillonnage")
    graine: Optional[int] = Field(default=42, description="Graine aléatoire pour reproductibilité")
    maximum: int = Field(default=20, ge=5, le=50, description="Longueur maximale")
    type_modele: str = Field(default="bigramme", description="bigramme ou trigramme")


class SentenceProbaRequest(BaseModel):
    phrase: str = Field(..., description="Phrase à évaluer")


class GPT2Request(BaseModel):
    prompt: str = Field(default="The artificial intelligence is", description="Texte de départ")
    max_new_tokens: int = Field(default=40, ge=10, le=100)
    top_k: int = Field(default=50, ge=1, le=100)
    do_sample: bool = True
    repetition_penalty: float = Field(default=1.2, ge=1.0, le=2.0)


class OllamaRequest(BaseModel):
    prompt: str = Field(..., description="Question ou consigne")
    temperature: float = Field(default=0.2, ge=0.0, le=1.5)
    role_system: Optional[str] = "Tu es un enseignant en Génie Logiciel à l'ISIG-GOMA. Réponds clairement en français."


class RAGQueryRequest(BaseModel):
    question: str = Field(..., description="Question documentaire à poser")
    methode: str = Field(default="tfidf", description="tfidf ou embeddings")
    top_k: int = Field(default=3, ge=1, le=5)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)


class DocumentAddRequest(BaseModel):
    filename: str = Field(..., description="Nom du fichier texte (ex: note_cours.txt)")
    contenu: str = Field(..., description="Contenu textuel du document")


# ---------------------------------------------------------------------------
# Endpoints de l'API
# ---------------------------------------------------------------------------
@app.get("/api/status", summary="Vérifie l'état de l'application et des modèles")
def get_status():
    ollama_statut = ollama_service.verifier_disponibilite()
    stats_corpus = ngram_service.statistiques()
    docs = document_service.lister_documents()
    return {
        "statut": "operationnel",
        "systeme": "Windows 11 / Intel Core i5-8250U",
        "ollama": ollama_statut,
        "ngram_statistiques": stats_corpus,
        "rag": {
            "nb_documents": len(docs),
            "nb_chunks": len(rag_service.chunks_en_memoire),
            "est_pret": rag_service.est_pret
        }
    }


# --- PARTIE A : N-GRAMMES ---
@app.get("/api/ngram/stats", summary="Statistiques détaillées du corpus n-grammes")
def get_ngram_stats():
    return ngram_service.statistiques()


@app.post("/api/ngram/generate", summary="Générer une phrase avec n-gramme et température")
def generate_ngram(req: NGramGenerateRequest):
    if req.type_modele == "trigramme":
        texte = ngram_service.generer_trigramme(graine=req.graine, maximum=req.maximum)
    else:
        texte = ngram_service.generer_avec_temperature(
            depart=req.depart,
            temperature=req.temperature,
            graine=req.graine,
            maximum=req.maximum
        )
    return {
        "phrase_generee": texte,
        "parametres": req.dict(),
        "type": req.type_modele
    }


@app.post("/api/ngram/proba", summary="Calcul de la probabilité d'une phrase et lissage de Laplace")
def calculate_proba(req: SentenceProbaRequest):
    p_brut, details_brut = ngram_service.proba_phrase_brute(req.phrase)
    p_laplace, log_laplace, details_laplace = ngram_service.proba_phrase_laplace(req.phrase)
    return {
        "phrase": req.phrase,
        "proba_brute": p_brut,
        "details_brut": details_brut,
        "proba_laplace": p_laplace,
        "log_proba_laplace": log_laplace,
        "details_laplace": details_laplace,
        "zero_detecte": (p_brut == 0.0)
    }


# --- PARTIE B : MODÈLES PRÉ-ENTRAÎNÉS ---
@app.post("/api/gpt2/generate", summary="Génération de texte avec GPT-2")
def generate_gpt2(req: GPT2Request):
    return gpt2_service.generer(
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        do_sample=req.do_sample,
        top_k=req.top_k,
        repetition_penalty=req.repetition_penalty
    )


@app.post("/api/ollama/chat", summary="Interroger le LLM local Ollama")
def chat_ollama(req: OllamaRequest):
    messages = [
        {"role": "system", "content": req.role_system or "Tu es un enseignant en IA."},
        {"role": "user", "content": req.prompt}
    ]
    reponse = ollama_service.demander(messages, temperature=req.temperature)
    return {
        "prompt": req.prompt,
        "reponse": reponse,
        "model": ollama_service.model
    }


# --- PARTIE C : ASSISTANT RAG ---
@app.get("/api/documents", summary="Lister les documents du corpus RAG")
def list_documents():
    return {
        "documents": document_service.lister_documents(),
        "total_chunks": len(rag_service.chunks_en_memoire)
    }


@app.post("/api/documents", summary="Ajouter un document au corpus")
def add_document(req: DocumentAddRequest):
    res = document_service.ajouter_document(req.filename, req.contenu)
    # Réindexation automatique selon la consigne C9
    rag_service.initialiser_ou_reindexer()
    return res


@app.delete("/api/documents/{filename}", summary="Supprimer un document")
def delete_document(filename: str):
    succes = document_service.supprimer_document(filename)
    if not succes:
        raise HTTPException(status_code=404, detail="Document introuvable")
    rag_service.initialiser_ou_reindexer()
    return {"statut": "supprimé", "filename": filename}


@app.post("/api/rag/reindex", summary="Déclencher la réindexation complète")
def reindex_rag():
    return rag_service.initialiser_ou_reindexer()


@app.post("/api/rag/query", summary="Poser une question à l'assistant RAG")
def query_rag(req: RAGQueryRequest):
    return rag_service.repondre(
        question=req.question,
        methode=req.methode,
        top_k=req.top_k,
        temperature=req.temperature
    )


# --- PARTIE D : ÉVALUATIONS & BENCHMARK ---
@app.get("/api/eval/comparative", summary="Comparaison TF-IDF vs Embeddings (Precision@3)")
def get_comparative_eval():
    return evaluation_service.evaluer_recherche_comparative()


@app.get("/api/eval/benchmark20", summary="Lancer l'évaluation sur le jeu des 20 questions")
def run_benchmark_20(methode: str = Query("tfidf", description="tfidf ou embeddings")):
    return evaluation_service.evaluer_rag_20_questions(methode=methode)


@app.get("/api/logs", summary="Consulter le journal des questions/réponses")
def get_logs(limite: int = Query(50, ge=1, le=200)):
    return {
        "total_lus": len(interaction_logger.lire_derniers(limite)),
        "interactions": interaction_logger.lire_derniers(limite)
    }


# ---------------------------------------------------------------------------
# Servir l'interface Web (Frontend)
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, summary="Interface Web GENAI LAB")
def serve_index():
    index_file = BASE_DIR / "frontend" / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return "<h1>GENAI LAB - Interface Web en cours de chargement...</h1>"
