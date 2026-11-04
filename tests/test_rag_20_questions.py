"""Tests automatisés avec pytest sur le benchmark des 20 questions (Parties C et D).
Exécutable directement par l'enseignant ou le pipeline CI/CD.
"""

import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.services.rag_service import rag_service
from app.services.evaluation_service import evaluation_service


@pytest.fixture(scope="session", autouse=True)
def setup_rag():
    """Initialise et indexe les documents une seule fois pour la session de tests."""
    rag_service.initialiser_ou_reindexer()


def test_indexation_initiale():
    """Vérifie que les documents et chunks sont bien indexés."""
    assert rag_service.est_pret is True
    assert len(rag_service.chunks_en_memoire) > 0


def test_precision_comparative_tfidf_embeddings():
    """Vérifie que la comparaison TF-IDF vs Embeddings calcule bien Precision@3 et HitRate@3."""
    comp = evaluation_service.evaluer_recherche_comparative()
    assert "tfidf" in comp
    assert "embeddings" in comp
    assert comp["tfidf"]["precision_moyenne_at_3"] > 0.0
    assert comp["embeddings"]["precision_moyenne_at_3"] > 0.0
    assert comp["tfidf"]["hit_rate_at_3"] >= 0.7


def test_recherche_extraits_pertinents():
    """Vérifie qu'une question sur le groupe de TP renvoie l'extrait du règlement."""
    question = "Combien d'étudiants par groupe de TP ?"
    extraits = rag_service.rechercher_extraits(question, methode="tfidf", top_k=3)
    assert len(extraits) == 3
    # L'un des extraits doit être le règlement de TP
    fichiers = [e.get("filename") for e in extraits]
    assert "reglement_tp_isig.txt" in fichiers


def test_benchmark_20_questions_structure():
    """Vérifie que le benchmark comprend bien exactement 20 questions (15 présentes, 5 absentes)."""
    questions = evaluation_service.charger_questions()
    assert len(questions) == 20
    presentes = [q for q in questions if q.get("est_presente")]
    absentes = [q for q in questions if not q.get("est_presente")]
    assert len(presentes) == 15
    assert len(absentes) == 5


def test_prompt_strict_refus_pour_question_absente():
    """Vérifie qu'une question absente génère un refus clair « Je ne sais pas » sans inventer."""
    question_absente = "Quel est le montant exact des frais d'inscription académique à l'ISIG-GOMA pour 2026 ?"
    res = rag_service.repondre(question_absente, methode="tfidf", top_k=3, temperature=0.0)
    assert res["je_ne_sais_pas"] is True
