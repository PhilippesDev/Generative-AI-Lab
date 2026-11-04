"""Tests unitaires et de conformité académique pour le service n-gramme (Partie A).
Vérifie rigoureusement la conformité avec les corrigés officiels :
- c2_bigramme.py (Exercices 1, 2, 3 et exemple du cours)
- c2_temperature.py (Formule d'élévation à la puissance 1/T)
"""

import math
import sys
from pathlib import Path
from fractions import Fraction

# Ajouter le chemin racine du projet pour les imports
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.services.ngram_service import NGramService


def test_cours_exemple_bigramme_exact():
    """Vérifie l'exemple du cours page 16/41 et 18/41 :

    Corpus : "le réseau de l'isig est rapide . le réseau de goma est lent .
              le serveur de l'isig est en panne . le serveur web est rapide ."
    Phrase testée : "le réseau de goma est rapide ." -> P = 1/12 ≈ 0.0833
    """
    texte = (
        "le réseau de l'isig est rapide . "
        "le réseau de goma est lent . "
        "le serveur de l'isig est en panne . "
        "le serveur web est rapide ."
    )
    service = NGramService()
    service.entrainer(texte)

    # Vérifications des fractions exactes du corrigé
    assert service.proba_bigramme_fraction("le", "réseau") == Fraction(2, 4)
    assert service.proba_bigramme_fraction("réseau", "de") == Fraction(2, 2)
    assert service.proba_bigramme_fraction("de", "goma") == Fraction(1, 3)
    assert service.proba_bigramme_fraction("goma", "est") == Fraction(1, 1)
    assert service.proba_bigramme_fraction("est", "rapide") == Fraction(2, 4)
    assert service.proba_bigramme_fraction("rapide", ".") == Fraction(2, 2)

    # Produit des probabilités
    p_produit = (
        service.proba_bigramme_fraction("le", "réseau")
        * service.proba_bigramme_fraction("réseau", "de")
        * service.proba_bigramme_fraction("de", "goma")
        * service.proba_bigramme_fraction("goma", "est")
        * service.proba_bigramme_fraction("est", "rapide")
        * service.proba_bigramme_fraction("rapide", ".")
    )
    assert p_produit == Fraction(1, 12)
    assert round(float(p_produit), 3) == 0.083


def test_temperature_formule_cours():
    """Vérifie le calcul exact de la température issu de c2_temperature.py (page 11/41).

    Mots = [lac, marché, stade, volcan, cinéma]
    P = [0.40, 0.25, 0.15, 0.12, 0.08]
    Pour T = 0.5 : lac ≈ 60.2%
    Pour T = 1.5 : lac ≈ 32.9%
    """
    p = [0.40, 0.25, 0.15, 0.12, 0.08]

    res_05 = NGramService.appliquer_temperature(p, 0.5)
    assert round(res_05[0], 3) == 0.602
    assert round(res_05[1], 3) == 0.235
    assert round(sum(res_05), 5) == 1.0

    res_15 = NGramService.appliquer_temperature(p, 1.5)
    assert round(res_15[0], 3) == 0.329
    assert round(sum(res_15), 5) == 1.0


def test_reproductibilite_graine_fixe():
    """Vérifie que deux tirages avec la même graine produisent exactement la même phrase."""
    texte = (
        "l architecture en microservices est moderne . "
        "l architecture monolithique est classique . "
        "un microservice est autonome et déployable ."
    )
    service = NGramService()
    service.entrainer(texte)

    p1 = service.generer_avec_temperature(depart="l", temperature=0.7, graine=42)
    p2 = service.generer_avec_temperature(depart="l", temperature=0.7, graine=42)
    assert p1 == p2


def test_laplace_smoothing_evite_zero():
    """Démontre que sans lissage, un mot jamais observé donne proba 0,

    tandis qu'avec Laplace, la probabilité est strictement positive.
    """
    texte = "le serveur web est rapide . le réseau est lent ."
    service = NGramService()
    service.entrainer(texte)

    # 'serveur' n'a jamais été suivi de 'lent' dans ce petit corpus
    p_brut, _ = service.proba_phrase_brute("le serveur lent .")
    assert p_brut == 0.0

    # Avec lissage de Laplace
    p_laplace, log_p, details = service.proba_phrase_laplace("le serveur lent .")
    assert p_laplace > 0.0
    assert not math.isinf(log_p)


def test_corpus_genie_logiciel_300_phrases():
    """Vérifie que le corpus complet du projet dépasse bien les 300 phrases et s'entraîne sans erreur."""
    corpus_path = ROOT_DIR / "data" / "corpus" / "corpus_genie_logiciel.txt"
    assert corpus_path.exists(), "Le fichier de corpus doit exister"

    service = NGramService()
    tokens_count = service.charger_corpus(corpus_path)

    stats = service.statistiques()
    assert stats["nb_phrases_detectees"] >= 300
    assert stats["vocabulaire_taille"] > 200
    assert tokens_count > 2500

    # Génération trigramme et bigramme
    phrase_bi = service.generer_bigramme(depart="le", graine=10)
    phrase_tri = service.generer_trigramme(graine=10)

    assert phrase_bi.endswith(".")
    assert phrase_tri.endswith(".")
