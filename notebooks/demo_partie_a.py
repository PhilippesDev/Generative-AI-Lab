"""Démonstration expérimentale complète de la Partie A (Mini-modèle de langage).
Exécute tous les calculs réels demandés par le TP 2 :
- A1. Statistiques du corpus
- A2. Génération bigramme
- A3. Génération trigramme & analyse comparative
- A4. Expérimentation des températures T=0.5, T=1.0, T=1.5 avec graine fixe
- A5. Calculs de probabilités avant et après lissage de Laplace
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.services.ngram_service import NGramService


def executer_demo():
    print("=" * 70)
    print("GENAI LAB - PARTIE A : DÉMONSTRATION DU MINI-MODÈLE DE LANGAGE")
    print("=" * 70)

    corpus_path = ROOT_DIR / "data" / "corpus" / "corpus_genie_logiciel.txt"
    service = NGramService()
    service.charger_corpus(corpus_path)

    # 1. Statistiques A1
    stats = service.statistiques()
    print("\n[A1. STATISTIQUES DU CORPUS GÉNIE LOGICIEL]")
    print(f"• Nombre total de tokens : {stats['total_tokens']:,}")
    print(f"• Taille du vocabulaire unique |V| : {stats['vocabulaire_taille']:,} mots")
    print(f"• Bigrammes uniques observés : {stats['bigrammes_uniques']:,}")
    print(f"• Trigrammes uniques observés : {stats['trigrammes_uniques']:,}")
    print(f"• Phrases détectées : {stats['nb_phrases_detectees']}")
    print("• Mots les plus fréquents :", ", ".join(f"{mot} ({nb})" for mot, nb in stats["mots_les_plus_frequents"][:6]))

    # 2. Génération Bigramme vs Trigramme A2 & A3
    print("\n[A2 & A3. COMPARAISON BIGRAMME VS TRIGRAMME (Graine = 42)]")
    print("\n--- Bigramme (conditionné sur 1 mot) ---")
    for i in range(1, 4):
        print(f"  {i}. {service.generer_bigramme(depart='le', graine=40 + i)}")

    print("\n--- Trigramme (conditionné sur 2 mots avec repli) ---")
    for i in range(1, 4):
        print(f"  {i}. {service.generer_trigramme(graine=40 + i)}")

    # 3. Température A4
    print("\n[A4. INFLUENCE DE LA TEMPÉRATURE T = 0.5, T = 1.0, T = 1.5]")
    print("Formule : P'(mot) = P(mot)^(1/T) / somme")
    for t_val in [0.5, 1.0, 1.5]:
        print(f"\n>>> Température T = {t_val} (5 phrases reproductibles, graines 1 à 5) :")
        for k in range(1, 6):
            phrase_gen = service.generer_avec_temperature(depart="le", temperature=t_val, graine=100 + k)
            print(f"    Essai {k} : {phrase_gen}")

    # 4. Probabilité d'une phrase et Lissage A5
    print("\n[A5. PROBABILITÉ D'UNE PHRASE ET LISSAGE DE LAPLACE]")
    phrase_connue = "le protocole http régit les échanges entre le navigateur client et le serveur web ."
    phrase_inconnue = "le protocole docker régit les serveurs quantiques ."

    print(f"\n1) Phrase contenant des enchaînements vus : « {phrase_connue} »")
    p_brut_1, details_1 = service.proba_phrase_brute(phrase_connue)
    p_lap_1, log_lap_1, _ = service.proba_phrase_laplace(phrase_connue)
    print(f"   • Proba brute (sans lissage)   : {p_brut_1:.4e}")
    print(f"   • Proba Laplace (avec lissage) : {p_lap_1:.4e} (log: {log_lap_1:.2f})")

    print(f"\n2) Phrase contenant un enchaînement jamais vu : « {phrase_inconnue} »")
    p_brut_2, details_2 = service.proba_phrase_brute(phrase_inconnue)
    p_lap_2, log_lap_2, _ = service.proba_phrase_laplace(phrase_inconnue)
    print(f"   • Proba brute (sans lissage)   : {p_brut_2:.4e} (Un seul zéro annule tout le produit !)")
    print(f"   • Proba Laplace (avec lissage) : {p_lap_2:.4e} (log: {log_lap_2:.2f})")
    print("   -> Grâce au lissage Add-1, la probabilité reste strictement positive.")
    print("=" * 70)


if __name__ == "__main__":
    executer_demo()
