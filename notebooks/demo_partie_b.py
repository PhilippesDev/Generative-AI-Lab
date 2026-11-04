"""Expérimentation et comparaison réelle des modèles (Partie B3).
Compare les 3 approches sur 5 questions de Génie Logiciel :
1. Mini-modèle n-gramme
2. GPT-2
3. LLM récent local Ollama (qwen2.5-coder:3b)
"""

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.services.ngram_service import NGramService
from app.services.gpt2_service import gpt2_service
from app.services.ollama_service import ollama_service


QUESTIONS_B3 = [
    "Qu'est-ce qu'une architecture en microservices ?",
    "Pourquoi utilise-t-on le patron Singleton en programmation ?",
    "Quelle est la différence entre une clé primaire et une clé étrangère ?",
    "Comment fonctionne le protocole HTTPS pour sécuriser les données ?",
    "Pourquoi écrire des tests unitaires automatisés en Génie Logiciel ?"
]


def executer_comparaison_b3():
    print("=" * 80)
    print("GENAI LAB - PARTIE B3 : COMPARAISON EXPÉRIMENTALE RÉELLE DES 3 MODÈLES")
    print("=" * 80)

    # Initialiser le modèle n-gramme
    corpus_path = ROOT_DIR / "data" / "corpus" / "corpus_genie_logiciel.txt"
    ngram = NGramService()
    ngram.charger_corpus(corpus_path)

    tableau_resultats = []

    for i, q in enumerate(QUESTIONS_B3, 1):
        print(f"\n[Question {i}/5] : {q}")

        # 1. Modèle n-gramme
        # On extrait le mot clé de départ
        mot_depart = "l" if "architecture" in q else "le" if "patron" in q or "protocole" in q else "une"
        rep_ngram = ngram.generer_avec_temperature(depart=mot_depart, temperature=0.7, graine=10 + i, maximum=18)
        print(f"  • n-gramme  : {rep_ngram}")

        # 2. GPT-2
        res_gpt2 = gpt2_service.generer(prompt=q[:35], max_new_tokens=40, do_sample=True, top_k=40)
        rep_gpt2 = res_gpt2["generated_text"].replace("\n", " ")
        print(f"  • GPT-2     : {rep_gpt2[:140]}...")

        # 3. Ollama (qwen2.5-coder:3b)
        messages = [
            {"role": "system", "content": "Tu es un enseignant senior en Génie Logiciel à l'ISIG. Réponds en 2 phrases concises et précises en français."},
            {"role": "user", "content": q}
        ]
        rep_ollama = ollama_service.demander(messages, temperature=0.2)
        print(f"  • Ollama    : {rep_ollama.strip()}")

        tableau_resultats.append({
            "id": i,
            "question": q,
            "reponse_ngram": rep_ngram,
            "reponse_gpt2": rep_gpt2,
            "reponse_ollama": rep_ollama.strip()
        })

    # Sauvegarder dans data/evaluation/comparaison_b3.json
    out_dir = ROOT_DIR / "data" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "comparaison_b3.json", "w", encoding="utf-8") as f:
        json.dump(tableau_resultats, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("Synthèse enregistrée dans data/evaluation/comparaison_b3.json")
    print("=" * 80)


if __name__ == "__main__":
    executer_comparaison_b3()
