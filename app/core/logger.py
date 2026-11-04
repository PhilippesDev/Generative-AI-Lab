"""Module de journalisation persistante des interactions RAG (Partie C10).
Stocke les requêtes, méthodes, scores, citations et temps de réponse
dans data/logs/interactions.jsonl.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class InteractionLogger:
    def __init__(self, log_path: Optional[Path] = None):
        if log_path is None:
            self.log_path = Path(__file__).resolve().parent.parent.parent / "logs" / "interactions.jsonl"
        else:
            self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def enregistrer(
        self,
        question: str,
        methode: str,
        chunks_recuperes: List[Dict[str, Any]],
        reponse: str,
        sources_citees: List[str],
        duree_ms: float,
        a_repondu_je_ne_sais_pas: bool
    ) -> Dict[str, Any]:
        """Écrit une ligne JSON dans le fichier de journalisation."""
        entree = {
            "timestamp": datetime.now().isoformat(),
            "date_heure": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": question,
            "methode": methode,
            "chunks_ids": [c.get("chunk_id", "") for c in chunks_recuperes],
            "scores": [round(float(c.get("score", 0.0)), 4) for c in chunks_recuperes],
            "sources_titres": [c.get("titre", "") for c in chunks_recuperes],
            "sources_citees": sources_citees,
            "reponse": reponse,
            "duree_ms": round(duree_ms, 2),
            "je_ne_sais_pas": a_repondu_je_ne_sais_pas
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entree, ensure_ascii=False) + "\n")

        return entree

    def lire_derniers(self, limite: int = 50) -> List[Dict[str, Any]]:
        """Lit les n dernières entrées du journal."""
        if not self.log_path.exists():
            return []
        lignes = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        lignes.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        return list(reversed(lignes[-limite:]))


interaction_logger = InteractionLogger()
