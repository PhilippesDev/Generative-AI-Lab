"""Service RAG complet (Partie C6).
Conforme au script c2_rag_complet.py de référence (M. Blaise CIZA).
Orchestre la recherche (TF-IDF ou Embeddings), la construction du prompt augmenté,
l'interrogation d'Ollama (qwen2.5-coder:3b) et la traçabilité des sources.
"""

import re
import time
from typing import Any, Dict, List, Optional
from app.core.logger import interaction_logger
from app.services.document_service import document_service
from app.services.embedding_service import embedding_service
from app.services.ollama_service import ollama_service
from app.services.tfidf_service import tfidf_service


class RAGService:
    def __init__(self):
        self.est_pret = False
        self.chunks_en_memoire = []

    def initialiser_ou_reindexer(self) -> Dict[str, Any]:
        """Indexe les documents une seule fois au démarrage ou sur modification (Consigne C9)."""
        self.chunks_en_memoire = document_service.charger_tous_les_chunks()
        nb_tfidf = tfidf_service.indexer(self.chunks_en_memoire)
        nb_emb = embedding_service.indexer(self.chunks_en_memoire)
        self.est_pret = (nb_tfidf > 0)
        return {
            "nb_documents": len(document_service.lister_documents()),
            "nb_chunks": len(self.chunks_en_memoire),
            "tfidf_indexe": nb_tfidf,
            "embeddings_indexe": nb_emb,
            "statut": "prêt" if self.est_pret else "vide"
        }

    def rechercher_extraits(self, question: str, methode: str = "tfidf", top_k: int = 3) -> List[Dict[str, Any]]:
        """Récupère les k extraits les plus pertinents selon la méthode choisie."""
        if not self.est_pret:
            self.initialiser_ou_reindexer()

        if methode.lower() == "embeddings":
            return embedding_service.rechercher(question, top_k=top_k)
        return tfidf_service.rechercher(question, top_k=top_k)

    def repondre(
        self,
        question: str,
        methode: str = "tfidf",
        top_k: int = 3,
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """Pipeline RAG complet : Récupérer -> Augmenter -> Générer -> Sourcer."""
        t0 = time.time()
        extraits = self.rechercher_extraits(question, methode=methode, top_k=top_k)

        # 1. Construction du contexte avec séparateurs clairs
        lignes_contexte = []
        for e in extraits:
            titre = e.get("titre", e.get("filename", "Extrait"))
            texte = e.get("contenu", "")
            lignes_contexte.append(f"[{titre}]\n{texte}")

        contexte_str = "\n\n".join(lignes_contexte)

        # 2. Prompt augmenté strict (Consigne du cours ISIG)
        system_prompt = (
            "Tu es un assistant documentaire rigoureux pour le cours d'IA et de Génie Logiciel à l'ISIG-GOMA.\n"
            "Consignes impératives :\n"
            "1. Réponds uniquement à partir des extraits fournis ci-dessous.\n"
            "2. Cite obligatoirement le titre de l'extrait utilisé entre crochets, par exemple : [Règlement officiel...].\n"
            "3. Si l'information nécessaire est absente des extraits ou incertaine, réponds exactement et clairement : « Je ne sais pas. »\n"
            "4. Ne complète jamais les informations avec tes connaissances générales et n'invente aucune citation."
        )

        user_prompt = f"Extraits documentaires :\n{contexte_str}\n\nQuestion de l'utilisateur : {question}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # 3. Appel du LLM (Ollama qwen2.5-coder:3b)
        reponse_brute = ollama_service.demander(messages, temperature=temperature)

        # 4. Détection du refus "Je ne sais pas"
        est_je_ne_sais_pas = (
            "je ne sais pas" in reponse_brute.lower()
            or "information est absente" in reponse_brute.lower()
            or "absente des documents" in reponse_brute.lower()
        )

        # 5. Extraction des sources citées
        sources_trouvees = re.findall(r"\[(.*?)\]", reponse_brute)
        duree_ms = (time.time() - t0) * 1000

        # 6. Journalisation persistante (C10)
        entree_log = interaction_logger.enregistrer(
            question=question,
            methode=methode,
            chunks_recuperes=extraits,
            reponse=reponse_brute,
            sources_citees=sources_trouvees,
            duree_ms=duree_ms,
            a_repondu_je_ne_sais_pas=est_je_ne_sais_pas
        )

        return {
            "question": question,
            "methode": methode,
            "reponse": reponse_brute,
            "je_ne_sais_pas": est_je_ne_sais_pas,
            "sources_citees": sources_trouvees,
            "extraits_recuperes": [
                {
                    "chunk_id": e.get("chunk_id"),
                    "titre": e.get("titre"),
                    "filename": e.get("filename"),
                    "score": round(float(e.get("score", 0.0)), 4),
                    "extrait": e.get("contenu", "")[:200] + "..."
                }
                for e in extraits
            ],
            "duree_ms": round(duree_ms, 2)
        }


rag_service = RAGService()
