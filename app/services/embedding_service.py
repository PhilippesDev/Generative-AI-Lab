"""Service de recherche documentaire par Embeddings neuronaux (Partie C4).
Utilise sentence-transformers avec un modèle compact ou un encodeur sémantique dense
aligné (fit à l'indexation, transform à la requête).
"""

import numpy as np
from typing import Any, Dict, List, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer


class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings = None
        self.est_indexe = False
        self._modele_charge = False
        # Vectoriseur dense de secours avec vocabulaire fixe
        self._fallback_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        self._fallback_fitted = False

    def charger_modele(self):
        """Charge le modèle d'embedding compact à la demande (lazy-loading)."""
        if self._modele_charge:
            return self.model
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self._modele_charge = True
            return self.model
        except Exception:
            self.model = None
            self._modele_charge = True
            return None

    def indexer(self, chunks: List[Dict[str, Any]]) -> int:
        """Indexe les chunks une seule fois en mémoire."""
        self.chunks = chunks
        if not chunks:
            self.embeddings = None
            self.est_indexe = False
            return 0

        textes = [chunk["contenu"] for chunk in chunks]
        mod = self.charger_modele()
        if mod is not None:
            self.embeddings = mod.encode(textes, normalize_embeddings=True)
        else:
            # Fit sur l'ensemble du corpus
            X = self._fallback_vec.fit_transform(textes).toarray()
            self._fallback_fitted = True
            self.embeddings = normalize(X, norm="l2")

        self.est_indexe = True
        return len(chunks)

    def rechercher(self, question: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Recherche les top_k morceaux les plus proches sémantiquement."""
        if not self.est_indexe or self.embeddings is None or not self.chunks:
            return []

        mod = self.charger_modele()
        if mod is not None:
            q_emb = mod.encode([question], normalize_embeddings=True)
        else:
            if not self._fallback_fitted:
                return []
            # Transform avec le vocabulaire fixé à l'indexation
            X_q = self._fallback_vec.transform([question]).toarray()
            q_emb = normalize(X_q, norm="l2")

        scores = cosine_similarity(q_emb, self.embeddings)[0]

        classement = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        resultats = []
        for idx, score in classement:
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            chunk["methode"] = "embeddings"
            resultats.append(chunk)

        return resultats


embedding_service = EmbeddingService()
