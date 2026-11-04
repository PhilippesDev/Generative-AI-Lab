"""Service de recherche documentaire par TF-IDF (Partie C3).
Conforme au script c2_rag_complet.py de référence (M. Blaise CIZA).
Utilise scikit-learn (TfidfVectorizer, cosine_similarity).
Respecte le principe d'indexation unique (Consigne C9).
"""

from typing import Any, Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFService:
    def __init__(self):
        self.vectorizer: TfidfVectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=None,
            ngram_range=(1, 2)
        )
        self.chunks: List[Dict[str, Any]] = []
        self.tfidf_matrix = None
        self.est_indexe = False

    def indexer(self, chunks: List[Dict[str, Any]]) -> int:
        """Indexe l'ensemble des morceaux une seule fois."""
        self.chunks = chunks
        if not chunks:
            self.tfidf_matrix = None
            self.est_indexe = False
            return 0

        corpus_textes = [chunk["contenu"] for chunk in chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus_textes)
        self.est_indexe = True
        return len(chunks)

    def rechercher(self, question: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Recherche les top_k morceaux les plus similaires à la question."""
        if not self.est_indexe or self.tfidf_matrix is None or not self.chunks:
            return []

        # Vectoriser la question avec le vocabulaire appris
        question_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(question_vec, self.tfidf_matrix)[0]

        # Associer chaque score à son chunk
        classement = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        resultats = []
        for idx, score in classement:
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            chunk["methode"] = "tfidf"
            resultats.append(chunk)

        return resultats


tfidf_service = TFIDFService()
