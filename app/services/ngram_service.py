"""Service de modélisation du langage par n-grammes (Partie A).
Conforme aux codes de référence du cours ISIG-GOMA :
- mini_llm.py
- c2_bigramme.py
- c2_temperature.py

Implémenté en pur Python sans dépendance NLP externe.
"""

import math
import random
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


class NGramService:
    """Modèle de langage statistique n-gramme (Bigramme et Trigramme)

    avec échantillonnage par température et lissage de Laplace.
    """

    def __init__(self):
        self.tokens: List[str] = []
        self.vocab: set = set()
        self.vocab_size: int = 0
        self.unigram_counts: Counter = Counter()
        self.bigram_counts: Dict[str, Counter] = defaultdict(Counter)
        self.trigram_counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
        self.sentence_starts: List[str] = []

    def charger_corpus(self, source: Union[str, Path]) -> int:
        """Charge et tokenise le corpus depuis un fichier ou une chaîne."""
        if isinstance(source, Path) or (isinstance(source, str) and "\n" not in source and Path(source).exists()):
            with open(source, "r", encoding="utf-8") as f:
                lignes = [ligne.strip() for ligne in f if ligne.strip() and not ligne.strip().startswith("#")]
                texte = " ".join(lignes)
        else:
            lignes = [ligne.strip() for ligne in source.splitlines() if ligne.strip() and not ligne.strip().startswith("#")]
            texte = " ".join(lignes)

        return self.entrainer(texte)

    def tokeniser(self, texte: str) -> List[str]:
        """Nettoie et tokenise le texte en mots minuscules et points terminaux.

        Préserve les apostrophes et tirets dans les mots (ex: l'architecture, micro-services).
        """
        # Remplacer les ponctuations de fin de phrase par un token point isolé
        texte_propre = re.sub(r"[.!?]+", " . ", texte)
        # Supprimer les caractères spéciaux superflus tout en préservant lettres accentuées, apostrophes, tirets
        texte_propre = re.sub(r"[^\w\s\.'-]", " ", texte_propre)
        mots_bruts = texte_propre.lower().split()
        return mots_bruts

    def entrainer(self, texte: str) -> int:
        """Entraîne les modèles bigramme et trigramme sur le texte tokenisé."""
        self.tokens = self.tokeniser(texte)
        self.vocab = set(self.tokens)
        self.vocab_size = len(self.vocab)

        self.unigram_counts = Counter(self.tokens)
        self.bigram_counts = defaultdict(Counter)
        self.trigram_counts = defaultdict(Counter)
        self.sentence_starts = []

        if not self.tokens:
            return 0

        # Identifier les débuts de phrases
        nouvelle_phrase = True
        for t in self.tokens:
            if nouvelle_phrase and t != ".":
                self.sentence_starts.append(t)
                nouvelle_phrase = False
            elif t == ".":
                nouvelle_phrase = True

        # Comptage des bigrammes (w1 -> w2)
        for w1, w2 in zip(self.tokens, self.tokens[1:]):
            self.bigram_counts[w1][w2] += 1

        # Comptage des trigrammes ((w1, w2) -> w3)
        for w1, w2, w3 in zip(self.tokens, self.tokens[1:], self.tokens[2:]):
            self.trigram_counts[(w1, w2)][w3] += 1

        return len(self.tokens)

    # -------------------------------------------------------------
    # A2. BIGRAMME
    # -------------------------------------------------------------
    def proba_bigramme(self, w1: str, w2: str) -> float:
        """P(w2 | w1) = Count(w1, w2) / sum_w Count(w1, w)."""
        total = sum(self.bigram_counts[w1].values())
        if total == 0:
            return 0.0
        return self.bigram_counts[w1][w2] / total

    def proba_bigramme_fraction(self, w1: str, w2: str) -> Fraction:
        """Retourne la probabilité exacte sous forme de fraction (comme dans c2_bigramme.py)."""
        total = sum(self.bigram_counts[w1].values())
        if total == 0:
            return Fraction(0, 1)
        return Fraction(self.bigram_counts[w1][w2], total)

    def generer_bigramme(self, depart: Optional[str] = None, graine: Optional[int] = None, maximum: int = 25) -> str:
        """Génère une phrase avec le modèle bigramme standard."""
        if graine is not None:
            random.seed(graine)

        if not depart or depart not in self.vocab:
            depart = random.choice(self.sentence_starts) if self.sentence_starts else "le"

        mot = depart
        phrase = [depart]

        while mot != "." and len(phrase) < maximum:
            candidats = self.bigram_counts[mot]
            if not candidats:
                break
            mots_suivants = list(candidats.keys())
            poids = list(candidats.values())
            mot = random.choices(mots_suivants, weights=poids)[0]
            phrase.append(mot)

        if phrase[-1] != ".":
            phrase.append(".")

        return " ".join(phrase)

    # -------------------------------------------------------------
    # A3. TRIGRAMME
    # -------------------------------------------------------------
    def proba_trigramme(self, w1: str, w2: str, w3: str) -> float:
        """P(w3 | w1, w2) = Count(w1, w2, w3) / sum_w Count(w1, w2, w)."""
        total = sum(self.trigram_counts[(w1, w2)].values())
        if total == 0:
            return 0.0
        return self.trigram_counts[(w1, w2)][w3] / total

    def generer_trigramme(self, depart: Optional[Tuple[str, str]] = None, graine: Optional[int] = None, maximum: int = 25) -> str:
        """Génère une phrase avec le modèle trigramme, avec repli (backoff) bigramme si contexte absent."""
        if graine is not None:
            random.seed(graine)

        if not depart or depart not in self.trigram_counts:
            # Choisir une paire de démarrage fréquente
            paires_valides = [p for p in self.trigram_counts.keys() if p[0] != "." and p[1] != "."]
            if paires_valides:
                depart = random.choice(paires_valides)
            else:
                depart = ("le", "code")

        w1, w2 = depart
        phrase = [w1, w2]

        while phrase[-1] != "." and len(phrase) < maximum:
            candidats = self.trigram_counts[(w1, w2)]
            if candidats:
                mots_suivants = list(candidats.keys())
                poids = list(candidats.values())
                w3 = random.choices(mots_suivants, weights=poids)[0]
            else:
                # Repli (backoff) sur le bigramme si le trigramme est inconnu
                candidats_bi = self.bigram_counts[w2]
                if not candidats_bi:
                    break
                w3 = random.choices(list(candidats_bi.keys()), weights=list(candidats_bi.values()))[0]

            phrase.append(w3)
            w1, w2 = w2, w3

        if phrase[-1] != ".":
            phrase.append(".")

        return " ".join(phrase)

    # -------------------------------------------------------------
    # A4. TEMPÉRATURE
    # -------------------------------------------------------------
    @staticmethod
    def appliquer_temperature(probas: List[float], T: float) -> List[float]:
        """Formule du cours (diapo c2_temperature.py) :

        P'(mot) = P(mot)^(1/T) / somme des probabilités transformées.
        T bas (<1.0) renforce les favoris.
        T haut (>1.0) égalise les chances.
        """
        if T <= 0.01:
            # Température quasi-nulle : argmax (mot le plus probable = 1, autres = 0)
            max_idx = max(range(len(probas)), key=lambda i: probas[i])
            res = [0.0] * len(probas)
            res[max_idx] = 1.0
            return res

        inv_t = 1.0 / T
        puissances = [math.pow(p, inv_t) if p > 0 else 0.0 for p in probas]
        somme = sum(puissances)
        if somme == 0:
            # Équirépartition en cas d'annulation
            return [1.0 / len(probas)] * len(probas)
        return [p / somme for p in puissances]

    def generer_avec_temperature(
        self,
        depart: str = "le",
        temperature: float = 1.0,
        graine: Optional[int] = None,
        maximum: int = 25,
        ordre: str = "bigramme"
    ) -> str:
        """Génère une phrase en appliquant la formule de température du cours."""
        if graine is not None:
            random.seed(graine)

        if depart not in self.vocab:
            depart = "le"

        phrase = [depart]
        mot = depart

        while mot != "." and len(phrase) < maximum:
            candidats = self.bigram_counts[mot]
            if not candidats:
                break

            mots = list(candidats.keys())
            total = sum(candidats.values())
            probas_brutes = [candidats[m] / total for m in mots]

            probas_ajustees = self.appliquer_temperature(probas_brutes, temperature)
            mot = random.choices(mots, weights=probas_ajustees)[0]
            phrase.append(mot)

        if phrase[-1] != ".":
            phrase.append(".")

        return " ".join(phrase)

    # -------------------------------------------------------------
    # A5. PROBABILITÉ D'UNE PHRASE ET LISSAGE DE LAPLACE
    # -------------------------------------------------------------
    def proba_phrase_brute(self, phrase: str) -> Tuple[float, List[Dict[str, Union[str, float, str]]]]:
        """Calcule la probabilité d'une phrase avec le modèle bigramme brut.

        P(phrase) = P(w1) * prod_{i=2}^n P(w_i | w_{i-1}).
        Retourne la probabilité totale et le détail de chaque transition.
        """
        tokens = self.tokeniser(phrase)
        if not tokens:
            return 0.0, []

        details = []
        # P(w1) empirique
        total_tokens = len(self.tokens)
        p_init = self.unigram_counts[tokens[0]] / total_tokens if total_tokens > 0 else 0.0
        details.append({
            "transition": f"P({tokens[0]})",
            "proba": p_init,
            "formule": f"{self.unigram_counts[tokens[0]]} / {total_tokens}"
        })

        proba_totale = p_init
        for a, b in zip(tokens, tokens[1:]):
            p = self.proba_bigramme(a, b)
            c_ab = self.bigram_counts[a][b]
            c_a = sum(self.bigram_counts[a].values())
            details.append({
                "transition": f"P({b} | {a})",
                "proba": p,
                "formule": f"{c_ab} / {c_a}" if c_a > 0 else "0 / 0"
            })
            proba_totale *= p

        return proba_totale, details

    def proba_laplace_bigramme(self, a: str, b: str) -> float:
        """Formule du lissage de Laplace (Add-1 smoothing) :

        P_laplace(b | a) = (Count(a, b) + 1) / (sum_w Count(a, w) + |V|)
        où |V| est la taille totale du vocabulaire.
        """
        c_ab = self.bigram_counts[a][b]
        c_a = sum(self.bigram_counts[a].values())
        v = self.vocab_size
        return (c_ab + 1) / (c_a + v) if (c_a + v) > 0 else 1.0 / max(1, v)

    def proba_phrase_laplace(self, phrase: str) -> Tuple[float, float, List[Dict[str, Union[str, float, str]]]]:
        """Calcule la probabilité avec lissage de Laplace pour éviter les probabilités nulles.

        Retourne : (proba_totale, log_prob, details)
        """
        tokens = self.tokeniser(phrase)
        if not tokens:
            return 0.0, -float("inf"), []

        v = self.vocab_size
        total_tokens = len(self.tokens)

        # Lissage pour le mot initial P(w1) = (Count(w1) + 1) / (N + |V|)
        c_w1 = self.unigram_counts[tokens[0]]
        p_init = (c_w1 + 1) / (total_tokens + v) if (total_tokens + v) > 0 else 1.0 / v

        details = [{
            "transition": f"P_laplace({tokens[0]})",
            "proba": p_init,
            "formule": f"({c_w1} + 1) / ({total_tokens} + {v})"
        }]

        log_prob = math.log(p_init)
        proba_totale = p_init

        for a, b in zip(tokens, tokens[1:]):
            p = self.proba_laplace_bigramme(a, b)
            c_ab = self.bigram_counts[a][b]
            c_a = sum(self.bigram_counts[a].values())
            details.append({
                "transition": f"P_laplace({b} | {a})",
                "proba": p,
                "formule": f"({c_ab} + 1) / ({c_a} + {v})"
            })
            proba_totale *= p
            log_prob += math.log(p)

        return proba_totale, log_prob, details

    # -------------------------------------------------------------
    # Statistiques du modèle
    # -------------------------------------------------------------
    def statistiques(self) -> Dict[str, Union[int, List[Tuple[str, int]]]]:
        """Retourne les métriques fondamentales du corpus."""
        nb_bigrammes_uniques = sum(len(c) for c in self.bigram_counts.values())
        nb_trigrammes_uniques = sum(len(c) for c in self.trigram_counts.values())
        mots_frequents = self.unigram_counts.most_common(10)
        return {
            "total_tokens": len(self.tokens),
            "vocabulaire_taille": self.vocab_size,
            "bigrammes_uniques": nb_bigrammes_uniques,
            "trigrammes_uniques": nb_trigrammes_uniques,
            "nb_phrases_detectees": len(self.sentence_starts),
            "mots_les_plus_frequents": mots_frequents
        }
