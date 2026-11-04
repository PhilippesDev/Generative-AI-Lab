"""Service pour le modèle GPT-2 (Partie B1).
Conforme au script c2_gpt2_local.py de référence (M. Blaise CIZA).
Permet de faire varier top_k, do_sample, max_new_tokens.
"""

from typing import Any, Dict


class GPT2Service:
    def __init__(self):
        self._pipeline = None
        self._charge = False
        self._erreur = None

    def charger_modele(self):
        """Charge le modèle HuggingFace GPT-2 à la demande (lazy-loading) pour économiser la RAM."""
        if self._charge:
            return self._pipeline
        try:
            from transformers import pipeline
            self._pipeline = pipeline("text-generation", model="openai-community/gpt2")
            self._charge = True
            return self._pipeline
        except Exception as e:
            self._erreur = str(e)
            return None

    def generer(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        do_sample: bool = True,
        top_k: int = 50,
        repetition_penalty: float = 1.2
    ) -> Dict[str, Any]:
        """Génère du texte avec GPT-2 selon les paramètres demandés."""
        pipe = self.charger_modele()
        if pipe is None:
            # Fallback didactique clair si transformers/torch n'est pas encore installé
            texte_genere = (
                f"{prompt} is a foundational concept in computer science. "
                f"However, GPT-2 generates primarily in English and can hallucinate on French technical prompts."
            )
            return {
                "prompt": prompt,
                "generated_text": texte_genere,
                "modele": "openai-community/gpt2 (simulation éducative - torch/transformers absent)",
                "parametres": {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                    "top_k": top_k,
                    "repetition_penalty": repetition_penalty
                },
                "remarque": self._erreur or "Modèle simulé si bibliothèques non installées"
            }

        resultat = pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_k=top_k,
            repetition_penalty=repetition_penalty
        )
        return {
            "prompt": prompt,
            "generated_text": resultat[0]["generated_text"],
            "modele": "openai-community/gpt2",
            "parametres": {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty
            }
        }


gpt2_service = GPT2Service()
