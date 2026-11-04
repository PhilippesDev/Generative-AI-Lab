"""Service d'interrogation du modèle local Ollama.
Conforme au script c2_ollama_chat.py de référence (M. Blaise CIZA).
Utilise uniquement la bibliothèque standard (urllib, json).
"""

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


class OllamaService:
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self.base_url = (base_url or os.environ.get("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        # Priorité au modèle configuré dans .env, ou qwen2.5-coder:3b / llama3.2
        self.model = model or os.environ.get("MODELE", "qwen2.5-coder:3b")

    def verifier_disponibilite(self) -> Dict[str, Any]:
        """Vérifie si le serveur Ollama est actif et liste les modèles installés."""
        url = f"{self.base_url}/api/tags"
        req = urllib.request.Request(url, headers={"User-Agent": "GenAILab-FastAPI"})
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                est_present = any(self.model in m for m in models)
                return {
                    "disponible": True,
                    "url": self.base_url,
                    "model_cible": self.model,
                    "model_present": est_present,
                    "modeles_installes": models,
                }
        except Exception as e:
            return {
                "disponible": False,
                "url": self.base_url,
                "model_cible": self.model,
                "model_present": False,
                "modeles_installes": [],
                "erreur": f"Ollama injoignable sur {self.base_url} : {str(e)}"
            }

    def demander(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        format_json: bool = False,
        timeout: int = 120
    ) -> str:
        """Envoie les messages à Ollama et renvoie le texte de réponse."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature}
        }
        if format_json:
            payload["format"] = "json"

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("message", {}).get("content", "").strip()
        except urllib.error.URLError as e:
            return (
                f"[Erreur Ollama] Impossible de joindre {self.base_url}. "
                f"Vérifiez qu'Ollama est lancé (ollama serve). Détail: {e}"
            )
        except Exception as e:
            return f"[Erreur Ollama] Exception lors de la requête: {e}"


# Instance globale
ollama_service = OllamaService()
