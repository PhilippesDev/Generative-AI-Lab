"""Service de gestion des documents sources du RAG (Partie C1 & C7).
Permet de lister, lire, ajouter, supprimer des documents et d'extraire l'ensemble des chunks.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from app.services.chunking_service import chunking_service


class DocumentService:
    def __init__(self, documents_dir: Optional[Path] = None):
        if documents_dir is None:
            self.documents_dir = Path(__file__).resolve().parent.parent.parent / "data" / "documents"
        else:
            self.documents_dir = documents_dir
        self.documents_dir.mkdir(parents=True, exist_ok=True)

    def lister_documents(self) -> List[Dict[str, Any]]:
        """Liste tous les documents disponibles dans le répertoire."""
        resultats = []
        for fichier in sorted(self.documents_dir.glob("*.txt")):
            taille_octets = fichier.stat().st_size
            with open(fichier, "r", encoding="utf-8", errors="ignore") as f:
                lignes = f.readlines()
                premiere_ligne = lignes[0].strip("# \n") if lignes else fichier.stem
                nb_mots = sum(len(l.split()) for l in lignes)

            resultats.append({
                "filename": fichier.name,
                "titre": premiere_ligne,
                "chemin": str(fichier),
                "taille_octets": taille_octets,
                "nb_mots": nb_mots,
                "nb_lignes": len(lignes)
            })
        return resultats

    def lire_document(self, filename: str) -> Optional[str]:
        fichier = self.documents_dir / filename
        if not fichier.exists():
            return None
        with open(fichier, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def ajouter_document(self, filename: str, contenu: str) -> Dict[str, Any]:
        if not filename.endswith(".txt"):
            filename = f"{filename}.txt"
        fichier = self.documents_dir / filename
        with open(fichier, "w", encoding="utf-8") as f:
            f.write(contenu)
        return {
            "filename": filename,
            "statut": "créé",
            "nb_mots": len(contenu.split())
        }

    def supprimer_document(self, filename: str) -> bool:
        fichier = self.documents_dir / filename
        if fichier.exists():
            fichier.unlink()
            return True
        return False

    def charger_tous_les_chunks(self) -> List[Dict[str, Any]]:
        """Extrait tous les chunks avec métadonnées de l'ensemble des documents."""
        tous_les_chunks = []
        for doc in self.lister_documents():
            contenu = self.lire_document(doc["filename"])
            if contenu:
                doc_id = Path(doc["filename"]).stem
                chunks = chunking_service.decouper_texte(
                    texte=contenu,
                    doc_id=doc_id,
                    filename=doc["filename"],
                    titre_doc=doc["titre"]
                )
                tous_les_chunks.extend(chunks)
        return tous_les_chunks


document_service = DocumentService()
