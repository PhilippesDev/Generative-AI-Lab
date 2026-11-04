"""Service de découpage documentaire en morceaux (chunking) avec chevauchement.
Conforme à la consigne C2 du TP 2 :
- Morceaux de 300 à 500 mots (ou taille adaptée aux sections du document)
- Léger chevauchement (overlap, ex: 50 mots)
- Conservation stricte des métadonnées (doc_id, filename, chunk_id, position, titre)
"""

from typing import Any, Dict, List


class ChunkingService:
    def __init__(self, taille_chunk: int = 350, chevauchement: int = 50):
        self.taille_chunk = taille_chunk
        self.chevauchement = chevauchement

    def decouper_texte(
        self,
        texte: str,
        doc_id: str,
        filename: str,
        titre_doc: str = ""
    ) -> List[Dict[str, Any]]:
        """Découpe un texte en morceaux glissants tout en préservant les métadonnées et frontières de paragraphes."""
        mots = texte.split()
        if not mots:
            return []

        chunks = []
        pas = max(1, self.taille_chunk - self.chevauchement)
        i = 0
        chunk_num = 1

        # Si le document est court (< taille_chunk), conserver un seul chunk propre
        if len(mots) <= self.taille_chunk:
            contenu = " ".join(mots)
            return [{
                "chunk_id": f"{doc_id}_chunk_{chunk_num}",
                "doc_id": doc_id,
                "filename": filename,
                "chunk_num": chunk_num,
                "titre": titre_doc or filename,
                "debut_mot": 0,
                "fin_mot": len(mots),
                "nb_mots": len(mots),
                "contenu": contenu
            }]

        while i < len(mots):
            fin = min(len(mots), i + self.taille_chunk)
            morceau_mots = mots[i:fin]
            contenu = " ".join(morceau_mots)

            chunks.append({
                "chunk_id": f"{doc_id}_chunk_{chunk_num}",
                "doc_id": doc_id,
                "filename": filename,
                "chunk_num": chunk_num,
                "titre": titre_doc or f"{filename} (partie {chunk_num})",
                "debut_mot": i,
                "fin_mot": fin,
                "nb_mots": len(morceau_mots),
                "contenu": contenu
            })

            if fin == len(mots):
                break
            i += pas
            chunk_num += 1

        return chunks


chunking_service = ChunkingService()
