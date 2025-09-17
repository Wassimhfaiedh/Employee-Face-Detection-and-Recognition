# ============================================================
# 📌 Imports nécessaires
# ============================================================
import os
import cv2
import json
import numpy as np
import insightface
from insightface.app import FaceAnalysis


# ============================================================
# 📌 Classe : Extracteur d'Embeddings corrigé
# ============================================================
class CorrectedEmbeddingExtractor:
    def __init__(self):
        """
        Initialisation du modèle InsightFace (buffalo_l).
        - ctx_id=0 -> utiliser le GPU si disponible, sinon CPU.
        - det_size -> taille de l’image pour la détection.
        - det_thresh -> seuil de détection.
        """
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)  

    # ------------------------------------------------------------
    # Fonction : Extraction de tous les visages d'une image
    # ------------------------------------------------------------
    def extract_all_faces(self, img):
        """
        Extraire tous les visages présents dans une image.
        Retourne une liste d’objets 'Face' ou une liste vide si erreur.
        """
        try:
            faces = self.app.get(img)
            return faces
        except Exception as e:
            print(f"❌ Erreur lors de l'extraction: {e}")
            return []

    # ------------------------------------------------------------
    # Fonction : Traiter toutes les images d’un dossier (personne)
    # ------------------------------------------------------------
    def process_person_folder(self, person_folder, person_name):
        """
        Parcourt toutes les images d'une personne donnée.
        - Extrait les visages et embeddings.
        - Stocke les embeddings dans une liste.
        Retour : (embeddings_list, processed_count)
        """
        embeddings_list = []
        processed_count = 0
        
        # Boucler sur toutes les images du dossier
        for img_name in os.listdir(person_folder):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue  # Ignorer les fichiers non-images
                
            img_path = os.path.join(person_folder, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"❌ Impossible de lire: {img_name}")
                continue
            
            # Redimensionner si l'image est trop grande (meilleure détection)
            if max(img.shape) > 2000:
                scale = 2000 / max(img.shape)
                new_w = int(img.shape[1] * scale)
                new_h = int(img.shape[0] * scale)
                img = cv2.resize(img, (new_w, new_h))
            
            # Extraire les visages
            faces = self.extract_all_faces(img)
            
            if not faces:
                print(f"⚠️ Aucun visage détecté dans: {img_name}")
                continue
            
            # Sauvegarder les embeddings de chaque visage détecté
            for i, face in enumerate(faces):
                embedding_data = {
                    'embedding': face.embedding.tolist(),  # vecteur d’embedding
                    'det_score': float(face.det_score),   # score de détection
                    'source_image': img_name,             # nom de l’image source
                    'face_index': i                       # index du visage dans l’image
                }
                embeddings_list.append(embedding_data)
                processed_count += 1
                print(f"✅ Visage {i+1} extrait de {img_name} (score: {face.det_score:.2f})")
        
        return embeddings_list, processed_count

    # ------------------------------------------------------------
    # Fonction : Construire la base de données d’embeddings
    # ------------------------------------------------------------
    def build_database(self, dataset_path, json_path):
        """
        Crée/Met à jour une base d’embeddings à partir d’un dataset.
        - dataset_path : chemin vers le dossier contenant les sous-dossiers de chaque personne.
        - json_path : chemin vers le fichier JSON des embeddings.
        """
        database = {}
        
        # Charger les embeddings existants si JSON déjà présent
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    existing_data = json.load(f)
                    database = existing_data.get('embeddings', {})
                except:
                    database = {}
        
        # --------------------------------------------------------
        # Supprimer les personnes qui n'existent plus dans dataset
        # --------------------------------------------------------
        existing_people = set(os.listdir(dataset_path))
        to_delete = [name for name in database if name not in existing_people]
        for name in to_delete:
            del database[name]
            print(f"🗑️ Supprimé {name} car le dossier est manquant.")
        
        total_faces = sum(len(v) for v in database.values())
        
        # --------------------------------------------------------
        # Ajouter de nouveaux embeddings pour les nouvelles personnes
        # --------------------------------------------------------
        for person_name in os.listdir(dataset_path):
            person_folder = os.path.join(dataset_path, person_name)
            
            if not os.path.isdir(person_folder):
                continue  # Ignorer si ce n'est pas un dossier

            if person_name in database:
                print(f"⚠️ {person_name} existe déjà dans la base.")
                continue

            embeddings, count = self.process_person_folder(person_folder, person_name)
            
            if embeddings:
                database[person_name] = embeddings
                total_faces += count
                print(f"✅ Embeddings ajoutés pour {person_name}")
        
        # --------------------------------------------------------
        # Sauvegarde des embeddings en JSON
        # --------------------------------------------------------
        metadata = {
            'version': '2.2',
            'model': 'insightface_buffalo_l',
            'total_people': len(database),
            'total_faces': total_faces,
            'embeddings': database
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Base mise à jour. Total personnes: {len(database)}, Total visages: {total_faces}")


# ============================================================
# 📌 Exemple d’utilisation
# ============================================================
if __name__ == "__main__":
    extractor = CorrectedEmbeddingExtractor()
    extractor.build_database("faces", "face_embeddings_corrected.json")
