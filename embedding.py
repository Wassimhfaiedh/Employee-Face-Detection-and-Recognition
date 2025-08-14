import os
import cv2
import numpy as np
from keras_facenet import FaceNet
import json

# =========================
# 1️⃣ Initialisation du modèle
# =========================
embedder = FaceNet()  # Crée l'embedder FaceNet pour extraire les vecteurs faciaux

# =========================
# 2️⃣ Préparation du dataset
# =========================
dataset_path = "faces"      # Chemin vers le dossier contenant les images
embeddings_dict = {}        # Dictionnaire pour stocker les embeddings de chaque personne

# =========================
# 3️⃣ Boucle sur chaque personne
# =========================
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    
    if not os.path.isdir(person_folder):
        continue  # Ignorer les fichiers qui ne sont pas des dossiers

    embeddings_list = []  # Liste pour stocker les embeddings de cette personne

    # =========================
    # 3a️⃣ Boucle sur chaque image
    # =========================
    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        img = cv2.imread(img_path)                  # Lire l'image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir BGR -> RGB

        try:
            # =========================
            # 3b️⃣ Extraction de l'embedding
            # =========================
            embedding = embedder.embeddings([img_rgb])[0]
            embeddings_list.append(embedding.tolist())
        except Exception as e:
            print(f"❌ Échec pour {img_path}: {e}")

    # Ajouter les embeddings extraits au dictionnaire principal
    if embeddings_list:
        embeddings_dict[person_name] = embeddings_list

# =========================
# 4️⃣ Sauvegarde des embeddings
# =========================
with open("face_embeddings.json", "w") as f:
    json.dump(embeddings_dict, f)

print("✅ Tous les embeddings faciaux ont été sauvegardés dans face_embeddings.json")
