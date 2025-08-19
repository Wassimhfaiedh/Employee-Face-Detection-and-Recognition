import os
import cv2
import numpy as np
from keras_facenet import FaceNet
import json

# =========================
# 1️⃣ Initialize the model
# =========================
embedder = FaceNet()  

# =========================
# 2️⃣ Prepare the dataset
# =========================
dataset_path = "faces"
json_path = "face_embeddings.json"

# Load existing embeddings if the file exists
if os.path.exists(json_path):
    with open(json_path, "r") as f:
        embeddings_dict = json.load(f)
else:
    embeddings_dict = {}

# =========================
# 2️⃣1️⃣ Delete people not in dataset anymore
# =========================
if os.path.exists(dataset_path):
    existing_people = set(os.listdir(dataset_path))
    to_delete = [name for name in embeddings_dict if name not in existing_people]
    for name in to_delete:
        del embeddings_dict[name]
        print(f"🗑️  Removed {name} from embeddings because folder is missing.")
else:
    print(f"Error: Dataset path '{dataset_path}' does not exist.")
    exit(1)

# =========================
# 3️⃣ Loop through each person to add new embeddings
# =========================
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    
    if not os.path.isdir(person_folder):
        continue  

    # Skip if already in JSON (we already have embeddings)
    if person_name in embeddings_dict:
        print(f"⚠️  {person_name} already exists in the database.")
        continue

    embeddings_list = []

    # Loop through each image
    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        
        # Check if file is an image
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Failed to read image: {img_path}")
            continue
                  
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

        try:
            embedding = embedder.embeddings([img_rgb])[0]
            embeddings_list.append(embedding.tolist())
            print(f"✅ Processed {img_name} for {person_name}")
        except Exception as e:
            print(f"❌ Failed to process {img_path}: {e}")

    if embeddings_list:
        embeddings_dict[person_name] = embeddings_list
        print(f"✅ Added {len(embeddings_list)} embeddings for {person_name}")

# =========================
# 4️⃣ Save updated embeddings
# =========================
with open(json_path, "w") as f:
    json.dump(embeddings_dict, f, indent=4)

print(f"✅ Embeddings JSON has been updated with {len(embeddings_dict)} people.")