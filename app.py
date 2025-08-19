import cv2
import numpy as np
from ultralytics import YOLO
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine
import json
import torch

device='cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load YOLOv8 face detector with tracker
detector = YOLO("models\\yolov8m_200e.pt").to(device)

# Load FaceNet embedder
embedder = FaceNet()

# Load saved embeddings
with open("face_embeddings.json", "r") as f:
    known_embeddings = json.load(f)

# Flatten known embeddings using the new method
names, vectors = [], []
for n, embs in known_embeddings.items():
    for e in embs:
        names.append(n)
        vectors.append(np.array(e))
vectors = np.array(vectors)  

threshold = 0.6
cap = cv2.VideoCapture("data\istockphoto-1532931548-640_adpp_is.mp4")
frame_skip = 3
frame_count = 0
scale_factor = 0.5

# Persistent tracker ID mapping: YOLO tracker ID → persistent ID
tracker_id_map = {}
next_persistent_id = 1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    h_orig, w_orig = frame.shape[:2]
    frame_resized = cv2.resize(frame, (int(w_orig * scale_factor), int(h_orig * scale_factor)))

    # Run YOLO with tracker
    results = detector.track(frame_resized, conf=0.5, persist=True, tracker="botsort.yaml")[0]

    faces_rgb = []
    face_coords = []
    tracker_ids = []

    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            tracker_id = int(box.id.item()) if box.id is not None else -1

            # Map YOLO tracker ID to persistent ID
            if tracker_id not in tracker_id_map:
                tracker_id_map[tracker_id] = next_persistent_id
                next_persistent_id += 1
            persistent_id = tracker_id_map[tracker_id]

            # Rescale coordinates
            x1 = int(x1 / scale_factor)
            y1 = int(y1 / scale_factor)
            x2 = int(x2 / scale_factor)
            y2 = int(y2 / scale_factor)

            # Padding
            face_area = (x2 - x1) * (y2 - y1)
            padding = max(5, min(10, int(np.sqrt(face_area) * 0.1)))
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w_orig, x2 + padding)
            y2 = min(h_orig, y2 + padding)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            faces_rgb.append(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            face_coords.append((x1, y1, x2, y2))
            tracker_ids.append(persistent_id)

    if faces_rgb:
        try:
            embeddings = embedder.embeddings(faces_rgb)
            
            # Store recognition results 
            recognition_results = []
            used_names = set()  
            
            for (x1, y1, x2, y2), emb, pid in zip(face_coords, embeddings, tracker_ids):
                distances = [cosine(emb, v) for v in vectors]
                min_dist = min(distances)
                min_idx = distances.index(min_dist)

                recognized = min_dist < threshold
                name = names[min_idx] if recognized else "Visitor"
                confidence = 1 - min_dist
                
                # Skip this face if it's a duplicate of a recognized person
                if recognized and name in used_names:
                    continue
                
                # Add to used names if it's a valid recognition
                if recognized:
                    used_names.add(name)
                
                box_color = (0, 255, 0) if recognized else (0, 0, 255)
                label = f"#{pid}: {name} ({confidence:.2f})" if recognized else f"#{pid}: Visitor"
                
                # Draw normal rectangle bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                # Label background with increased size, directly connected to bbox
                (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_x1 = x1
                label_y1 = y1 - text_h - baseline - 10
                label_x2 = x1 + text_w + 25
                label_y2 = y1

                overlay = frame.copy()
                alpha = 0.7
                cv2.rectangle(overlay, (label_x1, label_y1), (label_x2, label_y2), box_color, cv2.FILLED)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                cv2.putText(frame, label, (label_x1 + 7, label_y2 - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                # Confidence bar
                bar_width = x2 - x1
                bar_height = 8
                bar_x1 = x1
                bar_y1 = y2 + 10
                bar_x2 = x1 + int(bar_width * confidence)
                bar_y2 = bar_y1 + bar_height

                cv2.rectangle(frame, (bar_x1, bar_y1), (x2, bar_y2), (40, 40, 40), cv2.FILLED)
                cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), box_color, cv2.FILLED)

        except Exception as e:
            print(f"Embedding error: {e}")

    cv2.imshow("Face Recognition with Persistent ID", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
