import cv2, json
import numpy as np
from ultralytics import YOLO
from keras_facenet import FaceNet
from scipy.spatial.distance import cdist

# Load models
detector = YOLO("models\yolov8m_200e.pt").to('cuda')  # smaller and faster
embedder = FaceNet()

# Load embeddings
with open("face_embeddings.json") as f:
    data = json.load(f)
names, vectors = [], []
for n, embs in data.items():
    for e in embs:
        names.append(n)
        vectors.append(np.array(e))
vectors = np.array(vectors)

# Start video
cap = cv2.VideoCapture(r"data\istockphoto-1532931548-640_adpp_is.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for speed
    h, w = frame.shape[:2]
    scale = 0.5
    small_frame = cv2.resize(frame, (int(w*scale), int(h*scale)))

    # Detect faces
    results = detector(small_frame, conf=0.3)[0]
    faces = []
    boxes = []
    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box / scale)  # scale back to original
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue
        boxes.append((x1, y1, x2, y2))
        faces.append(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

    if faces:
        # Embed all faces at once
        embeddings = embedder.embeddings(faces)
        # Compute distances vectorized
        dists = cdist(embeddings, vectors, metric='cosine')

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            idx = np.argmin(dists[i])
            name = names[idx] if dists[i, idx] < 0.6 else "Visitor"

            # Draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 4, y1), (0, 0, 0), -1)
            cv2.putText(frame, name, (x1 + 2, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
