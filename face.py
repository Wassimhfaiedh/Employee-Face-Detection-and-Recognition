import cv2
import numpy as np
import json
from scipy.spatial.distance import cosine
import insightface
from insightface.app import FaceAnalysis
import os

def compute_iou(bbox1, bbox2):
    """Calculate Intersection over Union (IoU) for two bounding boxes"""
    x1, y1, x2, y2 = bbox1
    x1_b, y1_b, x2_b, y2_b = bbox2
    
    xi1 = max(x1, x1_b)
    yi1 = max(y1, y1_b)
    xi2 = min(x2, x2_b)
    yi2 = min(y2, y2_b)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    bbox1_area = (x2 - x1) * (y2 - y1)
    bbox2_area = (x2_b - x1_b) * (y2_b - y1_b)
    
    union_area = bbox1_area + bbox2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

class DebugFaceRecognizer:
    def __init__(self, embeddings_path):
        self.recognizer = FaceAnalysis(name='buffalo_l')
        self.recognizer.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)
        self.load_embeddings(embeddings_path)
        self.face_tracker = {}  
        self.face_id_counter = 0  
        self.locked_faces = {}  
    
    def load_embeddings(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.names = []
            self.vectors = []
            
            for person_name, faces in data['embeddings'].items():
                for face_data in faces:
                    embedding = np.array(face_data['embedding'])
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    self.names.append(person_name)
                    self.vectors.append(embedding)
            
            self.vectors = np.array(self.vectors)
        except:
            self.names = []
            self.vectors = np.array([])
    
    def normalize_embedding(self, embedding):
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding
    
    def simple_recognition(self, embedding):
        if len(self.vectors) == 0:
            return "Visitor", 0, 1
        
        embedding = self.normalize_embedding(embedding)
        
        similarities = []
        for db_emb in self.vectors:
            try:
                similarity = 1 - cosine(embedding, db_emb)
                similarities.append(similarity)
            except:
                similarities.append(0)
        
        if similarities:
            max_similarity = max(similarities)
            max_index = similarities.index(max_similarity)
            adaptive_threshold = 0.6 if max_similarity > 0.8 else 0.35
            name = self.names[max_index] if max_similarity > adaptive_threshold else "Visitor"
            distance = 1 - max_similarity
            return name, max_similarity, distance
        
        return "Visitor", 0, 1
    
    def draw_label(self, frame, text, x1, y1, color):
        """Draw a rounded label with semi-transparent background and black text"""
        font_scale = 0.5
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1

        (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        label_width = w + 12
        label_height = h + 8
        radius = 8
        top_left = (x1, y1 - label_height - 10)
        bottom_right = (x1 + label_width, y1 - 10)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, top_left, bottom_right, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, (top_left[0] + radius, top_left[1] + radius), radius, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, (bottom_right[0] - radius, top_left[1] + radius), radius, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, (top_left[0] + radius, bottom_right[1] - radius), radius, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, (bottom_right[0] - radius, bottom_right[1] - radius), radius, color, -1, lineType=cv2.LINE_AA)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        cv2.putText(frame, text, (x1 + 6, y1 - 14), font, font_scale, (0, 0, 0), thickness, lineType=cv2.LINE_AA)

    def process_video_with_debug(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return
        
        frame_count = 0
        confidence_threshold = 0.8
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            faces = self.recognizer.get(frame)
            current_faces = {}
            
            for i, face in enumerate(faces):
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                
                matched_id = None
                max_iou = 0
                for fid, (name, conf, prev_bbox) in self.face_tracker.items():
                    iou = compute_iou(bbox, prev_bbox)
                    if iou > max_iou and iou > 0.5:
                        max_iou = iou
                        matched_id = fid
                
                name, confidence, distance = self.simple_recognition(face.embedding)
                print(f"Frame {frame_count}, Face {i}: Name={name}, Confidence={confidence:.2f}, Distance={distance:.2f}")
                
                if matched_id in self.locked_faces:
                    name, confidence = self.locked_faces[matched_id]
                
                if confidence > confidence_threshold and name != "Visitor":
                    self.locked_faces[matched_id] = (name, confidence)
                
                if matched_id is not None:
                    self.face_tracker[matched_id] = (name, confidence, bbox)
                else:
                    self.face_id_counter += 1
                    matched_id = self.face_id_counter
                    self.face_tracker[matched_id] = (name, confidence, bbox)
                
                current_faces[matched_id] = (name, confidence, bbox)
                
                # Draw thin bounding box (green for known, red for visitor)
                box_color = (0, 200, 0) if name != "Visitor" else (0, 0, 200)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 1)
                
                # Draw rounded label
                label = f"{name} ({confidence:.2f})"
                self.draw_label(frame, label, x1, y1, box_color)
            
            self.face_tracker = current_faces
            cv2.imshow("Face Recognition", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recognizer = DebugFaceRecognizer("face_embeddings_corrected.json")
    recognizer.process_video_with_debug("data\istockphoto-1532931548-640_adpp_is.mp4")
