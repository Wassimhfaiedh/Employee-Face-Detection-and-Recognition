import cv2
import numpy as np
import json
from scipy.spatial.distance import cosine
import insightface
from insightface.app import FaceAnalysis
import os
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
from datetime import datetime, timedelta
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = ''
socketio = SocketIO(app, async_mode='threading')

# Define video paths for cameras
camera_1_path = "data\\istockphoto-1532931548-640_adpp_is.mp4"
camera_2_path = "data\Realistic_Factory_Surveillance_Video_Generation.mp4"

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

class FaceRecognizer:
    def __init__(self, embeddings_path, video_path, processor_id):
        self.recognizer = FaceAnalysis(name='buffalo_l')
        self.recognizer.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)
        self.video_path = video_path
        self.processor_id = processor_id
        self.load_embeddings(embeddings_path)
        self.face_tracker = {}
        self.face_id_counter = 0
        self.locked_faces = {}
        self.running = False
        self.counting = False
        self.paused = False
        self.detections = []
        self.pending_visitors = {}  
        self.VISITOR_CONFIRMATION_SECONDS = 6
        self.cap = cv2.VideoCapture(video_path)
        self.selected_person_id = None  # Track selected person
        self.selected_camera_id = None  # Track selected camera
        self._processing_task = None
        
        if not self.cap.isOpened():
            print(f"Error: Could not open video {video_path} for processor {processor_id}")
            self.cap = None
            socketio.emit(f'update_status_{self.processor_id}', {'message': 'Error: Could not open video'})
            return

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.wait_time = int(1000 / self.fps) if self.fps > 0 else 30
        self.ret, self.first_frame = self.cap.read()
        if not self.ret:
            print(f"Error: Could not read first frame from {video_path} for processor {processor_id}")
            self.cap = None
            socketio.emit(f'update_status_{self.processor_id}', {'message': 'Error: Could not read first frame'})
            return

        # Display first frame immediately
        frame_data = self.encode_frame(self.first_frame)
        socketio.emit(f'video_frame_{self.processor_id}', {'frame': frame_data})

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
            print(f"Loaded {len(self.names)} embeddings from {path}")
        except Exception as e:
            print(f"Error loading embeddings: {e}")
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

    def encode_frame(self, frame):
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return base64.b64encode(buffer).decode('utf-8')

    def process_frame(self, frame):
        if frame is None:
            return None

        display_frame = frame.copy()
        faces = self.recognizer.get(frame)
        current_faces = {}
        known_count = 0
        visitor_count = 0
        current_time = datetime.now()

        # Track face IDs in the current frame
        for i, face in enumerate(faces):
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            matched_id = None
            max_iou = 0
            for fid, (name, conf, prev_bbox, _) in self.face_tracker.items():
                iou = compute_iou(bbox, prev_bbox)
                if iou > max_iou and iou > 0.5:
                    max_iou = iou
                    matched_id = fid
            
            name, confidence, distance = self.simple_recognition(face.embedding)
            print(f"Processor {self.processor_id}, Face {i}: Name={name}, Confidence={confidence:.2f}, Distance={distance:.2f}")
            
            if matched_id in self.locked_faces:
                name, confidence = self.locked_faces[matched_id]
            
            if confidence > 0.8 and name != "Visitor":
                self.locked_faces[matched_id] = (name, confidence)
            
            if matched_id is not None:
                self.face_tracker[matched_id] = (name, confidence, bbox, current_time.strftime('%Y-%m-%d %H:%M:%S'))
            else:
                self.face_id_counter += 1
                matched_id = self.face_id_counter
                self.face_tracker[matched_id] = (name, confidence, bbox, current_time.strftime('%Y-%m-%d %H:%M:%S'))
            
            current_faces[matched_id] = (name, confidence, bbox, current_time.strftime('%Y-%m-%d %H:%M:%S'))
            
            # Handle detections
            if name != "Visitor":
                # Remove any "Visitor" entry with the same face_id from detections and pending_visitors
                self.detections = [d for d in self.detections if d['face_id'] != matched_id or d['name'] != "Visitor"]
                if matched_id in self.pending_visitors:
                    del self.pending_visitors[matched_id]
                # Add employee detection if name not already in detections
                if not any(d['name'] == name for d in self.detections):
                    face_img = frame[max(0, y1-20):min(frame.shape[0], y2+20), max(0, x1-20):min(frame.shape[1], x2+20)]
                    if face_img.size > 0:
                        _, buffer = cv2.imencode('.jpg', face_img)
                        face_base64 = base64.b64encode(buffer).decode('utf-8')
                        detection = {
                            'name': name,
                            'confidence': round(confidence, 2),
                            'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                            'face_image': f'data:image/jpeg;base64,{face_base64}',
                            'face_id': matched_id
                        }
                        self.detections.append(detection)
                        self.detections.sort(key=lambda x: x['timestamp'])
                        if len(self.detections) > 10:
                            self.detections.pop(0)
            else:
                # Handle visitor: add to pending_visitors if not already recognized as employee
                if matched_id not in self.locked_faces or self.locked_faces[matched_id][0] == "Visitor":
                    if matched_id in self.pending_visitors:
                        # Update timestamp but keep initial_time
                        self.pending_visitors[matched_id]['timestamp'] = current_time.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        face_img = frame[max(0, y1-20):min(frame.shape[0], y2+20), max(0, x1-20):min(frame.shape[1], x2+20)]
                        if face_img.size > 0:
                            _, buffer = cv2.imencode('.jpg', face_img)
                            face_base64 = base64.b64encode(buffer).decode('utf-8')
                            self.pending_visitors[matched_id] = {
                                'name': name,
                                'confidence': round(confidence, 2),
                                'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                                'face_image': f'data:image/jpeg;base64,{face_base64}',
                                'face_id': matched_id,
                                'initial_time': current_time
                            }
                    
                    # Confirm visitor after VISITOR_CONFIRMATION_SECONDS
                    if (current_time - self.pending_visitors[matched_id]['initial_time']).total_seconds() >= self.VISITOR_CONFIRMATION_SECONDS:
                        if not any(d['name'] == "Visitor" for d in self.detections):
                            # Create a new detection without initial_time to avoid serialization issues
                            detection = {
                                'name': self.pending_visitors[matched_id]['name'],
                                'confidence': self.pending_visitors[matched_id]['confidence'],
                                'timestamp': self.pending_visitors[matched_id]['timestamp'],
                                'face_image': self.pending_visitors[matched_id]['face_image'],
                                'face_id': self.pending_visitors[matched_id]['face_id']
                            }
                            self.detections.append(detection)
                            self.detections.sort(key=lambda x: x['timestamp'])
                            if len(self.detections) > 10:
                                self.detections.pop(0)
                        del self.pending_visitors[matched_id]
            
            # Draw thin bounding box and label
            box_color = (255, 0, 0) if matched_id == self.selected_person_id and self.processor_id == self.selected_camera_id else (0, 200, 0) if name != "Visitor" else (0, 0, 200)   
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 1)
            label = f"{name} ({confidence:.2f})"
            self.draw_label(display_frame, label, x1, y1, box_color)

            if name == "Visitor":
                visitor_count += 1
            else:
                known_count += 1

        # Remove detections and pending visitors for faces no longer in frame
        self.detections = [d for d in self.detections if d['face_id'] in current_faces]
        self.pending_visitors = {k: v for k, v in self.pending_visitors.items() if k in current_faces}
        self.face_tracker = current_faces

        # Emit detection data
        socketio.emit(f'update_detections_{self.processor_id}', {'detections': self.detections})
        
        # Emit analytics counts
        socketio.emit(f'update_analytics_{self.processor_id}', {
            'known_count': known_count,
            'visitor_count': visitor_count
        })

        return display_frame

    def process_video(self):
        self.running = True
        while self.running and self.cap and self.cap.isOpened():
            if self.paused:
                socketio.sleep(0.1)
                continue

            if self.counting:
                ret, frame = self.cap.read()
                if not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        print(f"Error: Could not reset video {self.video_path} for processor {self.processor_id}")
                        socketio.emit(f'update_status_{self.processor_id}', {'message': 'Error: Could not reset video'})
                        break
            else:
                frame = self.first_frame.copy() if self.first_frame is not None else None

            processed_frame = self.process_frame(frame)
            if processed_frame is not None:
                frame_data = self.encode_frame(processed_frame)
                socketio.emit(f'video_frame_{self.processor_id}', {'frame': frame_data})

            socketio.sleep(self.wait_time / 1000.0)

        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self._processing_task = None
        socketio.emit(f'update_status_{self.processor_id}', {'message': 'Face detection stopped'})

    def start(self):
        if self._processing_task is not None:
            print(f"Processor {self.processor_id} is already running")
            return
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"Error: Could not reopen video {self.video_path} for processor {self.processor_id}")
            socketio.emit(f'update_status_{self.processor_id}', {'message': 'Error: Could not reopen video'})
            return
        self.counting = True
        self.running = True
        self.paused = False
        socketio.emit(f'update_status_{self.processor_id}', {'message': 'System fully resumed'})
        self._processing_task = socketio.start_background_task(self.process_video)

    def toggle_pause_resume(self):
        self.paused = not self.paused
        if self.paused:
            if self.cap:
                self.cap.release()
                self.cap = None
            socketio.emit(f'update_status_{self.processor_id}', {'message': 'System fully paused'})
            print("System fully paused, detection stopped")
            self.detections = []
            self.pending_visitors = {}
            self.selected_person_id = None
            self.selected_camera_id = None
            socketio.emit(f'update_detections_{self.processor_id}', {'detections': []})
            socketio.emit(f'update_analytics_{self.processor_id}', {
                'known_count': 0,
                'visitor_count': 0
            })
        else:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                print(f"Error: Could not reopen video {self.video_path} for processor {self.processor_id}")
                socketio.emit(f'update_status_{self.processor_id}', {'message': 'Error: Could not reopen video'})
                return
            socketio.emit(f'update_status_{self.processor_id}', {'message': 'System fully resumed'})
            print("System fully resumed, detection restarted")

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def process_single_frame(self):
        """Process a single frame for the selected person"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    socketio.emit(f'update_status_{self.processor_id}', {'message': 'Error: Could not read frame'})
                    return
            processed_frame = self.process_frame(frame)
            if processed_frame is not None:
                frame_data = self.encode_frame(processed_frame)
                socketio.emit('person_frame', {'frame': frame_data, 'camera_id': self.processor_id})

# Global processors
processors = {}

def stop_processor(pid):
    if pid in processors:
        processors[pid].running = False
        processors[pid].release()
        del processors[pid]

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('select_camera')
def handle_select_camera(data):
    camera_id = data.get('camera_id')
    if camera_id == '1':
        active_pids = ['1']
        video_paths = {'1': camera_1_path}
    elif camera_id == '2':
        active_pids = ['2']
        video_paths = {'2': camera_2_path}
    elif camera_id == 'both':
        active_pids = ['1', '2']
        video_paths = {'1': camera_1_path, '2': camera_2_path}
    else:
        socketio.emit('update_status_1', {'message': 'Invalid camera selection'})
        return

    # Stop non-active processors
    for pid in list(processors.keys()):
        if pid not in active_pids:
            stop_processor(pid)

    # Start or resume active processors
    for pid in active_pids:
        if not os.path.exists(video_paths[pid]):
            socketio.emit(f'update_status_{pid}', {'message': f'Video file not found: {video_paths[pid]}'})
            continue
        if pid not in processors:
            processors[pid] = FaceRecognizer("face_embeddings_corrected.json", video_paths[pid], pid)
        if processors[pid].cap is None:
            socketio.emit(f'update_status_{pid}', {'message': f'Failed to process video: {video_paths[pid]}'})
            continue
        processors[pid].start()

@socketio.on('toggle_pause_resume')
def handle_toggle_pause_resume(data):
    for pid in list(processors.keys()):
        if pid in processors:
            processors[pid].toggle_pause_resume()

@socketio.on('highlight_person')
def handle_highlight_person(data):
    face_id = data.get('face_id')
    camera_id = data.get('camera_id')
    for pid in list(processors.keys()):
        if pid in processors:
            processors[pid].selected_person_id = face_id
            processors[pid].selected_camera_id = camera_id

@socketio.on('request_person_frame')
def handle_request_person_frame(data):
    camera_id = data.get('camera_id')
    if camera_id in processors and not processors[camera_id].paused:
        processors[camera_id].process_single_frame()

if __name__ == "__main__":
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
