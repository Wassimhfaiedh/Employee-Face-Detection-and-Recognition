import cv2
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
import json
import os
import time
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from flask_socketio import SocketIO, emit
import base64
from datetime import datetime, timedelta
from collections import deque
import logging
import threading
import torch

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'security-ox4labs-secret'
socketio = SocketIO(app, async_mode='threading')

# Ensure videos directory exists
VIDEOS_FOLDER = 'videos'
if not os.path.exists(VIDEOS_FOLDER):
    os.makedirs(VIDEOS_FOLDER)
app.config['VIDEOS_FOLDER'] = VIDEOS_FOLDER

# Available video streams
AVAILABLE_VIDEOS = {
    'vehicle_1': {'name': 'Vehicle Cam A', 'path': 'videos/v1.mp4', 'type': 'vehicle'},
    'vehicle_2': {'name': 'Vehicle Cam B', 'path': 'videos/v2.mp4', 'type': 'vehicle'},
    'face_1': {'name': 'Face Cam A', 'path': 'videos/Generate_a_realistic_202508171102_f1rci.mp4', 'type': 'face'},
    'face_2': {'name': 'Face Cam B', 'path': 'videos/istockphoto-1532931548-640_adpp_is.mp4', 'type': 'face'},
    'zone_1': {'name': 'Zone Cam A', 'path': 'videos/istockphoto-2148384684-640_adpp_is.mp4', 'type': 'zone'},
    'zone_2': {'name': 'Zone Cam B', 'path': 'videos/istockphoto-2148384684-640_adpp_is.mp4', 'type': 'zone'}
}

# Store current configuration
current_config = []

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

class VehicleProcessor:
    def __init__(self, video_path, processor_id, camera_name):
        self.video_path = video_path
        self.processor_id = processor_id
        self.camera_name = camera_name
        self.running = False
        self.counting = False
        self.paused = False
        self._processing_task = None
        self.cap = cv2.VideoCapture(video_path)
        self.frame_width = 960
        self.frame_height = 540
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.wait_time = int(1000 / self.fps)
        self.current_frame = None
        self.current_pos = 0
        self.preserved_data = {'in_count': 0, 'out_count': 0, 'vehicle_events': []}

        if not self.cap.isOpened():
            logging.error(f"Could not open video {video_path} for processor {processor_id}")
            self.cap = None
            socketio.emit(f'update_status_{self.processor_id}', {'message': f'{self.camera_name} - Error: Could not open video'})
            return

        ret, self.first_frame = self.cap.read()
        if not ret:
            logging.error(f"Could not read first frame from {video_path} for processor {processor_id}")
            self.cap = None
            socketio.emit(f'update_status_{self.processor_id}', {'message': f'{self.camera_name} - Error: Could not read first frame'})
            return

        self.first_frame = cv2.resize(self.first_frame, (self.frame_width, self.frame_height))
        socketio.emit(f'frame_dimensions_{self.processor_id}', {
            'width': self.frame_width,
            'height': self.frame_height
        })
        frame_data = self.encode_frame(self.first_frame)
        socketio.emit(f'video_frame_{self.processor_id}', {'frame': frame_data})

        # License Plate Detection Setup
        self.line_points = []
        self.line_completed = False
        self.in_count = 0
        self.out_count = 0
        self.crossed = {}
        self.captured_info = {}
        self.vehicle_events = []
        self.recordings = {}
        self.frame_buffer = deque(maxlen=60)
        self.car_counter = 0
        self.current_plate_ids = set()
        self.high_conf_threshold = 0.70
        try:
            self.plate_model = YOLO("models/license_plate_detector.pt").to(device)
            self.car_model = YOLO("models/yolov8n.pt").to(device)
            logging.info(f"YOLO models loaded on {device} for processor {processor_id}")
        except Exception as e:
            logging.error(f"Failed to load YOLO models for processor {processor_id}: {e}")
            socketio.emit(f'update_status_{self.processor_id}', {'message': f'{self.camera_name} - Error: Failed to load YOLO models - {str(e)}'})
            self.plate_model = None
            self.car_model = None
            return
        self.class_names = {2: "Car", 5: "Bus", 7: "Truck"}
        self.detect_classes = [2, 5, 7]

    def encode_frame(self, frame):
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return base64.b64encode(buffer).decode('utf-8')

    def encode_image(self, img):
        _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return base64.b64encode(buffer).decode('utf-8')

    def get_side(self, center):
        (x1, y1), (x2, y2) = self.line_points
        return (center[0] - x1) * (y2 - y1) - (center[1] - y1) * (x2 - x1)

    def process_frame(self, frame):
        if frame is None or self.plate_model is None or self.car_model is None:
            return None
        display_frame = frame.copy()

        for p in self.line_points:
            cv2.circle(display_frame, p, 5, (0, 0, 255), -1)
        if len(self.line_points) == 2:
            cv2.line(display_frame, self.line_points[0], self.line_points[1], (0, 0, 255), 2)

        if self.counting:
            plate_results = self.plate_model.track(frame, conf=0.3, persist=True, device=device)
            car_results = self.car_model.track(frame, conf=0.5, persist=True, device=device)
            self.current_plate_ids = set()

            for r in car_results[0].boxes:
                if int(r.cls) not in self.detect_classes:
                    continue
                cx1, cy1, cx2, cy2 = map(int, r.xyxy[0])
                cv2.rectangle(display_frame, (cx1, cy1), (cx2, cy2), (255, 0, 0), 2)

            for r in plate_results[0].boxes:
                x1, y1, x2, y2 = map(int, r.xyxy[0])
                plate_id = int(r.id) if r.id is not None else None
                if plate_id is None:
                    continue
                self.current_plate_ids.add(plate_id)
                conf = r.conf.item()
                has_crossed = plate_id in self.crossed and self.captured_info.get(plate_id, {}).get('mode') in ["In", "Out"]
                bbox_color = (0, 255, 0) if has_crossed else (0, 0, 255)
                shadow_color = (0, 0, 0)
                cv2.rectangle(display_frame, (x1+2, y1+2), (x2+2, y2+2), shadow_color, 2)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), bbox_color, 2)
                cv2.putText(display_frame, "License Plate", (x1+1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, shadow_color, 2)
                cv2.putText(display_frame, "License Plate", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 2)

                if plate_id not in self.captured_info:
                    self.captured_info[plate_id] = {
                        'last_time': 0,
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'detected_text': "",
                        'appear_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'disappear_time': None,
                        'mode': "Detected",
                        'cross_time': None,
                        'plate_img': "",
                        'car_img': "",
                        'video_path': "",
                        'video_duration': "4 seconds",
                        'best_conf': 0.0,
                        'event_id': None,
                        'camera': self.camera_name
                    }

                self.captured_info[plate_id]['x1'] = x1
                self.captured_info[plate_id]['y1'] = y1
                self.captured_info[plate_id]['x2'] = x2
                self.captured_info[plate_id]['y2'] = y2

                best_conf = self.captured_info[plate_id]['best_conf']
                now = time.time()
                last_time = self.captured_info[plate_id]['last_time']

                if conf > best_conf and conf >= self.high_conf_threshold and now - last_time > 1:
                    plate_img = frame[y1:y2, x1:x2]
                    if plate_img.size == 0:
                        continue
                    upscale_factor = 5
                    plate_upscaled = cv2.resize(plate_img, (plate_img.shape[1]*upscale_factor, plate_img.shape[0]*upscale_factor), interpolation=cv2.INTER_CUBIC)
                    detected_text = f"Plate {plate_id}"
                    self.captured_info[plate_id]['detected_text'] = detected_text
                    plate_base64 = self.encode_image(plate_upscaled)
                    car_base64 = ""
                    car_box = None
                    for cr in car_results[0].boxes:
                        if int(cr.cls) not in self.detect_classes:
                            continue
                        cx1, cy1, cx2, cy2 = map(int, cr.xyxy[0])
                        if x1 > cx1 and y1 > cy1 and x2 < cx2 and y2 < cy2:
                            car_box = (cx1, cy1, cx2, cy2)
                            break
                    if car_box:
                        cx1, cy1, cx2, cy2 = car_box
                        car_crop = frame[cy1:cy2, cx1:cx2]
                        car_base64 = self.encode_image(car_crop)
                        cv2.rectangle(display_frame, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
                        cv2.putText(display_frame, self.captured_info[plate_id].get('car_name', f'car{self.car_counter+1}'), (cx1, cy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    self.captured_info[plate_id]['plate_img'] = plate_base64
                    self.captured_info[plate_id]['car_img'] = car_base64
                    self.captured_info[plate_id]['last_time'] = now
                    self.captured_info[plate_id]['best_conf'] = conf

                    # Only create and append event if it hasn't been added yet
                    if self.captured_info[plate_id]['event_id'] is None:
                        self.car_counter += 1
                        car_name = f"car{self.car_counter}"
                        self.captured_info[plate_id]['car_name'] = car_name
                        event = self.captured_info[plate_id].copy()
                        self.vehicle_events.append(event)
                        self.preserved_data['vehicle_events'].append(event)
                        event_id = len(self.vehicle_events) - 1
                        self.captured_info[plate_id]['event_id'] = event_id
                        self.recordings[event_id] = list(self.frame_buffer)

                    event_id = self.captured_info[plate_id]['event_id']
                    self.vehicle_events[event_id]['plate_img'] = plate_base64
                    self.vehicle_events[event_id]['car_img'] = car_base64
                    self.preserved_data['vehicle_events'][event_id]['plate_img'] = plate_base64
                    self.preserved_data['vehicle_events'][event_id]['car_img'] = car_base64
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    mode = self.captured_info[plate_id]['mode']
                    self.captured_info[plate_id]['cross_time'] = timestamp
                    self.vehicle_events[event_id]['mode'] = mode
                    self.vehicle_events[event_id]['cross_time'] = timestamp
                    self.preserved_data['vehicle_events'][event_id]['mode'] = mode
                    self.preserved_data['vehicle_events'][event_id]['cross_time'] = timestamp
                    if event_id in self.recordings:
                        self.finalize_recording(event_id)
                    self.recordings[event_id] = list(self.frame_buffer)

            if self.line_completed:
                for plate_id in self.current_plate_ids:
                    if plate_id in self.captured_info:
                        info = self.captured_info[plate_id]
                        x1, y1, x2, y2 = info['x1'], info['y1'], info['x2'], info['y2']
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        side = self.get_side(center)
                        if plate_id not in self.crossed:
                            self.crossed[plate_id] = side
                        else:
                            last_side = self.crossed[plate_id]
                            if last_side * side < 0 and info['mode'] == "Detected":
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                mode = "In" if side > 0 else "Out"
                                info['mode'] = mode
                                info['cross_time'] = timestamp
                                if mode == "In":
                                    self.in_count += 1
                                    self.preserved_data['in_count'] += 1
                                else:
                                    self.out_count += 1
                                    self.preserved_data['out_count'] += 1
                                if info['event_id'] is not None:
                                    self.vehicle_events[info['event_id']]['mode'] = mode
                                    self.vehicle_events[info['event_id']]['cross_time'] = timestamp
                                    self.preserved_data['vehicle_events'][info['event_id']]['mode'] = mode
                                    self.preserved_data['vehicle_events'][info['event_id']]['cross_time'] = timestamp
                        self.crossed[plate_id] = side

            to_remove = [pid for pid in self.captured_info if pid not in self.current_plate_ids]
            for pid in to_remove:
                info = self.captured_info[pid]
                info['disappear_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if info['event_id'] is not None and info['event_id'] in self.recordings:
                    self.finalize_recording(info['event_id'])
                del self.captured_info[pid]
                if pid in self.crossed:
                    del self.crossed[pid]
                if pid in self.recordings:
                    del self.recordings[pid]

            # Filter vehicle events based on mode
            modes = ["In", "Out"] if self.line_completed else ["Detected"]
            filtered_vehicle_events = [
                {
                    'matricule': e['detected_text'],
                    'mode': e['mode'],
                    'cross_time': e['cross_time'],
                    'plate_img': e.get('plate_img', ''),
                    'car_img': e.get('car_img', ''),
                    'video_path': e.get('video_path', ''),
                    'video_duration': e.get('video_duration', '4 seconds'),
                    'id': i,
                    'camera': e.get('camera', self.camera_name)
                }
                for i, e in enumerate(self.vehicle_events)
                if e['mode'] in modes and e['cross_time'] is not None and e['best_conf'] >= self.high_conf_threshold
            ]

            socketio.emit(f'update_vehicle_counts_{self.processor_id}', {
                'in_count': self.in_count,
                'out_count': self.out_count,
                'vehicle_events': filtered_vehicle_events
            })

        return display_frame

    def finalize_recording(self, event_id):
        if event_id not in self.recordings:
            return
        frames = self.recordings[event_id]
        if len(frames) == 0:
            del self.recordings[event_id]
            return
        filename = f'event_{event_id}_zone_{self.processor_id}.mp4'
        path = os.path.join(app.config['VIDEOS_FOLDER'], filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 15.0
        writer = cv2.VideoWriter(path, fourcc, fps, (self.frame_width, self.frame_height))
        for f in frames:
            writer.write(f)
        writer.release()
        duration = len(frames) / fps
        self.vehicle_events[event_id]['video_path'] = filename
        self.vehicle_events[event_id]['video_duration'] = f"{duration:.1f} seconds"
        self.preserved_data['vehicle_events'][event_id]['video_path'] = filename
        self.preserved_data['vehicle_events'][event_id]['video_duration'] = f"{duration:.1f} seconds"
        logging.debug(f"Video saved: {path}")
        del self.recordings[event_id]

    def add_line_point(self, x, y):
        x = max(0, min(int(x), self.frame_width - 1))
        y = max(0, min(int(y), self.frame_height - 1))
        if len(self.line_points) < 2:
            self.line_points.append((x, y))
            socketio.emit(f'update_status_{self.processor_id}', {'message': f'{self.camera_name} - Point added at ({x}, {y})'})
        if self.running and self.current_frame is not None:
            frame = self.current_frame.copy()
        else:
            frame = self.first_frame.copy() if self.first_frame is not None else None
        processed_frame = self.process_frame(frame)
        if processed_frame is not None:
            frame_data = self.encode_frame(processed_frame)
            socketio.emit(f'video_frame_{self.processor_id}', {'frame': frame_data})
        if len(self.line_points) == 2:
            self.complete_line()

    def complete_line(self):
        if len(self.line_points) not in [0, 2]:
            socketio.emit(f'update_status_{self.processor_id}', {'message': f'{self.camera_name} - Invalid number of points. Add exactly 2 points for line or delete to start without.'})
            return
        self.line_completed = len(self.line_points) == 2
        mode_text = "counting" if self.line_completed else "detection"
        socketio.emit(f'update_status_{self.processor_id}', {'message': f'{self.camera_name} - System {mode_text} mode activated'})
        if not self.running:
            self.start()

    def delete_line(self):
        self.line_points = []
        self.line_completed = False
        socketio.emit(f'update_status_{self.processor_id}', {'message': f'{self.camera_name} - Line deleted. Switched to detection mode'})
        if self.running and self.current_frame is not None:
            frame = self.current_frame.copy()
        else:
            frame = self.first_frame.copy() if self.first_frame is not None else None
        processed_frame = self.process_frame(frame)
        if processed_frame is not None:
            frame_data = self.encode_frame(processed_frame)
            socketio.emit(f'video_frame_{self.processor_id}', {'frame': frame_data})

    def start(self):
        if self._processing_task is not None:
            return
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            socketio.emit(f'update_status_{self.processor_id}', {'message': f'{self.camera_name} - Error: Could not reopen video'})
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_pos)
        self.counting = True
        self.running = True
        self.paused = False
        socketio.emit(f'update_status_{self.processor_id}', {'message': f'{self.camera_name} - System started'})
        self._processing_task = socketio.start_background_task(self.process_video)

    def stop(self):
        self.running = False
        self.counting = False
        self.paused = False
        self.in_count = 0
        self.out_count = 0
        self.crossed = {}
        self.captured_info = {}
        self.vehicle_events = self.preserved_data['vehicle_events'].copy()
        self.recordings = {}
        self.frame_buffer.clear()
        self.car_counter = len(self.vehicle_events)
        self.current_plate_ids = set()
        self.current_pos = 0
        if self.cap:
            self.cap.release()
            self.cap = None
        socketio.emit(f'update_status_{self.processor_id}', {'message': f'{self.camera_name} - System stopped'})
        frame = self.first_frame.copy() if self.first_frame is not None else None
        processed_frame = self.process_frame(frame)
        if processed_frame is not None:
            frame_data = self.encode_frame(processed_frame)
            socketio.emit(f'video_frame_{self.processor_id}', {'frame': frame_data})

    def clear_dashboard(self):
        self.preserved_data = {'in_count': 0, 'out_count': 0, 'vehicle_events': []}
        self.in_count = 0
        self.out_count = 0
        self.vehicle_events = []
        self.crossed = {}
        self.captured_info = {}
        self.recordings = {}
        self.frame_buffer.clear()
        self.car_counter = 0
        self.current_plate_ids = set()
        socketio.emit(f'update_vehicle_counts_{self.processor_id}', {
            'in_count': self.in_count,
            'out_count': self.out_count,
            'vehicle_events': []
        })

    def toggle_pause_resume(self):
        if self.paused:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                socketio.emit(f'update_status_{self.processor_id}', {'message': f'{self.camera_name} - Error: Could not reopen video'})
                return
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_pos)
            self.paused = False
            socketio.emit(f'update_status_{self.processor_id}', {'message': f'{self.camera_name} - System resumed'})
        else:
            self.paused = True
            self.current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            if self.cap:
                self.cap.release()
                self.cap = None
            socketio.emit(f'update_status_{self.processor_id}', {'message': f'{self.camera_name} - System paused'})

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
                    self.current_pos = 0
                    continue
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                self.frame_buffer.append(frame.copy())
                self.current_frame = frame.copy()
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
        socketio.emit(f'update_status_{self.processor_id}', {'message': f'{self.camera_name} - Video processing stopped'})

    def get_vehicle_details(self, event_id):
        if event_id < 0 or event_id >= len(self.vehicle_events):
            return None
        event = self.vehicle_events[event_id]
        return {
            'matricule': event['detected_text'],
            'mode': event['mode'],
            'cross_time': event['cross_time'],
            'car_img': event.get('car_img', ''),
            'plate_img': event.get('plate_img', ''),
            'video_path': event.get('video_path', ''),
            'video_duration': event.get('video_duration', '4 seconds'),
            'camera': event.get('camera', self.camera_name)
        }

class FaceProcessor:
    def __init__(self, video_path, processor_id, camera_name):
        self.video_path = video_path
        self.processor_id = processor_id
        self.camera_name = camera_name
        self.running = False
        self.paused = False
        self._processing_task = None
        self.cap = cv2.VideoCapture(video_path)
        self.frame_width = 960
        self.frame_height = 540
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.wait_time = int(1000 / self.fps)
        self.current_frame = None
        self.current_pos = 0
        self.preserved_data = {'known_count': 0, 'visitor_count': 0, 'detections': []}
        self.selected_face_id = None

        if not self.cap.isOpened():
            logging.error(f"Could not open video {video_path} for processor {processor_id}")
            self.cap = None
            socketio.emit(f'update_status_{self.processor_id}', {'message': f'{self.camera_name} - Error: Could not open video'})
            return

        ret, self.first_frame = self.cap.read()
        if not ret:
            logging.error(f"Could not read first frame from {video_path} for processor {processor_id}")
            self.cap = None
            socketio.emit(f'update_status_{self.processor_id}', {'message': f'{self.camera_name} - Error: Could not read first frame'})
            return

        self.first_frame = cv2.resize(self.first_frame, (self.frame_width, self.frame_height))
        socketio.emit(f'frame_dimensions_{self.processor_id}', {
            'width': self.frame_width,
            'height': self.frame_height
        })
        frame_data = self.encode_frame(self.first_frame)
        socketio.emit(f'video_frame_{self.processor_id}', {'frame': frame_data})

        # Face Detection Setup
        try:
            provider = ['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            self.recognizer = FaceAnalysis(name='buffalo_l', providers=provider)
            self.recognizer.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)
            logging.info(f"FaceAnalysis initialized on {provider[0]} for processor {processor_id}")
        except Exception as e:
            logging.error(f"Failed to initialize FaceAnalysis for processor {processor_id}: {e}")
            socketio.emit(f'update_status_{self.processor_id}', {'message': f'{self.camera_name} - Error: FaceAnalysis initialization failed - {str(e)}'})
            self.recognizer = None
            return

        self.load_embeddings("face_embeddings_corrected.json")
        self.face_tracker = {}
        self.face_id_counter = 0
        self.locked_faces = {}
        self.detections = []
        self.pending_visitors = {}
        self.VISITOR_CONFIRMATION_SECONDS = 6
        self.selected_person_id = None
        self.selected_camera_id = None
        self.known_count = 0
        self.visitor_count = 0

    def load_embeddings(self, path):
        try:
            if not os.path.exists(path):
                logging.warning(f"Embedding file {path} not found. Initializing empty embeddings.")
                self.names = []
                self.vectors = np.array([])
                return
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
            if torch.cuda.is_available():
                self.vectors = torch.tensor(self.vectors, device=device, dtype=torch.float32)
            logging.info(f"Loaded {len(self.names)} embeddings from {path}")
        except Exception as e:
            logging.error(f"Error loading embeddings: {e}")
            self.names = []
            self.vectors = np.array([])
            socketio.emit(f'update_status_{self.processor_id}', {'message': f'{self.camera_name} - Error: Failed to load embeddings - {str(e)}'})

    def normalize_embedding(self, embedding):
        if torch.is_tensor(embedding):
            norm = torch.norm(embedding)
            return embedding / norm if norm > 0 else embedding
        else:
            norm = np.linalg.norm(embedding)
            return embedding / norm if norm > 0 else embedding

    def simple_recognition(self, embedding):
        if len(self.vectors) == 0:
            return "Visitor", 0, 1
        embedding = self.normalize_embedding(embedding)
        if torch.is_tensor(self.vectors):
            embedding = torch.tensor(embedding, device=device, dtype=torch.float32)
            similarities = torch.cosine_similarity(embedding.unsqueeze(0), self.vectors)
            max_similarity, max_index = torch.max(similarities, dim=0)
            max_similarity = max_similarity.item()
            max_index = max_index.item()
        else:
            similarities = [1 - cosine(embedding, db_emb) for db_emb in self.vectors]
            max_similarity = max(similarities)
            max_index = similarities.index(max_similarity)
        adaptive_threshold = 0.6 if max_similarity > 0.8 else 0.35
        name = self.names[max_index] if max_similarity > adaptive_threshold else "Visitor"
        distance = 1 - max_similarity
        return name, max_similarity, distance

    def compute_iou(self, bbox1, bbox2):
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

    def draw_label(self, frame, text, x1, y1, color):
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
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return base64.b64encode(buffer).decode('utf-8')

    def encode_image(self, img):
        _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return base64.b64encode(buffer).decode('utf-8')

    def process_frame(self, frame):
        if frame is None or self.recognizer is None:
            logging.error(f"Invalid frame or recognizer for processor {self.processor_id}")
            return None
        display_frame = frame.copy()
        try:
            faces = self.recognizer.get(frame)
        except Exception as e:
            logging.error(f"Face detection failed for processor {self.processor_id}: {e}")
            socketio.emit(f'update_status_{self.processor_id}', {'message': f'{self.camera_name} - Error: Face detection failed - {str(e)}'})
            return display_frame

        current_faces = {}
        known_count = 0
        visitor_count = 0
        current_time = datetime.now()

        for i, face in enumerate(faces):
            try:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                matched_id = None
                max_iou = 0
                for fid, (name, conf, prev_bbox, _) in self.face_tracker.items():
                    iou = self.compute_iou(bbox, prev_bbox)
                    if iou > max_iou and iou > 0.5:
                        max_iou = iou
                        matched_id = fid
                name, confidence, distance = self.simple_recognition(face.embedding)
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
                if self.selected_face_id is not None:
                    if matched_id != self.selected_face_id:
                        continue
                    box_color = (255, 0, 0)  # Blue in BGR
                else:
                    box_color = (255, 0, 0) if matched_id == self.selected_person_id and self.processor_id == self.selected_camera_id else (0, 200, 0) if name != "Visitor" else (0, 0, 200)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 1)
                label = f"{name} ({confidence:.2f})"
                self.draw_label(display_frame, label, x1, y1, box_color)

                if name != "Visitor":
                    self.detections = [d for d in self.detections if d['face_id'] != matched_id or d['name'] != "Visitor"]
                    if matched_id in self.pending_visitors:
                        del self.pending_visitors[matched_id]
                    if not any(d['name'] == name for d in self.detections):
                        face_img = frame[max(0, y1-20):min(frame.shape[0], y2+20), max(0, x1-20):min(frame.shape[1], x2+20)]
                        if face_img.size > 0:
                            face_base64 = self.encode_image(face_img)
                            detection = {
                                'name': name,
                                'confidence': round(confidence, 2),
                                'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                                'face_image': face_base64,
                                'face_id': matched_id,
                                'camera': self.camera_name
                            }
                            self.detections.append(detection)
                            self.preserved_data['detections'].append(detection)
                            self.detections.sort(key=lambda x: x['timestamp'])
                            self.preserved_data['detections'].sort(key=lambda x: x['timestamp'])
                            if len(self.detections) > 10:
                                self.detections.pop(0)
                            if len(self.preserved_data['detections']) > 10:
                                self.preserved_data['detections'].pop(0)
                else:
                    if matched_id not in self.locked_faces or self.locked_faces[matched_id][0] == "Visitor":
                        if matched_id in self.pending_visitors:
                            self.pending_visitors[matched_id]['timestamp'] = current_time.strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            face_img = frame[max(0, y1-20):min(frame.shape[0], y2+20), max(0, x1-20):min(frame.shape[1], x2+20)]
                            if face_img.size > 0:
                                face_base64 = self.encode_image(face_img)
                                self.pending_visitors[matched_id] = {
                                    'name': name,
                                    'confidence': round(confidence, 2),
                                    'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                                    'face_image': face_base64,
                                    'face_id': matched_id,
                                    'initial_time': current_time,
                                    'camera': self.camera_name
                                }
                        if (current_time - self.pending_visitors[matched_id]['initial_time']).total_seconds() >= self.VISITOR_CONFIRMATION_SECONDS:
                            if not any(d['name'] == "Visitor" for d in self.detections):
                                detection = {
                                    'name': self.pending_visitors[matched_id]['name'],
                                    'confidence': self.pending_visitors[matched_id]['confidence'],
                                    'timestamp': self.pending_visitors[matched_id]['timestamp'],
                                    'face_image': self.pending_visitors[matched_id]['face_image'],
                                    'face_id': self.pending_visitors[matched_id]['face_id'],
                                    'camera': self.pending_visitors[matched_id]['camera']
                                }
                                self.detections.append(detection)
                                self.preserved_data['detections'].append(detection)
                                self.detections.sort(key=lambda x: x['timestamp'])
                                self.preserved_data['detections'].sort(key=lambda x: x['timestamp'])
                                if len(self.detections) > 10:
                                    self.detections.pop(0)
                                if len(self.preserved_data['detections']) > 10:
                                    self.preserved_data['detections'].pop(0)
                            del self.pending_visitors[matched_id]
                if name == "Visitor":
                    visitor_count += 1
                else:
                    known_count += 1
            except Exception as e:
                logging.error(f"Error processing face {i} for processor {self.processor_id}: {e}")
                continue

        self.detections = [d for d in self.detections if d['face_id'] in current_faces]
        self.pending_visitors = {k: v for k, v in self.pending_visitors.items() if k in current_faces}
        self.face_tracker = current_faces
        self.known_count = known_count
        self.visitor_count = visitor_count
        self.preserved_data['known_count'] = known_count
        self.preserved_data['visitor_count'] = visitor_count
        socketio.emit(f'update_face_detections_{self.processor_id}', {'detections': self.detections})
        socketio.emit(f'update_face_analytics_{self.processor_id}', {
            'known_count': known_count,
            'visitor_count': visitor_count
        })
        return display_frame

    def start(self):
        if self._processing_task is not None:
            logging.info(f"Processor {self.processor_id} already running")
            return
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            logging.error(f"Could not reopen video {self.video_path} for processor {self.processor_id}")
            socketio.emit(f'update_status_{self.processor_id}', {'message': f'{self.camera_name} - Error: Could not reopen video'})
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_pos)
        if self.recognizer is None:
            logging.error(f"Face recognizer not initialized for processor {self.processor_id}")
            socketio.emit(f'update_status_{self.processor_id}', {'message': f'{self.camera_name} - Error: Face recognizer not initialized'})
            return
        self.running = True
        self.paused = False
        socketio.emit(f'update_status_{self.processor_id}', {'message': f'{self.camera_name} - System started'})
        self._processing_task = socketio.start_background_task(self.process_video)

    def stop(self):
        self.running = False
        self.paused = False
        self.face_tracker = {}
        self.face_id_counter = 0
        self.locked_faces = {}
        self.detections = self.preserved_data['detections'].copy()
        self.pending_visitors = {}
        self.selected_person_id = None
        self.selected_camera_id = None
        self.known_count = self.preserved_data['known_count']
        self.visitor_count = self.preserved_data['visitor_count']
        self.current_pos = 0
        self.selected_face_id = None
        if self.cap:
            self.cap.release()
            self.cap = None
        socketio.emit(f'update_status_{self.processor_id}', {'message': f'{self.camera_name} - System stopped'})
        frame = self.first_frame.copy() if self.first_frame is not None else None
        processed_frame = self.process_frame(frame)
        if processed_frame is not None:
            frame_data = self.encode_frame(processed_frame)
            socketio.emit(f'video_frame_{self.processor_id}', {'frame': frame_data})
        socketio.emit(f'update_face_detections_{self.processor_id}', {'detections': self.detections})
        socketio.emit(f'update_face_analytics_{self.processor_id}', {'known_count': self.known_count, 'visitor_count': self.visitor_count})

    def clear_dashboard(self):
        self.preserved_data = {'known_count': 0, 'visitor_count': 0, 'detections': []}
        self.face_tracker = {}
        self.face_id_counter = 0
        self.locked_faces = {}
        self.detections = []
        self.pending_visitors = {}
        self.selected_person_id = None
        self.selected_camera_id = None
        self.known_count = 0
        self.visitor_count = 0
        self.selected_face_id = None
        socketio.emit(f'update_face_detections_{self.processor_id}', {'detections': []})
        socketio.emit(f'update_face_analytics_{self.processor_id}', {'known_count': 0, 'visitor_count': 0})

    def toggle_pause_resume(self):
        if self.paused:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                socketio.emit(f'update_status_{self.processor_id}', {'message': f'{self.camera_name} - Error: Could not reopen video'})
                return
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_pos)
            self.paused = False
            socketio.emit(f'update_status_{self.processor_id}', {'message': f'{self.camera_name} - System resumed'})
        else:
            self.paused = True
            self.current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            if self.cap:
                self.cap.release()
                self.cap = None
            socketio.emit(f'update_status_{self.processor_id}', {'message': f'{self.camera_name} - System paused'})
            self.detections = []
            self.pending_visitors = {}
            self.selected_person_id = None
            self.selected_camera_id = None
            self.selected_face_id = None
            socketio.emit(f'update_face_detections_{self.processor_id}', {'detections': []})
            socketio.emit(f'update_face_analytics_{self.processor_id}', {'known_count': 0, 'visitor_count': 0})

    def process_video(self):
        self.running = True
        while self.running and self.cap and self.cap.isOpened():
            if self.paused:
                socketio.sleep(0.1)
                continue
            try:
                ret, frame = self.cap.read()
                if not ret:
                    logging.warning(f"Failed to read frame from {self.video_path}. Looping video.")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.current_pos = 0
                    continue
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                self.current_frame = frame.copy()
                processed_frame = self.process_frame(frame)
                if processed_frame is not None:
                    frame_data = self.encode_frame(processed_frame)
                    socketio.emit(f'video_frame_{self.processor_id}', {'frame': frame_data})
                socketio.sleep(self.wait_time / 1000.0)
            except Exception as e:
                logging.error(f"Error in video processing loop for processor {self.processor_id}: {e}")
                socketio.emit(f'update_status_{self.processor_id}', {'message': f'{self.camera_name} - Error: Video processing failed - {str(e)}'})
                break
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self._processing_task = None
        socketio.emit(f'update_status_{self.processor_id}', {'message': f'{self.camera_name} - Video processing stopped'})

class ZoneProcessor:
    def __init__(self, video_path, processor_id, camera_name):
        self.video_path = video_path
        self.processor_id = processor_id
        self.camera_name = camera_name
        self.model = YOLO('models/zone model.pt').to(device)  # Load your custom YOLO model
        self.cap = cv2.VideoCapture(video_path)
        self.running = False
        self.counting = False
        self.paused = False
        self._processing_task = None
        self.frame_width = 960
        self.frame_height = 540
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.wait_time = int(1000 / self.fps)
        self.current_frame = None
        self.current_pos = 0
        self.zone_id_map = {}
        self.next_zone_id = 1
        self.class_colors = {
            "Empty_zone": (0, 255, 0),      # Green
            "Partial_zone": (0, 255, 255),  # Yellow
            "Full_zone": (0, 0, 255),       # Red
        }
        self.preserved_data = {'zones': {}}

        if not self.cap.isOpened():
            logging.error(f"Could not open video {video_path} for processor {processor_id}")
            self.cap = None
            socketio.emit(f'update_status_{self.processor_id}', {'message': f'{self.camera_name} - Error: Could not open video'})
            return

        ret, self.first_frame = self.cap.read()
        if not ret:
            logging.error(f"Could not read first frame from {video_path} for processor {processor_id}")
            self.cap = None
            socketio.emit(f'update_status_{self.processor_id}', {'message': f'{self.camera_name} - Error: Could not read first frame'})
            return

        self.first_frame = cv2.resize(self.first_frame, (self.frame_width, self.frame_height))
        socketio.emit(f'frame_dimensions_{self.processor_id}', {
            'width': self.frame_width,
            'height': self.frame_height
        })
        frame_data = self.encode_frame(self.first_frame)
        socketio.emit(f'video_frame_{self.processor_id}', {'frame': frame_data})

    def draw_dashed_rect(self, img, pt1, pt2, color, thickness=2, dash_length=5):
        x1, y1 = pt1
        x2, y2 = pt2
        for i in range(x1, x2, dash_length * 2):
            cv2.line(img, (i, y1), (min(i + dash_length, x2), y1), color, thickness)
            cv2.line(img, (i, y2), (min(i + dash_length, x2), y2), color, thickness)
        for i in range(y1, y2, dash_length * 2):
            cv2.line(img, (x1, i), (x1, min(i + dash_length, y2)), color, thickness)
            cv2.line(img, (x2, i), (x2, min(i + dash_length, y2)), color, thickness)

    def compute_partial_percent(self, zone_xyxy, stock_boxes_xyxy):
        x1, y1, x2, y2 = zone_xyxy
        w, h = max(0, x2 - x1), max(0, y2 - y1)
        if w == 0 or h == 0:
            return 0

        zone_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(zone_mask, (0, 0), (w, h), 255, -1)

        stock_mask = np.zeros((h, w), dtype=np.uint8)
        stock_count = 0

        for sx1, sy1, sx2, sy2 in stock_boxes_xyxy:
            ox1 = max(x1, sx1); oy1 = max(y1, sy1)
            ox2 = min(x2, sx2); oy2 = min(y2, sy2)
            if ox1 < ox2 and oy1 < oy2:
                p1 = (ox1 - x1, oy1 - y1)
                p2 = (ox2 - x1, oy2 - y1)
                cv2.rectangle(stock_mask, p1, p2, 255, -1)
                stock_count += 1

        intersection = cv2.bitwise_and(zone_mask, stock_mask)
        zone_area = np.count_nonzero(zone_mask)
        stock_area = np.count_nonzero(intersection)

        if stock_count > 0:
            target_percent_per_stock = 10.0
            percent = min(stock_count * target_percent_per_stock, 100.0)
        else:
            percent = 0.0

        return int(round(percent))

    def assign_zone_id(self, x1, y1, x2, y2):
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        center = (center_x, center_y)

        for known_center, zone_id in self.zone_id_map.items():
            dist = np.sqrt((center_x - known_center[0])**2 + (center_y - known_center[1])**2)
            if dist < 50:
                return zone_id

        zone_id = self.next_zone_id
        self.zone_id_map[center] = zone_id
        self.next_zone_id += 1
        return zone_id

    def encode_frame(self, frame):
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return base64.b64encode(buffer).decode('utf-8')

    def process_frame(self, frame):
        if frame is None:
            return None

        display_frame = frame.copy()
        results = self.model.predict(frame, conf=0.9, iou=0.5)
        boxes = results[0].boxes
        names = results[0].names

        overlay = frame.copy()
        zone_boxes = []
        stock_boxes = []
        zone_data = {}

        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                label = names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if label.lower() == "stock":
                    stock_boxes.append((x1, y1, x2, y2))
                    continue
                else:
                    zone_id = self.assign_zone_id(x1, y1, x2, y2)
                    zone_boxes.append((label, (x1, y1, x2, y2), zone_id))

                    zone_color = self.class_colors.get(label, (255, 255, 255))
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), zone_color, -1)
                    self.draw_dashed_rect(display_frame, (x1, y1), (x2, y2), (0, 0, 0), 2)

        alpha = 0.3
        display_frame = cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0)

        for label, (x1, y1, x2, y2), zone_id in zone_boxes:
            zone_color = self.class_colors.get(label, (255, 255, 255))

            circle_center = (x1 + 20, y1 + 20)
            circle_radius = 15
            cv2.circle(display_frame, circle_center, circle_radius, (0, 0, 0), -1)
            cv2.circle(display_frame, circle_center, circle_radius, zone_color, 2)

            zone_text = str(zone_id)
            (text_w, text_h), _ = cv2.getTextSize(zone_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_pos = (circle_center[0] - text_w // 2, circle_center[1] + text_h // 2)
            cv2.putText(display_frame, zone_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone_color, 1, cv2.LINE_AA)

            if label == "Full_zone":
                percent = 100
            elif label == "Empty_zone":
                percent = 0
            elif label == "Partial_zone":
                percent = self.compute_partial_percent((x1, y1, x2, y2), stock_boxes)
            else:
                percent = 0

            zone_data[zone_id] = {'label': label, 'percent': percent, 'camera': self.camera_name}
            self.preserved_data['zones'][zone_id] = zone_data[zone_id]

            label_text = label
            percent_text = f"{percent}% STOCK"

            label_font_scale = 0.6
            percent_font_scale = 0.4
            label_thickness = 2
            percent_thickness = 1
            padding = 6

            (label_tw, label_th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, label_thickness)
            (percent_tw, percent_th), _ = cv2.getTextSize(percent_text, cv2.FONT_HERSHEY_SIMPLEX, percent_font_scale, percent_thickness)

            bbox_cx = (x1 + x2) // 2
            bbox_cy = (y1 + y2) // 2

            label_x1 = bbox_cx - (label_tw + padding*2) // 2
            label_y1 = bbox_cy - (label_th + padding*2 + percent_th + padding*2) // 2
            label_x2 = label_x1 + label_tw + padding*2
            label_y2 = label_y1 + label_th + padding*2

            percent_x1 = bbox_cx - (percent_tw + padding*2) // 2
            percent_y1 = label_y2 + 2
            percent_x2 = percent_x1 + percent_tw + padding*2
            percent_y2 = percent_y1 + percent_th + padding*2

            cv2.rectangle(display_frame, (label_x1, label_y1), (label_x2, label_y2), (0, 0, 0), -1)
            cv2.putText(display_frame, label_text, (label_x1 + padding, label_y2 - padding),
                        cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, zone_color, label_thickness, cv2.LINE_AA)

            cv2.rectangle(display_frame, (percent_x1, percent_y1), (percent_x2, percent_y2), (0, 0, 0), -1)
            cv2.putText(display_frame, percent_text, (percent_x1 + padding, percent_y2 - padding),
                        cv2.FONT_HERSHEY_SIMPLEX, percent_font_scale, zone_color, percent_thickness, cv2.LINE_AA)

        socketio.emit(f'update_zones_{self.processor_id}', {'zones': zone_data})

        return display_frame

    def start(self):
        if self._processing_task is not None:
            logging.info(f"Processor {self.processor_id} already running")
            return
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            logging.error(f"Could not reopen video {self.video_path} for processor {self.processor_id}")
            socketio.emit(f'update_status_{self.processor_id}', {'message': f'{self.camera_name} - Error: Could not reopen video'})
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_pos)
        self.counting = True
        self.running = True
        self.paused = False
        socketio.emit(f'update_status_{self.processor_id}', {'message': f'{self.camera_name} - System started'})
        self._processing_task = socketio.start_background_task(self.process_video)

    def stop(self):
        self.running = False
        self.counting = False
        self.paused = False
        self.current_pos = 0
        if self.cap:
            self.cap.release()
            self.cap = None
        socketio.emit(f'update_status_{self.processor_id}', {'message': f'{self.camera_name} - System stopped'})
        frame = self.first_frame.copy() if self.first_frame is not None else None
        processed_frame = self.process_frame(frame)
        if processed_frame is not None:
            frame_data = self.encode_frame(processed_frame)
            socketio.emit(f'video_frame_{self.processor_id}', {'frame': frame_data})

    def clear_dashboard(self):
        self.preserved_data = {'zones': {}}
        self.zone_id_map = {}
        self.next_zone_id = 1
        socketio.emit(f'update_zones_{self.processor_id}', {'zones': {}})

    def toggle_pause_resume(self):
        if self.paused:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                socketio.emit(f'update_status_{self.processor_id}', {'message': f'{self.camera_name} - Error: Could not reopen video'})
                return
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_pos)
            self.paused = False
            socketio.emit(f'update_status_{self.processor_id}', {'message': f'{self.camera_name} - System resumed'})
        else:
            self.paused = True
            self.current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            if self.cap:
                self.cap.release()
                self.cap = None
            socketio.emit(f'update_status_{self.processor_id}', {'message': f'{self.camera_name} - System paused'})

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
                    self.current_pos = 0
                    continue
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                self.current_frame = frame.copy()
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
        socketio.emit(f'update_status_{self.processor_id}', {'message': f'{self.camera_name} - Video processing stopped'})

processors = {}

def initialize_configured_videos():
    global current_config
    if not current_config:
        current_config = [
            {'id': 'vehicle_1', 'name': 'Vehicle Cam A', 'type': 'vehicle', 'video_path': 'videos/v1.mp4'},
            {'id': 'vehicle_2', 'name': 'Vehicle Cam B', 'type': 'vehicle', 'video_path': 'videos/v2.mp4'},
            {'id': 'face_1', 'name': 'Face Cam A', 'type': 'face', 'video_path': 'videos/Generate_a_realistic_202508171102_f1rci.mp4'},
            {'id': 'face_2', 'name': 'Face Cam B', 'type': 'face', 'video_path': 'videos/istockphoto-1532931548-640_adpp_is.mp4'},
            {'id': 'zone_1', 'name': 'Zone Cam A', 'type': 'zone', 'video_path': 'Uploads/istockphoto-2148384684-640_adpp_is.mp4'},
            {'id': 'zone_2', 'name': 'Zone Cam B', 'type': 'zone', 'video_path': 'Uploads/istockphoto-2148384684-640_adpp_is.mp4'}
        ]
    for camera in current_config:
        pid = camera['id']
        if os.path.exists(camera['video_path']):
            if camera['type'] == 'vehicle':
                processors[pid] = VehicleProcessor(camera['video_path'], pid, camera['name'])
            elif camera['type'] == 'face':
                processors[pid] = FaceProcessor(camera['video_path'], pid, camera['name'])
            elif camera['type'] == 'zone':
                processors[pid] = ZoneProcessor(camera['video_path'], pid, camera['name'])
            logging.info(f"Initialized processor {pid} with video: {camera['video_path']}")
        else:
            logging.error(f"Video path not found for processor {pid}: {camera['video_path']}")
            socketio.emit(f'update_status_{pid}', {'message': f'{camera["name"]} - Error: Video file {camera["video_path"]} not found'})

@app.route('/')
def index():
    return render_template('config.html')

@app.route('/dashboard')
def dashboard():
    if not processors:
        initialize_configured_videos()
    return render_template('index.html')

@app.route('/configure', methods=['POST'])
def configure():
    global current_config, processors
    try:
        data = request.get_json()
        cameras = data.get('cameras', [])
        if not cameras:
            return jsonify({'success': False, 'error': 'No cameras configured'})
        
        # Validate configuration
        used_videos = set()
        new_config = []
        for camera in cameras:
            video_id = camera['id']
            if video_id not in AVAILABLE_VIDEOS:
                return jsonify({'success': False, 'error': f'Invalid video ID: {video_id}'})
            if video_id in used_videos:
                return jsonify({'success': False, 'error': f'Duplicate video stream: {video_id}'})
            used_videos.add(video_id)
            new_config.append({
                'id': video_id,
                'name': AVAILABLE_VIDEOS[video_id]['name'],
                'type': camera['type'],
                'video_path': AVAILABLE_VIDEOS[video_id]['path']
            })

        # Stop existing processors
        for processor_id in list(processors.keys()):
            processors[processor_id].stop()
            del processors[processor_id]

        # Update configuration and initialize new processors
        current_config = new_config
        for camera in current_config:
            processor_id = camera['id']
            if camera['type'] == 'vehicle':
                processors[processor_id] = VehicleProcessor(
                    video_path=camera['video_path'],
                    processor_id=processor_id,
                    camera_name=camera['name']
                )
            elif camera['type'] == 'face':
                processors[processor_id] = FaceProcessor(
                    video_path=camera['video_path'],
                    processor_id=processor_id,
                    camera_name=camera['name']
                )
            elif camera['type'] == 'zone':
                processors[processor_id] = ZoneProcessor(
                    video_path=camera['video_path'],
                    processor_id=processor_id,
                    camera_name=camera['name']
                )
        
        return jsonify({'success': True, 'redirect': url_for('dashboard')})
    except Exception as e:
        logging.error(f"Configuration error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_config')
def get_config():
    return jsonify({'cameras': current_config})

@app.route('/videos/<filename>')
def serve_video(filename):
    try:
        return send_from_directory(app.config['VIDEOS_FOLDER'], filename, mimetype='video/mp4')
    except Exception as e:
        logging.error(f"Error serving video {filename}: {str(e)}")
        return {'error': 'Video not found'}, 404

@app.route('/get_vehicle_details/<processor_id>/<int:event_id>', methods=['GET'])
def get_vehicle_details(processor_id, event_id):
    if processor_id in processors and isinstance(processors[processor_id], VehicleProcessor):
        details = processors[processor_id].get_vehicle_details(event_id)
        if details:
            return jsonify(details)
    return jsonify({'error': 'Details not found'}), 404

@socketio.on('select_camera')
def handle_select_camera(data):
    camera_id = data.get('camera_id')
    active_pids = []
    if camera_id == 'all':
        active_pids = list(AVAILABLE_VIDEOS.keys())
    elif camera_id == 'vehicle':
        active_pids = ['vehicle_1', 'vehicle_2']
    elif camera_id == 'face':
        active_pids = ['face_1', 'face_2']
    elif camera_id == 'zone':
        active_pids = ['zone_1', 'zone_2']
    else:
        active_pids = [camera_id]
    for pid in list(processors.keys()):
        if pid not in active_pids and processors[pid].running:
            processors[pid].stop()
            del processors[pid]
    for pid in active_pids:
        if pid not in processors and pid in AVAILABLE_VIDEOS:
            camera = next((c for c in current_config if c['id'] == pid), None)
            if not camera:
                camera = {
                    'id': pid,
                    'name': AVAILABLE_VIDEOS[pid]['name'],
                    'type': AVAILABLE_VIDEOS[pid]['type'],
                    'video_path': AVAILABLE_VIDEOS[pid]['path']
                }
                current_config.append(camera)
            if os.path.exists(camera['video_path']):
                if camera['type'] == 'vehicle':
                    processors[pid] = VehicleProcessor(camera['video_path'], pid, camera['name'])
                elif camera['type'] == 'face':
                    processors[pid] = FaceProcessor(camera['video_path'], pid, camera['name'])
                elif camera['type'] == 'zone':
                    processors[pid] = ZoneProcessor(camera['video_path'], pid, camera['name'])
                if data.get('was_running', {}).get(pid, False):
                    processors[pid].start()

@socketio.on('add_line_point')
def handle_add_line_point(data):
    processor_id = data['processor_id']
    x, y = data['x'], data['y']
    if processor_id in processors and isinstance(processors[processor_id], VehicleProcessor):
        processors[processor_id].add_line_point(x, y)

@socketio.on('complete_line')
def handle_complete_line(data):
    processor_id = data['processor_id']
    if processor_id in processors and isinstance(processors[processor_id], VehicleProcessor):
        processors[processor_id].complete_line()

@socketio.on('delete_line')
def handle_delete_line(data):
    processor_id = data['processor_id']
    if processor_id in processors and isinstance(processors[processor_id], VehicleProcessor):
        processors[processor_id].delete_line()

@socketio.on('start')
def handle_start(data):
    processor_id = data['processor_id']
    if processor_id in processors:
        processors[processor_id].start()

@socketio.on('stop')
def handle_stop(data):
    processor_id = data['processor_id']
    if processor_id in processors:
        processors[processor_id].stop()

@socketio.on('toggle_pause_resume')
def handle_toggle_pause_resume(data):
    processor_id = data.get('processor_id')
    if processor_id in processors:
        processors[processor_id].toggle_pause_resume()

@socketio.on('clear_dashboard')
def handle_clear_dashboard(data):
    processor_id = data.get('processor_id')
    if processor_id in processors:
        processors[processor_id].clear_dashboard()

@socketio.on('request_frame')
def handle_request_frame(data):
    processor_id = data.get('processor_id')
    if processor_id in processors:
        if 'face_id' in data and isinstance(processors[processor_id], FaceProcessor):
            processors[processor_id].selected_face_id = data['face_id']

@socketio.on('reset_selected')
def handle_reset_selected(data):
    processor_id = data.get('processor_id')
    if processor_id in processors:
        if isinstance(processors[processor_id], FaceProcessor):
            processors[processor_id].selected_face_id = None

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)