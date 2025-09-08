import cv2
from ultralytics import YOLO
import numpy as np
import os
import time
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import base64
from io import BytesIO
from datetime import datetime
from collections import deque
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

# Ensure videos directory exists
VIDEOS_FOLDER = 'videos'
if not os.path.exists(VIDEOS_FOLDER):
    os.makedirs(VIDEOS_FOLDER)

app.config['VIDEOS_FOLDER'] = VIDEOS_FOLDER

# Static video paths
STATIC_VIDEO_PATHS = {
    '1': "vedios/v1.mp4",  # Fixed video for Cam A
    '2': "vedios/v2.mp4"   # Fixed video for Cam B
}

class VideoProcessor:
    def __init__(self, video_path, processor_id):
        self.video_path = video_path
        self.processor_id = processor_id
        self.line_points = []
        self.line_completed = False
        self.counting = False
        self.in_count = 0
        self.out_count = 0
        self.crossed = {}
        self.captured_info = {}
        self.vehicle_events = []
        self.recordings = {}
        self.frame_buffer = deque(maxlen=60)
        self.car_counter = 0
        self.current_plate_ids = set()
        self.cap = cv2.VideoCapture(video_path)
        self.running = False
        self._processing_task = None
        self.high_conf_threshold = 0.70
        self.plate_model = YOLO("license_plate_detector.pt")
        self.car_model = YOLO("yolov8n.pt")
        self.class_names = {2: "Car", 5: "Bus", 7: "Truck"}
        self.detect_classes = [2, 5, 7]
        
        if not self.cap.isOpened():
            logging.error(f"Could not open video {video_path} for processor {processor_id}")
            self.cap = None
            socketio.emit(f'update_status_{self.processor_id}', {'message': 'Error: Could not open video'})
            return
            
        ret, self.first_frame = self.cap.read()
        if not ret:
            logging.error(f"Could not read first frame from {video_path} for processor {processor_id}")
            self.cap = None
            socketio.emit(f'update_status_{self.processor_id}', {'message': 'Error: Could not read first frame'})
            return
            
        self.frame_width = 960  # Normal size for clear display
        self.frame_height = 540
        self.first_frame = cv2.resize(self.first_frame, (self.frame_width, self.frame_height))
        
        socketio.emit(f'frame_dimensions_{self.processor_id}', {
            'width': self.frame_width,
            'height': self.frame_height
        })
        
        frame_data = self.encode_frame(self.first_frame)
        socketio.emit(f'video_frame_{self.processor_id}', {'frame': frame_data})

    def add_line_point(self, x, y):
        x = max(0, min(int(x), self.frame_width - 1))
        y = max(0, min(int(y), self.frame_height - 1))
        if len(self.line_points) < 2:
            self.line_points.append((x, y))
            socketio.emit(f'update_status_{self.processor_id}', {'message': f'Point added at ({x}, {y})'})
            
        frame = self.first_frame.copy() if self.first_frame is not None else None
        processed_frame = self.process_frame(frame)
        if processed_frame is not None:
            frame_data = self.encode_frame(processed_frame)
            socketio.emit(f'video_frame_{self.processor_id}', {'frame': frame_data})

    def complete_line(self):
        if len(self.line_points) not in [0, 2]:
            socketio.emit(f'update_status_{self.processor_id}', {'message': 'Invalid number of points. Add exactly 2 points for line or delete to start without.'})
            return
        self.line_completed = len(self.line_points) == 2
        socketio.emit(f'update_status_{self.processor_id}', {'message': f'System starting{" with counting line" if self.line_completed else " without counting"}'})
        self.start()

    def delete_line(self):
        self.line_points = []
        self.line_completed = False
        socketio.emit(f'update_status_{self.processor_id}', {'message': 'Line deleted'})
        frame = self.first_frame.copy() if self.first_frame is not None else None
        processed_frame = self.process_frame(frame)
        if processed_frame is not None:
            frame_data = self.encode_frame(processed_frame)
            socketio.emit(f'video_frame_{self.processor_id}', {'frame': frame_data})

    def get_side(self, center):
        (x1, y1), (x2, y2) = self.line_points
        return (center[0] - x1) * (y2 - y1) - (center[1] - y1) * (x2 - x1)

    def process_frame(self, frame):
        if frame is None:
            return None
            
        display_frame = frame.copy()
        for p in self.line_points:
            cv2.circle(display_frame, p, 5, (0, 0, 255), -1)
        if len(self.line_points) == 2:
            cv2.line(display_frame, self.line_points[0], self.line_points[1], (0, 0, 255), 2)

        if self.counting:
            plate_results = self.plate_model.track(frame, conf=0.3, persist=True)
            car_results = self.car_model.track(frame, conf=0.5, persist=True)
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
                cv2.putText(display_frame, "License Plate", (x1+1, y1-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, shadow_color, 2)
                cv2.putText(display_frame, "License Plate", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 2)

                if plate_id not in self.captured_info:
                    self.car_counter += 1
                    car_name = f"car{self.car_counter}"
                    self.captured_info[plate_id] = {
                        'last_time': 0,
                        'car_name': car_name,
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'detected_text': "",
                        'appear_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'disappear_time': None,
                        'mode': None,
                        'cross_time': None,
                        'plate_img': "",
                        'car_img': "",
                        'video_path': "",
                        'video_duration': "4 seconds",
                        'best_conf': 0.0,
                        'event_id': None,
                        'camera': 'Cam A' if self.processor_id == '1' else 'Cam B'
                    }
                    # Create an event immediately upon detection
                    event = self.captured_info[plate_id].copy()
                    event['mode'] = "Detected"
                    self.vehicle_events.append(event)
                    event_id = len(self.vehicle_events) - 1
                    self.captured_info[plate_id]['event_id'] = event_id
                    # Start recording when a new car is detected
                    self.recordings[event_id] = list(self.frame_buffer)  # Initialize recording with current buffer

                self.captured_info[plate_id]['x1'] = x1
                self.captured_info[plate_id]['y1'] = y1
                self.captured_info[plate_id]['x2'] = x2
                self.captured_info[plate_id]['y2'] = y2

                best_conf = self.captured_info[plate_id]['best_conf']
                now = time.time()
                last_time = self.captured_info[plate_id]['last_time']
                
                if conf > best_conf and now - last_time > 1:
                    plate_img = frame[y1:y2, x1:x2]
                    if plate_img.size == 0:
                        continue
                        
                    upscale_factor = 5
                    plate_upscaled = cv2.resize(
                        plate_img,
                        (plate_img.shape[1]*upscale_factor, plate_img.shape[0]*upscale_factor),
                        interpolation=cv2.INTER_CUBIC
                    )
                    
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
                        cv2.putText(display_frame, self.captured_info[plate_id]['car_name'], (cx1, cy1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    self.captured_info[plate_id]['plate_img'] = plate_base64
                    self.captured_info[plate_id]['car_img'] = car_base64
                    self.captured_info[plate_id]['last_time'] = now
                    self.captured_info[plate_id]['best_conf'] = conf

                    # Update the event with new images
                    event_id = self.captured_info[plate_id]['event_id']
                    self.vehicle_events[event_id]['plate_img'] = plate_base64
                    self.vehicle_events[event_id]['car_img'] = car_base64

                    # If this is a new best confidence, finalize the previous recording and start a new one
                    if event_id in self.recordings:
                        self.finalize_recording(event_id)
                    self.recordings[event_id] = list(self.frame_buffer)  # Start new recording

                    if conf >= self.high_conf_threshold:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        mode = self.captured_info[plate_id]['mode'] if self.captured_info[plate_id]['mode'] is not None else "Detected"
                        self.captured_info[plate_id]['cross_time'] = timestamp if self.captured_info[plate_id]['cross_time'] is None else self.captured_info[plate_id]['cross_time']
                        self.vehicle_events[event_id]['mode'] = mode
                        self.vehicle_events[event_id]['cross_time'] = timestamp
                        logging.info(f"High confidence detection for vehicle {plate_id}: {mode} at {timestamp}")

            if self.line_completed:
                for plate_id in self.current_plate_ids:
                    if plate_id in self.captured_info:
                        info = self.captured_info[plate_id]
                        x1 = info['x1']
                        y1 = info['y1']
                        x2 = info['x2']
                        y2 = info['y2']
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        side = self.get_side(center)
                        
                        if plate_id not in self.crossed:
                            self.crossed[plate_id] = side
                        else:
                            last_side = self.crossed[plate_id]
                            if last_side * side < 0 and info['mode'] is None:
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                mode = "In" if side > 0 else "Out"
                                info['mode'] = mode
                                info['cross_time'] = timestamp
                                logging.info(f"Vehicle {plate_id} crossed line: {mode} at {timestamp}")
                                
                                if mode == "In":
                                    self.in_count += 1
                                else:
                                    self.out_count += 1
                                
                                if info['event_id'] is not None:
                                    self.vehicle_events[info['event_id']]['mode'] = mode
                                    self.vehicle_events[info['event_id']]['cross_time'] = timestamp
                                    
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

            socketio.emit(f'update_counts_{self.processor_id}', {
                'in_count': self.in_count,
                'out_count': self.out_count,
                'vehicle_events': [{
                    'matricule': e['detected_text'],
                    'mode': e['mode'],
                    'cross_time': e['cross_time'],
                    'plate_img': e.get('plate_img', ''),
                    'car_img': e.get('car_img', ''),
                    'video_path': e.get('video_path', ''),
                    'video_duration': e.get('video_duration', '4 seconds'),
                    'id': i,
                    'camera': e.get('camera', 'Cam A' if self.processor_id == '1' else 'Cam B')
                } for i, e in enumerate(self.vehicle_events) if e['mode'] is not None]
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
        logging.debug(f"Video saved: {path}")
        del self.recordings[event_id]

    def encode_image(self, img):
        _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return base64.b64encode(buffer).decode('utf-8')

    def encode_frame(self, frame):
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return base64.b64encode(buffer).decode('utf-8')

    def get_vehicle_details(self, event_id):
        if 0 <= event_id < len(self.vehicle_events):
            event = self.vehicle_events[event_id]
            return {
                'matricule': event['detected_text'],
                'mode': event['mode'],
                'cross_time': event['cross_time'],
                'car_img': event.get('car_img', ''),
                'plate_img': event.get('plate_img', ''),
                'video_path': event.get('video_path', ''),
                'video_duration': event.get('video_duration', '4 seconds'),
                'camera': event.get('camera', 'Cam A' if self.processor_id == '1' else 'Cam B')
            }
        logging.error(f"Invalid event_id: {event_id}")
        return None

    def process_video(self):
        self.running = True
        while self.running and self.cap and self.cap.isOpened():
            if self.counting:
                ret, frame = self.cap.read()
                if not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                self.frame_buffer.append(frame.copy())
            else:
                frame = self.first_frame.copy() if self.first_frame is not None else None
                
            processed_frame = self.process_frame(frame)
            if processed_frame is not None:
                frame_data = self.encode_frame(processed_frame)
                socketio.emit(f'video_frame_{self.processor_id}', {'frame': frame_data})
                
            socketio.sleep(0.03)
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self._processing_task = None
        socketio.emit(f'update_status_{self.processor_id}', {'message': 'Video processing stopped'})

    def start(self):
        if self._processing_task is not None:
            return
            
        if self.cap:
            self.cap.release()
            
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            socketio.emit(f'update_status_{self.processor_id}', {'message': 'Error: Could not reopen video'})
            return
            
        self.counting = True
        self.running = True
        socketio.emit(f'update_status_{self.processor_id}', {'message': 'System started'})
        self._processing_task = socketio.start_background_task(self.process_video)

    def stop(self):
        self.running = False
        self.counting = False
        self.in_count = 0
        self.out_count = 0
        self.crossed = {}
        self.captured_info = {}
        self.vehicle_events = []
        self.recordings = {}
        self.frame_buffer.clear()
        self.car_counter = 0
        self.current_plate_ids = set()
        
        if self.cap:
            self.cap.release()
            self.cap = None
            
        socketio.emit(f'update_status_{self.processor_id}', {'message': 'System stopped'})
        socketio.emit(f'update_counts_{self.processor_id}', {
            'in_count': 0,
            'out_count': 0,
            'vehicle_events': []
        })
        
        frame = self.first_frame.copy() if self.first_frame is not None else None
        processed_frame = self.process_frame(frame)
        if processed_frame is not None:
            frame_data = self.encode_frame(processed_frame)
            socketio.emit(f'video_frame_{self.processor_id}', {'frame': frame_data})

processors = {}

def initialize_static_videos():
    for pid, video_path in STATIC_VIDEO_PATHS.items():
        if os.path.exists(video_path):
            processors[pid] = VideoProcessor(video_path, pid)
            logging.info(f"Initialized processor {pid} with static video: {video_path}")
        else:
            logging.error(f"Static video path not found for processor {pid}: {video_path}")
            socketio.emit(f'update_status_{pid}', {'message': f'Error: Video file {video_path} not found'})

@app.route('/')
def index():
    if not processors:
        initialize_static_videos()
    return render_template('index.html')

@app.route('/videos/<filename>')
def serve_video(filename):
    try:
        return send_from_directory(app.config['VIDEOS_FOLDER'], filename, mimetype='video/mp4')
    except Exception as e:
        logging.error(f"Error serving video {filename}: {str(e)}")
        return {'error': 'Video not found'}, 404

@app.route('/get_details/<processor_id>/<int:event_id>', methods=['GET'])
def get_details(processor_id, event_id):
    if processor_id in processors:
        details = processors[processor_id].get_vehicle_details(event_id)
        if details:
            return jsonify(details)
    return jsonify({'error': 'Details not found'}), 404

@socketio.on('add_line_point')
def handle_add_line_point(data):
    processor_id = data['processor_id']
    x, y = data['x'], data['y']
    if processor_id in processors:
        processors[processor_id].add_line_point(x, y)

@socketio.on('complete_line')
def handle_complete_line(data):
    processor_id = data['processor_id']
    if processor_id in processors:
        processors[processor_id].complete_line()

@socketio.on('delete_line')
def handle_delete_line(data):
    processor_id = data['processor_id']
    if processor_id in processors:
        processors[processor_id].delete_line()

@socketio.on('stop')
def handle_stop(data):
    processor_id = data['processor_id']
    if processor_id in processors:
        processors[processor_id].stop()

if __name__ == '__main__':
    initialize_static_videos()
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)