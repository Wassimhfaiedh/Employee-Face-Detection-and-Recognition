import cv2
from ultralytics import YOLO
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
import threading
import time
import torch
import os


device ="cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

# Define video paths for cameras (replace with actual paths)
camera_1_path = "Uploads\\istockphoto-2148384684-640_adpp_is.mp4"  
camera_2_path = "Uploads\\istockphoto-2148384684-640_adpp_is.mp4"  

class ZoneProcessor:
    def __init__(self, video_path, processor_id):
        self.video_path = video_path
        self.processor_id = processor_id
        self.model = YOLO('best.pt').to(device)  # Load your custom YOLO model
        self.cap = cv2.VideoCapture(video_path)
        self.running = False
        self.counting = False
        self.paused = False
        self._processing_task = None
        self.zone_id_map = {}
        self.next_zone_id = 1
        self.class_colors = {
            "Empty_zone": (0, 255, 0),      # Green
            "Partial_zone": (0, 255, 255),  # Yellow
            "Full_zone": (0, 0, 255),       # Red
        }

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

            zone_data[zone_id] = {'label': label, 'percent': percent}

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

    def encode_frame(self, frame):
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return base64.b64encode(buffer).decode('utf-8')

    def process_video(self):
        self.running = True
        while self.running and self.cap and self.cap.isOpened():
            if self.paused:
                socketio.sleep(0.1)  # Sleep to reduce CPU usage when paused
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
        socketio.emit(f'update_status_{self.processor_id}', {'message': 'Video processing stopped'})

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

# Global processor
processor = None

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('select_camera')
def handle_select_camera(data):
    global processor
    camera_id = data.get('camera_id')
    if camera_id == '1':
        video_path = camera_1_path
    elif camera_id == '2':
        video_path = camera_2_path
    else:
        socketio.emit('update_status_1', {'message': 'Invalid camera selection'})
        return

    if not os.path.exists(video_path):
        socketio.emit('update_status_1', {'message': f'Video file not found: {video_path}'})
        return

    if processor and processor.paused:
        socketio.emit('update_status_1', {'message': 'System is paused, please resume to select a camera'})
        return

    if processor:
        processor.release()
    processor = ZoneProcessor(video_path, '1')
    if processor.cap is None:
        socketio.emit('update_status_1', {'message': f'Failed to process video: {video_path}'})
        return
    processor.start()

@socketio.on('toggle_pause_resume')
def handle_toggle_pause_resume(data):
    global processor
    if processor:
        processor.toggle_pause_resume()

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)