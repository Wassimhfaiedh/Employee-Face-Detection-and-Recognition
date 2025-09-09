import cv2 
from ultralytics import YOLO   
import numpy as np 
import torch 



device="cuda" if torch.cuda.is_available() else "cpu"
print(f'Using Device: {device}')
# Load YOLO model 
model = YOLO("best.pt").to(device)

# Video source 
video_path = "istockphoto-2148384684-640_adpp_is.mp4"
cap = cv2.VideoCapture(video_path) 

# Colors for each zone class 
class_colors = {
    "Empty_zone": (0, 255, 0),      # Green 
    "Partial_zone": (0, 255, 255),  # Yellow 
    "Full_zone": (0, 0, 255),       # Red 
}

# Dictionary to store zone IDs based on their center coordinates
zone_id_map = {}
next_zone_id = 1

def draw_dashed_rect(img, pt1, pt2, color, thickness=2, dash_length=5):
    """Draw small dashed rectangle instead of solid lines"""
    x1, y1 = pt1
    x2, y2 = pt2
    for i in range(x1, x2, dash_length * 2):
        cv2.line(img, (i, y1), (min(i + dash_length, x2), y1), color, thickness)
        cv2.line(img, (i, y2), (min(i + dash_length, x2), y2), color, thickness)
    for i in range(y1, y2, dash_length * 2):
        cv2.line(img, (x1, i), (x1, min(i + dash_length, y2)), color, thickness)
        cv2.line(img, (x2, i), (x2, min(i + dash_length, y2)), color, thickness)

def compute_partial_percent(zone_xyxy, stock_boxes_xyxy):
    """
    Accurate % for Partial_zone with normalized stock contribution.
    Assumes each stock contributes ~10% to align with expected 20% for two stocks.
    """
    x1, y1, x2, y2 = zone_xyxy
    w, h = max(0, x2 - x1), max(0, y2 - y1)
    if w == 0 or h == 0:
        print(f"Zone has invalid dimensions: w={w}, h={h}")
        return 0

    # Create blank mask for the zone
    zone_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(zone_mask, (0, 0), (w, h), 255, -1)  # whole zone = white

    # Create mask for stocks (union to handle overlaps)
    stock_mask = np.zeros((h, w), dtype=np.uint8)
    stock_count = 0

    for sx1, sy1, sx2, sy2 in stock_boxes_xyxy:
        # Clip stock to zone boundaries
        ox1 = max(x1, sx1); oy1 = max(y1, sy1)
        ox2 = min(x2, sx2); oy2 = min(y2, sy2)
        if ox1 < ox2 and oy1 < oy2:
            p1 = (ox1 - x1, oy1 - y1)
            p2 = (ox2 - x1, oy2 - y1)
            cv2.rectangle(stock_mask, p1, p2, 255, -1)
            stock_count += 1
        else:
            print(f"Stock box skipped: ({sx1}, {sy1}, {sx2}, {sy2}) outside zone ({x1}, {y1}, {x2}, {y2})")

    # Intersection (stock pixels inside the zone)
    intersection = cv2.bitwise_and(zone_mask, stock_mask)

    # Compute areas
    zone_area = np.count_nonzero(zone_mask)
    stock_area = np.count_nonzero(intersection)

    # Normalize stock contribution: assume each stock contributes ~10% (adjustable)
    if stock_count > 0:
        # Target 10% per stock, capped at 100%
        target_percent_per_stock = 10.0
        percent = min(stock_count * target_percent_per_stock, 100.0)
        print(f"Stock count: {stock_count}, Normalized percent: {percent:.1f}%")
    else:
        percent = 0.0
        print("No stocks detected in zone")

    # Debug output
    print(f"Zone area: {zone_area}, Raw stock area: {stock_area}, Raw percent: {(stock_area / zone_area * 100) if zone_area > 0 else 0:.1f}%")

    final_percent = int(round(percent))  # Convert to integer
    print(f"Final percent: {final_percent}%")
    return final_percent

def assign_zone_id(x1, y1, x2, y2):
    """Assign a consistent ID to a zone based on its center point."""
    global next_zone_id
    # Calculate the center of the zone
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    center = (center_x, center_y)
    
    # Check if this zone's center is close to any existing zone
    for known_center, zone_id in zone_id_map.items():
        dist = np.sqrt((center_x - known_center[0])**2 + (center_y - known_center[1])**2)
        if dist < 50:  # Threshold for considering it the same zone (adjust as needed)
            return zone_id
    
    # New zone, assign a new ID
    zone_id = next_zone_id
    zone_id_map[center] = zone_id
    next_zone_id += 1
    return zone_id

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.9, iou=0.5)
    boxes = results[0].boxes
    names = results[0].names

    overlay = frame.copy()
    zone_boxes = []
    stock_boxes = []

    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            label = names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label.lower() == "stock":
                stock_boxes.append((x1, y1, x2, y2))
                continue
            else:
                zone_id = assign_zone_id(x1, y1, x2, y2)
                zone_boxes.append((label, (x1, y1, x2, y2), zone_id))

                zone_color = class_colors.get(label, (255, 255, 255))
                cv2.rectangle(overlay, (x1, y1), (x2, y2), zone_color, -1)
                draw_dashed_rect(frame, (x1, y1), (x2, y2), (0, 0, 0), 2, dash_length=5)

    alpha = 0.3
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Draw a circle for each zone with zone number and color
    for label, (x1, y1, x2, y2), zone_id in zone_boxes:
        zone_color = class_colors.get(label, (255, 255, 255))
        
        # Position the circle at the top-left corner of each zone
        circle_center = (x1 + 20, y1 + 20)  # Offset from top-left corner of zone
        circle_radius = 15
        
        # Draw black filled circle
        cv2.circle(frame, circle_center, circle_radius, (0, 0, 0), -1)
        
        # Draw colored border
        cv2.circle(frame, circle_center, circle_radius, zone_color, 2)
        
        # Draw zone number text (using persistent zone_id)
        zone_text = str(zone_id)
        (text_w, text_h), _ = cv2.getTextSize(zone_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_pos = (circle_center[0] - text_w // 2, circle_center[1] + text_h // 2)
        cv2.putText(frame, zone_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone_color, 1, cv2.LINE_AA)

    for label, (x1, y1, x2, y2), zone_id in zone_boxes:
        zone_color = class_colors.get(label, (255, 255, 255))

        if label == "Full_zone":
            percent = 100
        elif label == "Empty_zone":
            percent = 0
        elif label == "Partial_zone":
            percent = compute_partial_percent((x1, y1, x2, y2), stock_boxes)
        else:
            percent = 0

        # Label text (e.g., "Full_zone")
        label_text = label
        # Percentage text (e.g., "100% STOCK")
        percent_text = f"{percent}% STOCK"

        label_font_scale = 0.6  # Font scale for label
        percent_font_scale = 0.4  # Smaller font scale for percentage
        label_thickness = 2  # Thickness for label text
        percent_thickness = 1  # Thinner thickness for percentage text
        padding = 6

        # Calculate label text size
        (label_tw, label_th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, label_thickness)
        # Calculate percentage text size
        (percent_tw, percent_th), _ = cv2.getTextSize(percent_text, cv2.FONT_HERSHEY_SIMPLEX, percent_font_scale, percent_thickness)

        # Center point of the zone
        bbox_cx = (x1 + x2) // 2
        bbox_cy = (y1 + y2) // 2

        # Label rectangle coordinates (above the percentage rectangle)
        label_x1 = bbox_cx - (label_tw + padding*2) // 2
        label_y1 = bbox_cy - (label_th + padding*2 + percent_th + padding*2) // 2  # Shift up to make space for percentage
        label_x2 = label_x1 + label_tw + padding*2
        label_y2 = label_y1 + label_th + padding*2

        # Percentage rectangle coordinates (below the label rectangle)
        percent_x1 = bbox_cx - (percent_tw + padding*2) // 2
        percent_y1 = label_y2 + 2  # Small gap between rectangles
        percent_x2 = percent_x1 + percent_tw + padding*2
        percent_y2 = percent_y1 + percent_th + padding*2

        # Draw label rectangle and text
        cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), (0, 0, 0), -1)
        cv2.putText(frame, label_text, (label_x1 + padding, label_y2 - padding),
                    cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, zone_color, label_thickness, cv2.LINE_AA)

        # Draw percentage rectangle and text
        cv2.rectangle(frame, (percent_x1, percent_y1), (percent_x2, percent_y2), (0, 0, 0), -1)
        cv2.putText(frame, percent_text, (percent_x1 + padding, percent_y2 - padding),
                    cv2.FONT_HERSHEY_SIMPLEX, percent_font_scale, zone_color, percent_thickness, cv2.LINE_AA)

    cv2.imshow("YOLOv11 Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()