import cv2
import numpy as np
from ultralytics import YOLO

# Initialize YOLOv8 model
model = YOLO('yolov8n.pt')

class SimpleTracker:
    def __init__(self):
        self.next_id = 1
        self.objects = {}

    def update(self, detections):
        new_objects = {}
        
        for detection in detections:
            x1, y1, x2, y2, conf, class_id = detection
            centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            closest_id = None
            closest_distance = float('inf')
            
            for obj_id, obj in self.objects.items():
                distance = np.linalg.norm(np.array(centroid) - np.array(obj['centroid']))
                if distance < closest_distance:
                    closest_id = obj_id
                    closest_distance = distance
            
            if closest_distance < 50:  # Threshold for considering it the same object
                new_objects[closest_id] = {'bbox': (x1, y1, x2, y2), 'centroid': centroid}
            else:
                new_objects[self.next_id] = {'bbox': (x1, y1, x2, y2), 'centroid': centroid}
                self.next_id += 1
        
        self.objects = new_objects
        return self.objects

tracker = SimpleTracker()

def detect_and_track(frame):
    # Perform detection
    results = model(frame)
    
    # Extract detections for persons only (class 0 in COCO dataset)
    detections = results[0].boxes.data
    persons = detections[detections[:, 5] == 0].cpu().numpy()  # Filter for persons (class 0)
    
    # Update tracker
    tracked_objects = tracker.update(persons)
    
    return tracked_objects

def draw_boxes_and_ids(frame, objects):
    for obj_id, obj in objects.items():
        x1, y1, x2, y2 = map(int, obj['bbox'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {obj_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

def main():
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide video file path
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        tracked_objects = detect_and_track(frame)
        frame_with_boxes = draw_boxes_and_ids(frame, tracked_objects)
        
        cv2.imshow('Object Detection and Tracking', frame_with_boxes)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()