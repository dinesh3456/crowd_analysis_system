import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# Initialize YOLOv8 model
model = YOLO('yolov8n.pt')

class SimpleTracker:
    def __init__(self):
        self.next_id = 1
        self.objects = {}
        self.trajectories = {}
        self.max_trajectory_length = 30  # Store last 30 positions

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
                self.trajectories[closest_id].append(centroid)
                if len(self.trajectories[closest_id]) > self.max_trajectory_length:
                    self.trajectories[closest_id].popleft()
            else:
                new_objects[self.next_id] = {'bbox': (x1, y1, x2, y2), 'centroid': centroid}
                self.trajectories[self.next_id] = deque([centroid], maxlen=self.max_trajectory_length)
                self.next_id += 1
        
        self.objects = new_objects
        return self.objects

tracker = SimpleTracker()

def detect_and_track(frame):
    results = model(frame)
    detections = results[0].boxes.data
    persons = detections[detections[:, 5] == 0].cpu().numpy()  # Filter for persons (class 0)
    tracked_objects = tracker.update(persons)
    return tracked_objects

def estimate_crowd_density(frame, objects):
    height, width = frame.shape[:2]
    grid_size = 3  # 3x3 grid
    density_map = np.zeros((grid_size, grid_size), dtype=int)
    
    cell_height = height // grid_size
    cell_width = width // grid_size
    
    for obj in objects.values():
        centroid = obj['centroid']
        grid_x = int(centroid[0] // cell_width)
        grid_y = int(centroid[1] // cell_height)
        if grid_x < grid_size and grid_y < grid_size:
            density_map[grid_y, grid_x] += 1
    
    return density_map

def analyze_behavior(trajectories):
    behaviors = {}
    for obj_id, trajectory in trajectories.items():
        if len(trajectory) < 2:
            continue
        
        # Calculate speed
        distances = [np.linalg.norm(np.array(trajectory[i]) - np.array(trajectory[i-1])) 
                     for i in range(1, len(trajectory))]
        avg_speed = np.mean(distances)
        
        # Determine behavior based on speed
        if avg_speed > 10:
            behaviors[obj_id] = "Running"
        elif avg_speed > 2:
            behaviors[obj_id] = "Walking"
        else:
            behaviors[obj_id] = "Standing"
    
    return behaviors

def draw_boxes_and_behaviors(frame, objects, behaviors):
    behavior_colors = {"Walking": (0, 255, 0), "Running": (0, 0, 255), "Standing": (255, 0, 0)}
    
    for obj_id, obj in objects.items():
        x1, y1, x2, y2 = map(int, obj['bbox'])
        behavior = behaviors.get(obj_id, "Unknown")
        color = behavior_colors.get(behavior, (128, 128, 128))  # Gray for unknown
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'ID: {obj_id} - {behavior}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def detect_anomalies(density_map, behaviors):
    anomalies = []
    
    # Detect high density anomalies
    max_density = np.max(density_map)
    if max_density > 10:  # Arbitrary threshold, adjust as needed
        anomalies.append(f"High density detected: {max_density} people in one area")
    
    # Detect unusual behavior patterns
    behavior_counts = {"Running": 0, "Walking": 0, "Standing": 0}
    for behavior in behaviors.values():
        behavior_counts[behavior] += 1
    
    total_people = sum(behavior_counts.values())
    if total_people > 0:
        running_ratio = behavior_counts["Running"] / total_people
        if running_ratio > 0.5:  # If more than 50% of people are running
            anomalies.append(f"Unusual number of people running: {running_ratio:.2%}")
    
    return anomalies
def draw_density_map(frame, density_map):
    height, width = frame.shape[:2]
    grid_size = density_map.shape[0]
    cell_height = height // grid_size
    cell_width = width // grid_size
    
    for y in range(grid_size):
        for x in range(grid_size):
            density = density_map[y, x]
            color = (0, 255 * min(density / 5, 1), 0)  # Green to Red based on density
            cv2.rectangle(frame, (x * cell_width, y * cell_height),
                          ((x + 1) * cell_width, (y + 1) * cell_height),
                          color, -1)
            cv2.putText(frame, str(density), (x * cell_width + 10, y * cell_height + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return cv2.addWeighted(frame, 0.7, frame, 0.3, 0)  # Add transparency



def main():

    video_path = r"dataset1.mov"  # Update this path if necessary
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        tracked_objects = detect_and_track(frame)
        density_map = estimate_crowd_density(frame, tracked_objects)
        behaviors = analyze_behavior(tracker.trajectories)
        anomalies = detect_anomalies(density_map, behaviors)
        
        frame_with_density = draw_density_map(frame.copy(), density_map)
        frame_with_boxes = draw_boxes_and_behaviors(frame, tracked_objects, behaviors)
        
        combined_frame = cv2.addWeighted(frame_with_density, 0.7, frame_with_boxes, 0.3, 0)
        
        # Display anomalies
        y_offset = 30
        for anomaly in anomalies:
            cv2.putText(combined_frame, f"ANOMALY: {anomaly}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30
        
        # Calculate and display behavior statistics
        behavior_counts = {"Running": 0, "Walking": 0, "Standing": 0}
        for behavior in behaviors.values():
            if behavior in behavior_counts:
                behavior_counts[behavior] += 1
        
        y_offset = 90
        for behavior, count in behavior_counts.items():
            cv2.putText(combined_frame, f"{behavior}: {count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30
        
        cv2.imshow('Intelligent Video Surveillance', combined_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()