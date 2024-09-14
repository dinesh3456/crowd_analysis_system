import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import threading
from PIL import Image, ImageTk

# Import all the functions from our previous script
from intelligent_video_surveillance import SimpleTracker, detect_and_track, estimate_crowd_density, analyze_behavior, detect_anomalies

# Initialize YOLOv8 model and tracker
model = YOLO('yolov8n.pt')
tracker = SimpleTracker()

class SurveillanceDashboard:
    def __init__(self, window, video_source=0):
        self.window = window
        self.window.title("Intelligent Video Surveillance Dashboard")
        
        self.video_source = video_source
        self.vid = cv2.VideoCapture(self.video_source)
        
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.sidebar = ttk.Frame(window, padding="10")
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y, expand=False)
        
        self.density_label = ttk.Label(self.sidebar, text="Crowd Density:")
        self.density_label.pack(pady=5)
        
        self.behavior_label = ttk.Label(self.sidebar, text="Behavior Statistics:")
        self.behavior_label.pack(pady=5)
        
        self.anomaly_label = ttk.Label(self.sidebar, text="Anomalies:", foreground="red")
        self.anomaly_label.pack(pady=5)
        
        self.update_thread = threading.Thread(target=self.update, daemon=True)
        self.update_thread.start()
        
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def update(self):
        ret, frame = self.vid.read()
        if ret:
            tracked_objects = detect_and_track(frame)
            density_map = estimate_crowd_density(frame, tracked_objects)
            behaviors = analyze_behavior(tracker.trajectories)
            anomalies = detect_anomalies(density_map, behaviors)
            
            frame_with_density = self.draw_density_map(frame.copy(), density_map)
            frame_with_boxes = self.draw_boxes_and_behaviors(frame, tracked_objects, behaviors)
            
            combined_frame = cv2.addWeighted(frame_with_density, 0.7, frame_with_boxes, 0.3, 0)
            
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            
            # Update sidebar information
            self.update_density_info(density_map)
            self.update_behavior_info(behaviors)
            self.update_anomaly_info(anomalies)
        
        self.window.after(15, self.update)
    
   
    
    def draw_density_map(self, frame, density_map):
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


    
    def draw_boxes_and_behaviors(self, frame, objects, behaviors):
        behavior_colors = {"Walking": (0, 255, 0), "Running": (0, 0, 255), "Standing": (255, 0, 0)}
        for obj_id, obj in objects.items():
            x1, y1, x2, y2 = map(int, obj['bbox'])
            centroid = obj['centroid']
            
            # Determine behavior based on position
            grid_x, grid_y = int(centroid[0] // (frame.shape[1] / 3)), int(centroid[1] // (frame.shape[0] / 3))
            cell_behaviors = [b for b, count in behaviors.items() if count > 0]
            behavior = cell_behaviors[0] if cell_behaviors else "Unknown"
            
            color = behavior_colors.get(behavior, (128, 128, 128))  # Gray for unknown
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'ID: {obj_id} - {behavior}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame
    
    def update_density_info(self, density_map):
        total_density = np.sum(density_map)
        max_density = np.max(density_map)
        density_text = f"Total People: {total_density}\nMax Density: {max_density}"
        self.density_label.config(text=density_text)
    
    def update_behavior_info(self, behaviors):
        behavior_counts = {"Running": 0, "Walking": 0, "Standing": 0}
        for behavior in behaviors.values():
            behavior_counts[behavior] += 1
        behavior_text = "\n".join([f"{k}: {v}" for k, v in behavior_counts.items()])
        self.behavior_label.config(text=f"Behavior Statistics:\n{behavior_text}")
    
    def update_anomaly_info(self, anomalies):
        if anomalies:
            anomaly_text = "\n".join(anomalies)
            self.anomaly_label.config(text=f"Anomalies:\n{anomaly_text}", foreground="red")
        else:
            self.anomaly_label.config(text="No anomalies detected", foreground="green")
    
    def on_closing(self):
        self.vid.release()
        self.window.destroy()

def main():
    root = tk.Tk()
    app = SurveillanceDashboard(root)
    root.mainloop()

if __name__ == "__main__":
    main()