import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import threading
from PIL import Image, ImageTk
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from intelligent_video_surveillance import SimpleTracker, detect_and_track, estimate_crowd_density, analyze_behavior, detect_anomalies

model = YOLO('yolov8n.pt')
tracker = SimpleTracker()

class SurveillanceDashboard:
    def __init__(self, window, video_source='dataset1.mov'):
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
        
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.sidebar)
        self.canvas_plot.get_tk_widget().pack(pady=10)
        
        self.update_thread = threading.Thread(target=self.update, daemon=True)
        self.update_thread.start()
        
        self.log_file = 'surveillance_log.csv'
        self.init_log_file()

        self.tracker = SimpleTracker()
        
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def init_log_file(self):
        with open(self.log_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Total People', 'Max Density', 'Running', 'Walking', 'Standing', 'Anomalies'])
    
    def log_data(self, total_people, max_density, behaviors, anomalies):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                timestamp,
                total_people,
                max_density,
                behaviors['Running'],
                behaviors['Walking'],
                behaviors['Standing'],
                '; '.join(anomalies) if anomalies else 'None'
            ])
    
    def update(self):
        ret, frame = self.vid.read()
        if ret:
            # Detect and track objects
            detections = model(frame)[0].boxes.data.cpu().numpy()
            persons = detections[detections[:, 5] == 0]  # Filter for persons (class 0)
            tracked_objects = self.tracker.update(persons)

            # Analyze behaviors
            behaviors = analyze_behavior(self.tracker.trajectories)
            
            # Estimate crowd density
            density_map = estimate_crowd_density(frame, tracked_objects)
            
            # Detect anomalies
            anomalies = detect_anomalies(density_map, behaviors)
            
            # Update visualizations
            frame_with_density = self.draw_density_map(frame.copy(), density_map)
            frame_with_boxes = self.draw_boxes_and_behaviors(frame, tracked_objects, behaviors)
            combined_frame = cv2.addWeighted(frame_with_density, 0.7, frame_with_boxes, 0.3, 0)
            
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            
            # Update information displays
            total_people, max_density = self.update_density_info(density_map)
            behavior_counts = self.count_behaviors(behaviors)
            self.update_behavior_info(behavior_counts)
            self.update_anomaly_info(anomalies)
            
            # Log data
            self.log_data(total_people, max_density, behavior_counts, anomalies)
            self.update_historical_plot()
        else:
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
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
            behavior = behaviors.get(obj_id, "Unknown")
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
        return total_density, max_density
    
    def count_behaviors(self, behaviors):
        behavior_counts = {"Running": 0, "Walking": 0, "Standing": 0}
        for behavior in behaviors.values():
            if behavior in behavior_counts:
                behavior_counts[behavior] += 1
        return behavior_counts
    
    def update_behavior_info(self, behavior_counts):
        behavior_text = "\n".join([f"{k}: {v}" for k, v in behavior_counts.items()])
        self.behavior_label.config(text=f"Behavior Statistics:\n{behavior_text}")
    
    def log_data(self, total_people, max_density, behavior_counts, anomalies):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                timestamp,
                total_people,
                max_density,
                behavior_counts['Running'],
                behavior_counts['Walking'],
                behavior_counts['Standing'],
                '; '.join(anomalies) if anomalies else 'None'
            ])
    
    def update_anomaly_info(self, anomalies):
        if anomalies:
            anomaly_text = "\n".join(anomalies)
            self.anomaly_label.config(text=f"Anomalies:\n{anomaly_text}", foreground="red")
        else:
            self.anomaly_label.config(text="No anomalies detected", foreground="green")
    
    def update_historical_plot(self):
        data = self.read_log_data()
        if len(data) > 1:
            timestamps = [datetime.strptime(row['Timestamp'], "%Y-%m-%d %H:%M:%S") for row in data]
            total_people = [int(row['Total People']) for row in data]
            
            self.ax.clear()
            self.ax.plot(timestamps, total_people)
            self.ax.set_title('Total People Over Time')
            self.ax.set_xlabel('Time')
            self.ax.set_ylabel('Total People')
            self.ax.tick_params(axis='x', rotation=45)
            self.fig.tight_layout()
            self.canvas_plot.draw()
    
    def read_log_data(self):
        with open(self.log_file, 'r') as file:
            reader = csv.DictReader(file)
            return list(reader)[-100:]  # Return last 100 entries for plotting
    
    def on_closing(self):
        self.vid.release()
        self.window.destroy()

def main():
    root = tk.Tk()
    app = SurveillanceDashboard(root, video_source='dataset1.mov')
    root.mainloop()

if __name__ == "__main__":
    main()