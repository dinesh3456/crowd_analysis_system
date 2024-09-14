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

class CameraFeed:
    def __init__(self, video_source):
        self.video_source = video_source
        self.vid = cv2.VideoCapture(self.video_source)
        self.tracker = SimpleTracker()

class MultiCameraSurveillanceDashboard:
    def __init__(self, window, video_sources):
        self.window = window
        self.window.title("Multi-Camera Intelligent Video Surveillance Dashboard")
        
        self.camera_feeds = [CameraFeed(source) for source in video_sources]
        
        self.notebook = ttk.Notebook(window)
        self.notebook.pack(side=tk.LEFT, padx=10, pady=10, expand=True, fill=tk.BOTH)
        
        self.camera_tabs = []
        for i, feed in enumerate(self.camera_feeds):
            tab = ttk.Frame(self.notebook)
            self.notebook.add(tab, text=f"Camera {i+1}")
            canvas = tk.Canvas(tab, width=640, height=480)
            canvas.pack(side=tk.LEFT, padx=10, pady=10)
            self.camera_tabs.append((tab, canvas))
        
        self.sidebar = ttk.Frame(window, padding="10")
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y, expand=False)
        
        self.total_density_label = ttk.Label(self.sidebar, text="Total Crowd Density:")
        self.total_density_label.pack(pady=5)
        
        self.total_behavior_label = ttk.Label(self.sidebar, text="Total Behavior Statistics:")
        self.total_behavior_label.pack(pady=5)
        
        self.anomaly_label = ttk.Label(self.sidebar, text="Anomalies:", foreground="red")
        self.anomaly_label.pack(pady=5)
        
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.sidebar)
        self.canvas_plot.get_tk_widget().pack(pady=10)
        
        self.log_file = 'multi_camera_surveillance_log.csv'
        self.init_log_file()

        self.is_running = True 
        
        self.update_thread = threading.Thread(target=self.update_all_cameras, daemon=True)
        self.update_thread.start()
        
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
    
    def update_all_cameras(self):
        while self.is_running:
            total_people = 0
            max_density = 0
            total_behaviors = {"Running": 0, "Walking": 0, "Standing": 0}
            all_anomalies = []
            
            for i, feed in enumerate(self.camera_feeds):
                ret, frame = feed.vid.read()
                if ret:
                    # Detect and track objects
                    detections = model(frame)[0].boxes.data.cpu().numpy()
                    persons = detections[detections[:, 5] == 0]  # Filter for persons (class 0)
                    tracked_objects = feed.tracker.update(persons)
                    
                    # Analyze behaviors
                    behaviors = analyze_behavior(feed.tracker.trajectories)
                    
                    # Estimate crowd density
                    density_map = estimate_crowd_density(frame, tracked_objects)
                    
                    # Detect anomalies
                    anomalies = detect_anomalies(density_map, behaviors)
                    
                    frame_with_density = self.draw_density_map(frame.copy(), density_map)
                    frame_with_boxes = self.draw_boxes_and_behaviors(frame, tracked_objects, behaviors)
                    
                    combined_frame = cv2.addWeighted(frame_with_density, 0.7, frame_with_boxes, 0.3, 0)
                    
                    try:
                        photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)))
                        self.camera_tabs[i][1].create_image(0, 0, image=photo, anchor=tk.NW)
                        self.camera_tabs[i][1].image = photo
                    except Exception:
                        # If an exception occurs while updating the GUI, it's likely that the window is being destroyed
                        self.is_running = False
                        return
                    
                    total_people += np.sum(density_map)
                    max_density = max(max_density, np.max(density_map))
                    
                    # Update total behaviors
                    for behavior, count in self.count_behaviors(behaviors).items():
                        total_behaviors[behavior] += count
                    
                    all_anomalies.extend(anomalies)
                
                else:
                    # Reset video to the beginning when it reaches the end
                    feed.vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            self.update_total_density_info(total_people, max_density)
            self.update_total_behavior_info(total_behaviors)
            self.update_anomaly_info(all_anomalies)
            
            self.log_data(total_people, max_density, total_behaviors, all_anomalies)
            self.update_historical_plot()
            
            self.window.update_idletasks()
    
    def count_behaviors(self, behaviors):
        behavior_counts = {"Running": 0, "Walking": 0, "Standing": 0}
        for behavior in behaviors.values():
            if behavior in behavior_counts:
                behavior_counts[behavior] += 1
        return behavior_counts
    
    def on_closing(self):
        self.is_running = False  # Set the flag to stop the update thread
        self.update_thread.join(timeout=1.0)  # Wait for the thread to finish, with a timeout
        for feed in self.camera_feeds:
            feed.vid.release()
        self.window.destroy()
    
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
    
    def update_total_density_info(self, total_people, max_density):
        density_text = f"Total People: {total_people}\nMax Density: {max_density}"
        self.total_density_label.config(text=density_text)
    
    def update_total_behavior_info(self, behaviors):
        behavior_text = "\n".join([f"{k}: {v}" for k, v in behaviors.items()])
        self.total_behavior_label.config(text=f"Total Behavior Statistics:\n{behavior_text}")
    
    def update_anomaly_info(self, anomalies):
        if anomalies:
            anomaly_text = "\n".join(anomalies)
            self.anomaly_label.config(text=f"Anomalies:\n{anomaly_text}", foreground="red")
        else:
            self.anomaly_label.config(text="No anomalies detected", foreground="green")
    
    def update_historical_plot(self):
        data = self.read_log_data()
        if len(data) > 1:  # We need at least two data points to plot
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
            return list(reader)[-100:]  
    
    def on_closing(self):
        for feed in self.camera_feeds:
            feed.vid.release()
        self.window.destroy()

def main():
    root = tk.Tk()
    video_sources = [0, 'dataset.mov', 'dataset1.mov']  # Add your video sources here
    app = MultiCameraSurveillanceDashboard(root, video_sources)
    root.mainloop()

if __name__ == "__main__":
    main()