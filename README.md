# Intelligent Video Surveillance System

## Overview
This project implements an advanced Intelligent Video Surveillance System using computer vision and AI techniques. It provides real-time object detection, tracking, behavior analysis, and anomaly detection capabilities for video surveillance applications.

## Features
- Real-time object detection using YOLOv8
- Object tracking across video frames
- Crowd density estimation
- Behavior analysis (Running, Walking, Standing)
- Basic anomaly detection
- Multi-camera support
- User-friendly graphical interface
- Data logging and visualization

## Components
1. `intelligent_video_surveillance.py`: Core functions for video processing and analysis
2. `surveillance_dashboard_with_logging.py`: Main application with GUI and logging capabilities

## Requirements
- Python 3.8+
- OpenCV
- NumPy
- Ultralytics YOLOv8
- Tkinter
- Matplotlib
- Pillow (PIL)

## Setup
1. Clone the repository:
   ```
   git clone https://github.com/dinesh3456/crowd_analysis_system.git
   cd crowd_analysis_system
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install opencv-python numpy ultralytics tkinter matplotlib pillow
   ```

4. Download the YOLOv8 weights file and place it in the project directory.

## Usage
1. Run the main application:
   ```
   python surveillance_dashboard_with_logging.py
   ```

2. The application will open, displaying the video feed and analysis results.

3. To use your own video source, modify the `video_source` parameter in the `main()` function of `surveillance_dashboard_with_logging.py`.

## System Components
- **Object Detection**: Uses YOLOv8 to detect people in each frame.
- **Object Tracking**: Implements a simple tracking algorithm to maintain object identities across frames.
- **Crowd Density Estimation**: Divides the frame into a grid and estimates crowd density in each cell.
- **Behavior Analysis**: Analyzes movement patterns to classify behaviors as Running, Walking, or Standing.
- **Anomaly Detection**: Identifies unusual patterns in crowd density or behavior.
- **User Interface**: Displays video feed, density map, behavior statistics, and anomaly alerts.
- **Data Logging**: Records key statistics and events to a CSV file for further analysis.

## Future Enhancements
- Advanced AI model training and optimization
- More sophisticated data analysis and pattern recognition
- Enhanced performance optimization for large-scale deployments
- Integration with external systems and infrastructure
- Privacy-preserving features (e.g., face blurring)

