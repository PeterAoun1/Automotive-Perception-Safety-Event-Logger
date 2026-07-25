# YOLO Real-Time Vehicle Speed & Distance Tracker

A robust Python application built with **Ultralytics YOLO**, **OpenCV**, and **NumPy** for real-time object tracking, speed estimation, and pairwise ground distance calculation from video feeds or live cameras.

## 🚀 Features

* **Object Detection:** Powered by Ultralytics YOLO (supports custom models and standard weights like `yolov8n.pt`).
* **Custom Centroid Tracking:** Tracks objects across frames, handles lost tracking IDs, and maintains object paths with temporal memory.
* **Accurate Speed Estimation:** 
  * Supports perspective correction via a **Homography matrix (`.npy`)** for real-world measurements.
  * Fallback scaling using a **pixels-per-meter** ratio.
* **Pairwise Distance Monitoring:** Calculates and visualizes real-world distances between tracked objects.
* **Data Export:** Automatically logs object speeds (`m/s` and `km/h`) into a structured CSV file.
* **Video Recording:** Optionally saves the annotated video output with speed overlays and distance lines.

---

## 📋 Prerequisites

Make sure you have Python 3.8+ installed. You will need the following libraries:

```bash
pip install ultralytics opencv-python numpy
