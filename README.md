# yolo-site-guard-automation
"Automated computer vision pipeline for construction site safety and PPE monitoring."
@'
# Construction Site PPE Watcher 🏗️
An MLOps project to automatically detect workers and safety gear on-site.

## Features
- **Real-time Monitoring:** Automatically scans the `data/` folder for new images.
- **AI Detection:** Uses YOLOv11 for high-speed object detection.
- **Safety Logging:** Records all detections into a `safety_log.csv` for audit.

## Setup
1. `python -m venv venv`
2. `.\venv\Scripts\Activate.ps1`
3. `pip install ultralytics watchdog pandas`
4. `python src/monitor.py`
'@ | Out-File -FilePath README.md -Encoding utf8

