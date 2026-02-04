import time
import os
import pandas as pd
from ultralytics import YOLO
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 1. Setup the Brain
model = YOLO('yolo11n.pt')
log_file = 'data/safety_log.csv'

class SiteHandler(FileSystemEventHandler):
    def on_created(self, event):
        # Only look at images
        if event.is_directory or not event.src_path.lower().endswith(('.jpg', '.png', '.jpeg')):
            return
        
        print(f"🏗️  New Site Photo Detected: {os.path.basename(event.src_path)}")
        
        # 2. Run AI Inference
        # We give it a second to make sure Windows has finished 'writing' the file
        time.sleep(1)
        results = model.predict(source=event.src_path, save=True, conf=0.3)
        
        for r in results:
            count = len(r.boxes) 
            
            # 3. Log to CSV
            new_entry = pd.DataFrame([{
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'file': os.path.basename(event.src_path),
                'detections': count
            }])
            
            header = not os.path.exists(log_file)
            new_entry.to_csv(log_file, mode='a', index=False, header=header)
            print(f"✅ Logged {count} workers/objects to CSV.\n")

if __name__ == "__main__":
    # Make sure data folder exists
    if not os.path.exists('data'): os.makedirs('data')
    
    observer = Observer()
    observer.schedule(SiteHandler(), path='data', recursive=False)
    print("👀 SITE GUARD ACTIVE. Drop photos into the 'data' folder...")
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
