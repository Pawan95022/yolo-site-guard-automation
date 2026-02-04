from ultralytics import YOLO
import os
import cv2

# 1. Path to your new image
img_path = r'data\site.jpg'

if not os.path.exists(img_path):
    print(f"❌ Still can't find the file at {img_path}")
    print(f"Current folder files: {os.listdir('data')}")
else:
    # 2. Load the Brain
    print("🧠 Loading AI...")
    model = YOLO('yolo11n.pt') 

    # 3. Read and Predict
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Windows is still locking the file. Close any image viewers!")
    else:
        print(f"🚀 Analyzing your generated image: {img_path}")
        results = model.predict(source=img, save=True, conf=0.25)
        print("\n" + "="*30)
        print("🎉 MISSION ACCOMPLISHED!")
        print("Go to: runs/detect/predict to see the boxes!")
        print("="*30)
