# detection/ai/model_loader.py
import os
from ultralytics import YOLO
print("yolo loads")
import easyocr
print("ocr loads")

print("🔄 Pre-loading models...")

# Vehicle model (auto-downloads)
vehicle_model = YOLO("yolov8n.pt")
print("✅ Vehicle model loaded")

# 🔥 CORRECT PATH TO YOUR FILE
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
plate_model_path = os.path.join(PROJECT_ROOT, "license_plate_detector.pt")

print(f"🔍 Looking for plate model at: {plate_model_path}")

if os.path.exists(plate_model_path):
    plate_model = YOLO(plate_model_path)
    print(f"✅ License plate model loaded: {plate_model_path}")
else:
    print(f"❌ License plate model NOT found at: {plate_model_path}")
    print("📁 Please put license_plate_detector.pt in project root!")
    plate_model = vehicle_model  # Use vehicle model as fallback

# OCR model (auto-downloads)
ocr_reader = easyocr.Reader(['en'], gpu=False)
print("✅ OCR model loaded")

print("🎉 ALL MODELS READY!")
__all__ = ['vehicle_model', 'plate_model', 'ocr_reader']
