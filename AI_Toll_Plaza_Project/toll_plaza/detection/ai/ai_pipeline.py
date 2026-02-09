# ai_pipeline.py

import os, django
import cv2, re, numpy as np
from ultralytics import YOLO
import easyocr
from collections import Counter
from django.core.files.base import ContentFile

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "toll_plaza.settings")
django.setup()

from alerts.models import UploadedVideo
from detection.utils import verify_vehicle

# ================== LOAD MODELS (ONCE) ==================
vehicle_model = YOLO("yolov8n.pt")
BASE_DIR = os.path.dirname(__file__)
plate_model = YOLO(os.path.join(BASE_DIR, "license_plate_detector.pt"))

ocr = easyocr.Reader(['en'], gpu=False)

# ================== HELPERS ==================
def clean_plate(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text if 8 <= len(text) <= 10 else None

def get_color(img):
    if img.size == 0:
        return "Unknown"
    b,g,r = np.mean(img, axis=(0,1))
    if r > 150 and g < 120: return "Red"
    if g > 150 and r < 120: return "Green"
    if b > 150: return "Blue"
    if r > 200 and g > 200: return "White"
    if r < 80 and g < 80 and b < 80: return "Black"
    return "Gray"

def center(box):
    x1,y1,x2,y2 = box
    return ((x1+x2)//2, (y1+y2)//2)

# ================== MAIN PROCESS ==================
def process_video(video_obj):

    cap = cv2.VideoCapture(video_obj.video.path)
    video_obj.status = "PROCESSING"
    video_obj.save()

    vehicles = {}
    vehicle_id = 0
    DIST_TH = 70
    VOTE_TH = 3

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = vehicle_model(frame, conf=0.4)[0]

        for box in results.boxes:
            cls = int(box.cls[0])
            vtype = vehicle_model.names[cls]
            if vtype not in ["car","bus","truck","motorcycle"]:
                continue

            x1,y1,x2,y2 = map(int, box.xyxy[0])
            c = center((x1,y1,x2,y2))

            vid = None
            for k,v in vehicles.items():
                px,py = v["center"]
                if abs(px-c[0]) < DIST_TH and abs(py-c[1]) < DIST_TH:
                    vid = k
                    break

            if vid is None:
                vehicle_id += 1
                vid = vehicle_id
                vehicles[vid] = {
                    "center": c,
                    "type": vtype,
                    "color": None,
                    "plate": None,
                    "buffer": [],
                    "done": False,
                    "reported": False
                }

            vehicles[vid]["center"] = c
            roi = frame[y1:y2, x1:x2]

            if vehicles[vid]["color"] is None:
                vehicles[vid]["color"] = get_color(roi)

            if not vehicles[vid]["done"]:
                plate_results = plate_model(roi, conf=0.45)[0]
                for pbox in plate_results.boxes:
                    px1,py1,px2,py2 = map(int, pbox.xyxy[0])
                    plate_roi = roi[py1:py2, px1:px2]
                    if plate_roi.size == 0:
                        continue

                    gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
                    ocr_res = ocr.readtext(gray)

                    for (_, txt, _) in ocr_res:
                        plate = clean_plate(txt)
                        if plate:
                            vehicles[vid]["buffer"].append(plate)

                    if len(vehicles[vid]["buffer"]) >= VOTE_TH:
                        plate, cnt = Counter(vehicles[vid]["buffer"]).most_common(1)[0]
                        if cnt >= VOTE_TH:
                            vehicles[vid]["plate"] = plate
                            vehicles[vid]["done"] = True

            if vehicles[vid]["done"] and not vehicles[vid]["reported"]:
                _, img_encoded = cv2.imencode(".jpg", roi)
                image_file = ContentFile(img_encoded.tobytes(), name=f"{vehicles[vid]['plate']}.jpg")

                verify_vehicle(
                    vehicles[vid]["type"],
                    vehicles[vid]["plate"],
                    vehicles[vid]["color"],
                    image=image_file
                )

                vehicles[vid]["reported"] = True

    cap.release()
    video_obj.status = "DONE"
    video_obj.save()
