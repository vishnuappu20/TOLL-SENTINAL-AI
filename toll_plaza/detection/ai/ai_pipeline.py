import os
import cv2
import re
import uuid
import numpy as np
from collections import Counter
from django.core.files.base import ContentFile

from detection.ai.model_loader import vehicle_model, plate_model, ocr_reader
from detection.models import DetectedVehicle
from detection.utils import verify_vehicle


# ================= PLATE CLEANING =================

def clean_plate(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text

    

# ================= COLOR DETECTION =================

def get_color(img):
    if img.size == 0:
        return "Unknown"
    
    # Convert to HSV (OpenCV uses 0-179 for H)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = np.mean(hsv[:, :, 0]), np.mean(hsv[:, :, 1]), np.mean(hsv[:, :, 2])
    
    # Black: very low value
    if v < 30:
        return "Black"
    
    # White: high value, low saturation
    if s < 30 and v > 200:
        return "White"
    
    # Gray: low saturation (desaturated colors)
    if s < 40:
        return "Gray"
    
    # Red wraps around (0-10 or 170-179)
    if h < 10 or h > 170:
        return "Red"
    
    # Blue
    if 100 < h < 130:
        return "Blue"
    
    # Green
    if 40 < h < 80:
        return "Green"
    
    # Yellow
    if 20 < h < 40:
        return "Yellow"
    
    # Cyan
    if 80 < h < 100:
        return "Cyan"
    
    # Magenta
    if 130 < h < 160:
        return "Magenta"
    
    return "Other"


def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


# ================= MAIN PROCESS =================

def process_video(video_path, user):

    cap = cv2.VideoCapture(video_path)

    os.makedirs("media/processed", exist_ok=True)

    video_name = os.path.basename(video_path)
    name_without_ext = os.path.splitext(video_name)[0]
    unique_id = uuid.uuid4().hex[:8]

    output_path = f"media/processed/{name_without_ext}_{unique_id}.mp4"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    vehicles = {}
    vehicle_id = 0

    DIST_TH = 70
    VOTE_TH = 3

    profile = user.userprofile

    print("🚀 PROCESSING STARTED")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ================= VEHICLE DETECTION =================
        results = vehicle_model(frame, conf=0.30)[0]

        for box in results.boxes:
            cls = int(box.cls[0])
            vtype = vehicle_model.names[cls]

            if vtype not in ["car", "bus", "truck", "motorcycle"]:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if (x2 - x1) < 40 or (y2 - y1) < 40:
                continue

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            c = center((x1, y1, x2, y2))

            # ================= TRACKING =================
            vid = None
            for k, v in vehicles.items():
                px, py = v["center"]
                if abs(px - c[0]) < DIST_TH and abs(py - c[1]) < DIST_TH:
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
                    "reported": False,
                    "bbox": (x1, y1, x2, y2)
                }

            vehicles[vid]["center"] = c
            vehicles[vid]["bbox"] = (x1, y1, x2, y2)

            # ================= COLOR =================
            if vehicles[vid]["color"] is None:
                vehicles[vid]["color"] = get_color(roi)

            # ================= PLATE DETECTION =================
            if not vehicles[vid]["done"]:

                plate_results = plate_model(roi, conf=0.35)[0]

                for pbox in plate_results.boxes:
                    px1, py1, px2, py2 = map(int, pbox.xyxy[0])
                    plate_roi = roi[py1:py2, px1:px2]

                    if plate_roi.size < 100:
                        continue

                    # OCR preprocessing
                    gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
                    gray = cv2.bilateralFilter(gray, 11, 17, 17)
                    gray = cv2.adaptiveThreshold(
                         gray, 255,
                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                         cv2.THRESH_BINARY,
                         11, 2
                    )

                    ocr_res = ocr_reader.readtext(gray)

                    for (_, txt, conf) in ocr_res:
                        if conf > 0.45:
                            plate = clean_plate(txt)
                            if plate:
                                vehicles[vid]["buffer"].append(plate)

                # ================= VOTING =================
                if len(vehicles[vid]["buffer"]) >= VOTE_TH:
                    plate, cnt = Counter(vehicles[vid]["buffer"]).most_common(1)[0]
                    if cnt >= VOTE_TH:
                        vehicles[vid]["plate"] = plate
                        vehicles[vid]["done"] = True
                        print("✅ CONFIRMED:", plate)

        # ================= DRAW BOXES =================
        for vid, data in vehicles.items():

            x1, y1, x2, y2 = data["bbox"]

            box_color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            label = f"{data['type']} | {data['color']}"
            if data["plate"]:
                label += f" | {data['plate']}"

            cv2.putText(frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        box_color,
                        2)

        out.write(frame)

        # ================= DATABASE REPORT =================
        for vid in list(vehicles):
            if vehicles[vid]["done"] and not vehicles[vid]["reported"]:

                x1, y1, x2, y2 = vehicles[vid]["bbox"]
                vehicle_roi = frame[y1:y2, x1:x2]

                _, img_encoded = cv2.imencode(".jpg", vehicle_roi)
                image_file = ContentFile(
                    img_encoded.tobytes(),
                    f"{vehicles[vid]['plate']}.jpg"
                )

                detected_vehicle = DetectedVehicle.objects.create(
                    user=user,
                    plate_number=vehicles[vid]["plate"],
                    detected_color=vehicles[vid]["color"],
                    detected_vehicle_type=vehicles[vid]["type"],
                    toll_id=profile.toll_id,
                    toll_name=profile.toll_name,
                    lane_number=profile.lane_number
                )

                verify_vehicle(
                    vehicle_type=vehicles[vid]["type"],
                    plate=vehicles[vid]["plate"],
                    color=vehicles[vid]["color"],
                    image=image_file,
                    detected_vehicle=detected_vehicle
                )

                vehicles[vid]["reported"] = True

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("🎥 Output saved at:", output_path)
    return output_path