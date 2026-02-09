import cv2
import re
from ultralytics import YOLO
import easyocr
from collections import Counter

# ================== LOAD MODELS (ONCE) ==================
plate_model = YOLO("license_plate_detector.pt")
ocr_reader = easyocr.Reader(['en'], gpu=False)

# ================== OCR MEMORY ==================
# Keeps OCR history per vehicle ROI location
plate_buffer = []
CONFIRM_THRESHOLD = 5   # number of repeated reads required

# ================== CLEAN PLATE TEXT ==================
def clean_plate(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)

    # Typical Indian plate length
    if 8 <= len(text) <= 10:
        return text
    return None

# ================== MAIN FUNCTION ==================
def read_plate(vehicle_roi):
    """
    Input  : vehicle ROI (image)
    Output : confirmed plate string OR None
    """

    global plate_buffer

    if vehicle_roi is None or vehicle_roi.size == 0:
        return None

    # -------- Detect plate inside vehicle --------
    results = plate_model(vehicle_roi, conf=0.45)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate_roi = vehicle_roi[y1:y2, x1:x2]

        if plate_roi.size == 0:
            continue

        # -------- Preprocess for OCR --------
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # -------- OCR --------
        ocr_results = ocr_reader.readtext(thresh)

        for (_, text, conf) in ocr_results:
            plate = clean_plate(text)

            if plate:
                plate_buffer.append(plate)

    # -------- Stabilization (Voting) --------
    if len(plate_buffer) >= CONFIRM_THRESHOLD:
        plate, count = Counter(plate_buffer).most_common(1)[0]

        if count >= CONFIRM_THRESHOLD:
            plate_buffer.clear()   # reset for next vehicle
            return plate

    return None
