import cv2
import re
import torch
import easyocr
import mysql.connector
import smtplib
from email.mime.text import MIMEText
from ultralytics import YOLO
from datetime import datetime, timedelta

# ================= SETTINGS =================
COOLDOWN_SECONDS = 30
last_detected = {}

EMAIL_ADDRESS = "pdspersonal7@gmail.com"
EMAIL_PASSWORD = "opml whgz ozpa fsvm"
ALERT_RECEIVER = "parth123dlps@gmail.com"

# ================= LOAD MODELS =================
vehicle_model = YOLO("yolov8n.pt")
plate_model = YOLO("plate_model.pt")
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# ================= DATABASE =================
def get_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="toll_sentinel_ai",
        autocommit=True
    )

# ================= EMAIL FUNCTION =================
def send_email(subject, message):
    try:
        msg = MIMEText(message)
        msg["Subject"] = subject
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = ALERT_RECEIVER

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)

        print("Email sent successfully")

    except Exception as e:
        print("Email error:", e)

# ================= HELPERS =================
def clean_plate(text):
    text = re.sub('[^A-Z0-9]', '', text.upper())
    return text.strip()

def valid_plate(text):
    # Strict Indian format: KL07AB1234
    pattern = r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$'
    return re.match(pattern, text)

def detect_color(image):
    import numpy as np
    import cv2

    if image is None or image.size == 0:
        return "Unknown"

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h_mean = np.mean(hsv[:, :, 0])
    s_mean = np.mean(hsv[:, :, 1])
    v_mean = np.mean(hsv[:, :, 2])

    # ---- BLACK ----
    if v_mean < 60:
        return "Black"

    # ---- GREY / SILVER ----
    if s_mean < 40 and 60 < v_mean < 200:
        return "Grey"

    # ---- WHITE ----
    if s_mean < 25 and v_mean > 200:
        return "White"

    # ---- RED ----
    if h_mean < 10 or h_mean > 170:
        return "Red"

    # ---- BLUE ----
    if 95 < h_mean < 130 and s_mean > 60:
        return "Blue"

    # ---- GREEN ----
    if 40 < h_mean < 90:
        return "Green"

    # ---- YELLOW ----
    if 15 < h_mean < 35:
        return "Yellow"

    return "Grey"
# ================= MAIN PROCESS =================
def process_frame(frame, lane_no=1, toll_id=1, location="Ernakulam"):

    db = get_db()
    cursor = db.cursor()

    results = vehicle_model(frame, conf=0.5, verbose=False)

    for r in results:
        for box in r.boxes:

            cls_id = int(box.cls[0])
            label = vehicle_model.names[cls_id]

            if label not in ["car", "truck", "bus", "motorcycle"]:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            vehicle_crop = frame[y1:y2, x1:x2]
            h, w = vehicle_crop.shape[:2]
            center_crop = vehicle_crop[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
            detected_color = detect_color(center_crop)

            plate_results = plate_model(vehicle_crop, conf=0.5, verbose=False)

            for pr in plate_results:
                for pbox in pr.boxes:

                    px1, py1, px2, py2 = map(int, pbox.xyxy[0])
                    plate_crop = vehicle_crop[py1:py2, px1:px2]

                    if plate_crop.size == 0:
                        continue

                    ocr_result = reader.readtext(plate_crop)

                    if not ocr_result:
                        continue

                    plate = clean_plate(ocr_result[0][1])

                    if not valid_plate(plate):
                        continue

                    now = datetime.now()

                    # ================= MEMORY COOLDOWN =================
                    if plate in last_detected:
                        if (now - last_detected[plate]).seconds < COOLDOWN_SECONDS:
                            continue

                    # ================= DB DUPLICATE CHECK =================
                    cursor.execute("""
                        SELECT id FROM detected_vehicle
                        WHERE vehicle_reg_no=%s
                        AND detection_minute >= %s
                    """, (plate, now - timedelta(seconds=COOLDOWN_SECONDS)))

                    if cursor.fetchone():
                        continue

                    last_detected[plate] = now
                    minute_stamp = now.strftime("%Y-%m-%d %H:%M:%S")

                    # ================= INSERT DETECTION =================
                    try:
                        cursor.execute("""
                            INSERT INTO detected_vehicle
                            (vehicle_reg_no, detected_colour, detected_type,
                             lane_no, toll_id, detection_minute)
                            VALUES (%s,%s,%s,%s,%s,%s)
                        """, (plate, detected_color, label,
                              lane_no, toll_id, minute_stamp))

                        print("Detected:", plate)

                    except Exception as e:
                        print("Detection insert error:", e)
                        continue

                    # ================= REGISTERED CHECK =================
                    cursor.execute("""
                        SELECT vehicle_colour, vehicle_type,
                               vehicle_status, owner_name, owner_contact_no
                        FROM registered_vehicle_data
                        WHERE vehicle_reg_no=%s
                    """, (plate,))

                    data = cursor.fetchone()

                    alert_reason = None
                    owner = "Unknown"
                    contact = "Unknown"

                    if data:
                        reg_color, reg_type, status, owner, contact = data

                        if status.lower() == "stolen":
                            alert_reason = "Stolen Vehicle"

                        elif (reg_color.lower() != detected_color.lower() or
                              reg_type.lower() != label.lower()):
                            alert_reason = "Number Plate Mismatch"
                    else:
                        alert_reason = "Fake Number Plate"

                    # ================= INSERT ALERT =================
                    if alert_reason:
                        try:
                            cursor.execute("""
                                INSERT INTO alert_table
                                (vehicle_reg_no, alert_reason,
                                 toll_id, toll_location,
                                 owner_name, owner_contact_no)
                                VALUES (%s,%s,%s,%s,%s,%s)
                            """, (plate, alert_reason,
                                  toll_id, location,
                                  owner, contact))

                            send_email(
                                "🚨 Toll Sentinel Alert",
                                f"Vehicle: {plate}\n"
                                f"Reason: {alert_reason}\n"
                                f"Location: {location}"
                            )

                            print("Alert:", alert_reason)

                        except Exception as e:
                            print("Alert insert error:", e)

    cursor.close()
    db.close()