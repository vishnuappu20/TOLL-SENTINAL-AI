import cv2
import re
import torch
import easyocr
import mysql.connector
import smtplib
from email.mime.text import MIMEText
from ultralytics import YOLO
from datetime import datetime

COOLDOWN = 10
last_detected = {}

vehicle_model = YOLO("yolov8n.pt")
plate_model = YOLO("plate_model.pt")
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

EMAIL_ADDRESS = "pdspersonal7@gmail.com"
EMAIL_PASSWORD = "wyqr upjt ydoe wkrw"
ALERT_RECEIVER = "parth123dlps@gmail.com"

def send_email(subject, message):
    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = ALERT_RECEIVER

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)

def get_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="toll_sentinel_ai",
        autocommit=True
    )

def clean_plate(text):
    return re.sub('[^A-Z0-9]', '', text.upper())

def valid_plate(text):
    pattern = r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$'
    return re.match(pattern, text)

def detect_color(image):
    import numpy as np
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    avg = np.mean(hsv, axis=(0,1))
    h,s,v = avg
    if v > 200 and s < 40:
        return "White"
    if v < 60:
        return "Black"
    if 90 < h < 130:
        return "Blue"
    return "Grey"

def process_frame(frame):

    db = get_db()
    cursor = db.cursor()

    results = vehicle_model(frame, conf=0.5, verbose=False)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = vehicle_model.names[cls_id]

            if label not in ["car","truck","bus","motorcycle"]:
                continue

            x1,y1,x2,y2 = map(int, box.xyxy[0])
            vehicle_crop = frame[y1:y2,x1:x2]
            detected_color = detect_color(vehicle_crop)

            plate_results = plate_model(vehicle_crop, conf=0.5, verbose=False)

            for pr in plate_results:
                for pbox in pr.boxes:
                    px1,py1,px2,py2 = map(int, pbox.xyxy[0])
                    plate_crop = vehicle_crop[py1:py2,px1:px2]

                    ocr = reader.readtext(plate_crop)
                    if not ocr:
                        continue

                    plate = clean_plate(ocr[0][1])

                    if not valid_plate(plate):
                        continue

                    now = datetime.now()

                    if plate in last_detected:
                        diff = (now-last_detected[plate]).seconds
                        if diff < COOLDOWN:
                            continue

                    last_detected[plate] = now

                    cursor.execute("""
                        INSERT INTO detected_vehicle
                        (vehicle_reg_no, detected_colour, detected_type, lane_no, toll_id)
                        VALUES (%s,%s,%s,%s,%s)
                    """,(plate,detected_color,label,1,1))

                    cursor.execute("""
                        SELECT vehicle_colour, vehicle_type, vehicle_status,
                               owner_name, owner_contact_no
                        FROM registered_vehicle_data
                        WHERE vehicle_reg_no=%s
                    """,(plate,))
                    data = cursor.fetchone()

                    alert_reason=None
                    owner="Unknown"
                    contact="Unknown"

                    if data:
                        reg_color, reg_type, status, owner, contact = data
                        if status=="Stolen":
                            alert_reason="Stolen Vehicle"
                        elif reg_color.lower()!=detected_color.lower() or reg_type.lower()!=label.lower():
                            alert_reason="Number Plate Mismatch"
                    else:
                        alert_reason="Fake Number Plate"

                    if alert_reason:
                        cursor.execute("""
                            INSERT INTO alert_table
                            (vehicle_reg_no,alert_reason,toll_id,toll_location,
                             owner_name,owner_contact_no)
                            VALUES (%s,%s,%s,%s,%s,%s)
                        """,(plate,alert_reason,1,"Ernakulam",owner,contact))

                        send_email(
                            "🚨 Toll Alert",
                            f"Vehicle: {plate}\nReason: {alert_reason}"
                        )

    cursor.close()
    db.close()