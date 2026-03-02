import cv2
from .vehicle_detection import detect_vehicles
from .plate_ocr import read_plate
from .color_detection import detect_color
from .send_to_backend import send_to_backend

FRAME_SKIP = 15

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        vehicles = detect_vehicles(frame)

        for (x1, y1, x2, y2) in vehicles:
            roi = frame[y1:y2, x1:x2]

            color = detect_color(roi)
            plate = read_plate(roi)

            if plate is None:
                plate = "UNKNOWN"

            result = send_to_backend(plate, color)
            print("Sent:", plate, color, result)

    cap.release()
