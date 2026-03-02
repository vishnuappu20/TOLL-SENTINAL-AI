from ultralytics import YOLO

# Load once
model = YOLO("yolov8n.pt")

VEHICLE_CLASSES = ["car", "bus", "truck", "motorcycle"]

def detect_vehicles(frame):
    boxes_out = []

    results = model(frame, verbose=False)
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label in VEHICLE_CLASSES and box.conf[0] > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes_out.append((x1, y1, x2, y2))

    return boxes_out
