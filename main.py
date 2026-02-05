# ================= IMPORTS =================
import cv2, sqlite3, numpy as np, re, time
from ultralytics import YOLO
import easyocr
from rapidfuzz import fuzz

# ================= VIDEO PATH =================
video_path = "C:\\Users\\parth\\OneDrive\\Desktop\\VehicleDetectionProject\\taigun.mp4"

# ================= LOAD MODELS =================
model = YOLO("yolov8n.pt")
ocr = easyocr.Reader(['en'])

# ================= DATABASE =================
conn = sqlite3.connect("vehicles.db")
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS vehicles(
plate TEXT UNIQUE,
type TEXT,
color TEXT)
""")
conn.commit()

# ================= CLEAN TEXT =================
def clean_plate(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

# ================= VALIDATE INDIAN PLATE =================
def validate_plate(text):
    text = clean_plate(text)
    pattern = r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$'
    if re.match(pattern, text):
        return text
    return "Unknown"

# ================= COLOR DETECTION =================
def get_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = np.mean(hsv[:,:,0]), np.mean(hsv[:,:,1]), np.mean(hsv[:,:,2])

    if v < 50: return "Black"
    if s < 40 and v > 180: return "White"
    if 15 < h < 35 and s > 80: return "Yellow"
    if h < 10 or h > 160: return "Red"
    if 35 < h < 85: return "Green"
    if 90 < h < 130: return "Blue"
    if s < 30 and 120 < v < 200: return "Silver"
    return "Gray"

# ================= VIDEO SETUP =================
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(3))
height = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

seen_plates = {}   # plate : (x, y, timestamp)
FRAME_GAP = 8
frame_count = 0

# ================= PROCESS VIDEO =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % FRAME_GAP != 0:
        out.write(frame)
        continue

    results = model(frame)[0]

    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]

        if label in ["car","bus","truck","motorcycle"]:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            color = get_color(crop)

            # -------- PLATE ROI --------
            h, w, _ = crop.shape
            plate_roi = crop[int(h*0.55):int(h*0.9), int(w*0.15):int(w*0.85)]

            if plate_roi.size == 0:
                continue

            plate_roi = cv2.resize(plate_roi, None, fx=2, fy=2)

            gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray,(3,3),0)
            gray = cv2.threshold(gray,120,255,cv2.THRESH_BINARY)[1]

            # SHARPEN
            kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
            gray = cv2.filter2D(gray, -1, kernel)

            # -------- OCR --------
            texts = ocr.readtext(gray)
            best_conf = 0
            best_text = ""

            for t in texts:
                txt, conf = t[1], t[2]
                if conf > best_conf:
                    best_conf = conf
                    best_text = txt

            plate = "Unknown"
            if best_conf > 0.55:
                plate = validate_plate(best_text)

            # -------- DUPLICATE CHECK --------
            duplicate = False
            now = time.time()

            if plate != "Unknown":
                for p, data in seen_plates.items():
                    px, py, ts = data
                    dist = abs(px-x1) + abs(py-y1)

                    if fuzz.ratio(p, plate) > 85 and dist < 150 and now - ts < 8:
                        duplicate = True
                        break

                if not duplicate:
                    seen_plates[plate] = (x1, y1, now)
                    cur.execute("INSERT OR IGNORE INTO vehicles VALUES(?,?,?)",
                                (plate,label,color))
                    conn.commit()

            # -------- DRAW --------
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"{label} | {plate} | {color}",
                        (x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,(0,255,0),2)

    out.write(frame)

# ================= RELEASE =================
cap.release()
out.release()
conn.close()

# ================= PRINT DATABASE =================
print("\n=== VEHICLE DATABASE ===")
conn = sqlite3.connect("vehicles.db")
for row in conn.execute("SELECT * FROM vehicles"):
    print("Plate:",row[0]," | Type:",row[1]," | Color:",row[2])
conn.close()

# ================= PLAY OUTPUT VIDEO =================
cap2 = cv2.VideoCapture("output.mp4")
while True:
    ret, frame = cap2.read()
    if not ret:
        break

    cv2.imshow("Output", frame)
    key = cv2.waitKey(35)

    if key == 32:  # SPACE to pause
        cv2.waitKey(0)
    if key & 0xFF == ord('q'):
        break

cap2.release()
cv2.destroyAllWindows()
