from flask import Response
from flask import Flask, render_template, request, redirect, session, url_for
import os
import cv2
from ai_engine import process_frame
import mysql.connector

app = Flask(__name__)
app.secret_key = "toll_sentinel_secret_key"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ================= DATABASE CONNECTION =================
def get_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="toll_sentinel_ai"
    )


# ================= LOGIN =================
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        db = get_db()
        cursor = db.cursor(dictionary=True)

        cursor.execute(
            "SELECT * FROM users WHERE username=%s AND password=%s",
            (username, password)
        )
        user = cursor.fetchone()

        cursor.close()
        db.close()

        if user:
            session["username"] = user["username"]
            session["role"] = user["role"]

            if user["role"] == "admin":
                return redirect(url_for("admin_dashboard"))
            else:
                return redirect(url_for("operator_dashboard"))

        return render_template("login.html", error="Invalid Credentials")

    return render_template("login.html")


# ================= OPERATOR DASHBOARD =================
@app.route("/operator")
def operator_dashboard():
    if "role" not in session or session["role"] != "operator":
        return redirect(url_for("login"))

    return render_template("operator_dashboard.html")


# ================= ADMIN DASHBOARD =================
@app.route("/admin")
def admin_dashboard():
    if "role" not in session or session["role"] != "admin":
        return redirect(url_for("login"))

    db = get_db()
    cursor = db.cursor(dictionary=True)

    # Registered Vehicles
    cursor.execute("SELECT * FROM registered_vehicle_data")
    vehicles = cursor.fetchall()

    # Detected Vehicles
    cursor.execute("SELECT * FROM detected_vehicle ORDER BY id DESC")
    detected = cursor.fetchall()

    # Alerts
    cursor.execute("SELECT * FROM alert_table ORDER BY id DESC")
    alerts = cursor.fetchall()

    cursor.close()
    db.close()

    return render_template(
        "admin_dashboard.html",
        vehicles=vehicles,
        detected=detected,
        alerts=alerts
    )

# ================= ADD USER =================
@app.route("/add_user", methods=["POST"])
def add_user():
    if "role" not in session or session["role"] != "admin":
        return redirect("/")

    username = request.form["username"]
    password = request.form["password"]
    role = request.form["role"]

    db = get_db()
    cursor = db.cursor()

    cursor.execute("""
        INSERT INTO users (username, password, role)
        VALUES (%s, %s, %s)
    """, (username, password, role))

    db.commit()
    cursor.close()
    db.close()

    return redirect("/admin")

# ================= ADD REGISTERED VEHICLE =================
@app.route("/add_vehicle", methods=["POST"])
def add_vehicle():
    if "role" not in session or session["role"] != "admin":
        return redirect("/")

    reg_no = request.form["vehicle_reg_no"].upper()
    colour = request.form["vehicle_colour"]
    vtype = request.form["vehicle_type"]
    status = request.form["vehicle_status"]
    owner = request.form["owner_name"]
    contact = request.form["owner_contact_no"]

    db = get_db()
    cursor = db.cursor()

    cursor.execute("""
        INSERT INTO registered_vehicle_data
        (vehicle_reg_no, vehicle_colour, vehicle_type,
         vehicle_status, owner_name, owner_contact_no)
        VALUES (%s,%s,%s,%s,%s,%s)
    """, (reg_no, colour, vtype, status, owner, contact))

    db.commit()
    cursor.close()
    db.close()

    return redirect("/admin")

# ================= DELETE REGISTERED VEHICLE =================
@app.route("/delete_vehicle/<reg_no>")
def delete_vehicle(reg_no):
    if "role" not in session or session["role"] != "admin":
        return redirect("/")

    db = get_db()
    cursor = db.cursor()

    cursor.execute("""
        DELETE FROM registered_vehicle_data
        WHERE vehicle_reg_no=%s
    """, (reg_no,))

    db.commit()
    cursor.close()
    db.close()

    return redirect("/admin")
# ================= VIDEO UPLOAD =================
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if "role" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        file = request.files.get("video")

        if not file or file.filename == "":
            return render_template("upload.html", message="No file selected")

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        cap = cv2.VideoCapture(file_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            process_frame(frame)

        cap.release()

        return render_template("upload.html", message="Processing Completed")

    return render_template("upload.html")

# ================= LIVE CAMERA PAGE =================
@app.route("/live")
def live_camera():
    if "role" not in session:
        return redirect(url_for("login"))
    return render_template("live.html")
# ================= LIVE VIDEO STREAM =================
def generate_frames():
    cap = cv2.VideoCapture(0)  # Webcam

    while True:
        success, frame = cap.read()
        if not success:
            break

        process_frame(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ================= ALERT PAGE =================
@app.route("/alerts")
def view_alerts():
    if "role" not in session:
        return redirect(url_for("login"))

    db = get_db()
    cursor = db.cursor(dictionary=True)

    cursor.execute("SELECT * FROM alert_table ORDER BY id DESC")
    alerts = cursor.fetchall()

    cursor.close()
    db.close()

    return render_template("alerts.html", alerts=alerts)


# ================= LOGOUT =================
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)