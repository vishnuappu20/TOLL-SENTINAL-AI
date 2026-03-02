from flask import Flask, render_template, request, redirect, session
import os
import cv2
from ai_engine import process_frame
import mysql.connector

app = Flask(__name__)
app.secret_key = "secret"

UPLOAD_FOLDER="uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def get_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="toll_sentinel_ai"
    )

@app.route("/",methods=["GET","POST"])
def login():
    if request.method=="POST":
        username=request.form["username"]
        password=request.form["password"]

        db=get_db()
        cursor=db.cursor()
        cursor.execute("SELECT role FROM users WHERE username=%s AND password=%s",(username,password))
        user=cursor.fetchone()
        cursor.close()
        db.close()

        if user:
            session["role"]=user[0]
            if user[0]=="admin":
                return redirect("/admin")
            else:
                return redirect("/operator")

    return render_template("login.html")

@app.route("/operator")
def operator():
    return render_template("operator_dashboard.html")

@app.route("/admin")
def admin():
    db=get_db()
    cursor=db.cursor(dictionary=True)

    cursor.execute("SELECT * FROM registered_vehicle_data")
    vehicles=cursor.fetchall()

    cursor.execute("SELECT * FROM detected_vehicle ORDER BY detection_time DESC")
    detected=cursor.fetchall()

    cursor.execute("SELECT * FROM alert_table ORDER BY alert_time DESC")
    alerts=cursor.fetchall()

    cursor.close()
    db.close()

    return render_template("admin_dashboard.html",
                           vehicles=vehicles,
                           detected=detected,
                           alerts=alerts)

@app.route("/upload",methods=["GET","POST"])
def upload():
    if request.method=="POST":
        file=request.files["video"]
        path=os.path.join(UPLOAD_FOLDER,file.filename)
        file.save(path)

        cap=cv2.VideoCapture(path)

        while True:
            ret,frame=cap.read()
            if not ret:
                break
            process_frame(frame)

        cap.release()

        return render_template("upload.html",message="Processing Completed")

    return render_template("upload.html")

@app.route("/alerts")
def alerts():
    db=get_db()
    cursor=db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM alert_table ORDER BY alert_time DESC")
    alerts=cursor.fetchall()
    cursor.close()
    db.close()
    return render_template("alerts.html",alerts=alerts)

if __name__=="__main__":
    app.run(debug=True)