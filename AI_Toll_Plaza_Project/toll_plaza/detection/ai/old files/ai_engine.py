from ultralytics import YOLO
import easyocr
import cv2
import socket
import pickle
print("loading yolo...")
model=YOLO("yolov8.pt")
print("loading ocr")
reader=easyocr.Reader(['en'])
print("ai engine ready")
server=socket.socket()
server.bind(("localhost",9000))
server.listen(1)
while True:
    conn, =server.accept()
    video_path=conn.recv(1024).decode()
    cap=cv2.VideoCapture(video_path)
    
