import cv2
import numpy as np

def detect_color(image):
    if image is None or image.size == 0:
        return "Unknown"

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = np.mean(hsv[:,:,0]), np.mean(hsv[:,:,1]), np.mean(hsv[:,:,2])

    if v < 50:
        return "Black"
    if s < 40 and v > 180:
        return "White"
    if h < 10 or h > 160:
        return "Red"
    if 35 < h < 85:
        return "Green"
    if 90 < h < 130:
        return "Blue"

    return "Gray"
