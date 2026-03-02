import requests

BACKEND_URL = "http://127.0.0.1:8000/api/verify/"

def send_to_backend(plate, color):
    payload = {
        "plate": plate,
        "color": color
    }

    try:
        res = requests.post(BACKEND_URL, json=payload, timeout=5)
        return res.json()
    except Exception as e:
        return {"error": str(e)}
