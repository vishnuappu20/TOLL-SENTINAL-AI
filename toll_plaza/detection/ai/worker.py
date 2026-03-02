import os
import sys
import time
import django
# 🔥 FIX 1: Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


print(">>> AI WORKER STARTED <<<")

# ================= DJANGO SETUP =================
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "toll_plaza.settings")
django.setup()


# ================= IMPORT AFTER DJANGO SETUP =================
# ✅ IMPORT MODELS FIRST (they cache themselves)
from detection.ai.model_loader import vehicle_model, plate_model, ocr_reader
print(">>> AI MODELS LOADED (INSTANT AFTER FIRST TIME) <<<")
from django.contrib.auth import get_user_model
from detection.ai.ai_pipeline import process_video

User = get_user_model()

print(">>> AI Models Loaded Once <<<")

# ================= CONFIG =================
WATCH_FOLDER = os.path.join(PROJECT_ROOT, "media", "temp_videos")

if not os.path.exists(WATCH_FOLDER):
    os.makedirs(WATCH_FOLDER)

# ================= MAIN LOOP =================
while True:
    try:
        files = os.listdir(WATCH_FOLDER)

        for file in files:
            if not file.endswith(".mp4"):
                continue

            video_path = os.path.join(WATCH_FOLDER, file)

            print(f"\n>>> Processing {file} <<<")

            # ================= EXTRACT USERNAME =================
            if "__" not in file:
                print("Invalid filename format. Skipping...")
                os.remove(video_path)
                continue

            username = file.split("__")[0]

            try:
                user = User.objects.get(username=username)
            except User.DoesNotExist:
                print(f"User '{username}' not found. Deleting file.")
                os.remove(video_path)
                continue

            # ================= PROCESS VIDEO =================
            try:
                process_video(video_path, user)
                print("Processing completed successfully.")
            except Exception as e:
                print("Video Processing Error:", e)

            # ================= DELETE AFTER PROCESSING =================
            try:
                os.remove(video_path)
                print(f"{file} deleted after processing.")
            except Exception as e:
                print("File deletion error:", e)

        time.sleep(5)

    except Exception as e:
        print("Worker Loop Error:", e)
        time.sleep(5)