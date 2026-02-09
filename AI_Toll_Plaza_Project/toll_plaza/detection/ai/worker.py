# worker.py

import os
import time
import django

print(">>> AI WORKER STARTED <<<")

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "toll_plaza.settings")
django.setup()

from alerts.models import UploadedVideo
from detection.ai.ai_pipeline import process_video

print(">>> Models loaded once <<<")

while True:
    print(">>> Checking for pending videos <<<")

    videos = UploadedVideo.objects.filter(status="PENDING")

    if not videos.exists():
        print(">>> No pending videos <<<")
        time.sleep(10)
        continue

    for video in videos:
        print(f">>> Sending video ID {video.id} to AI pipeline <<<")

        try:
            # 🔥 IMPORTANT:
            # Pass the FULL model object, NOT video path
            process_video(video)

        except Exception as e:
            print(f">>> ERROR processing video {video.id}: {e}")
            video.status = "ERROR"
            video.save()

    time.sleep(2)
