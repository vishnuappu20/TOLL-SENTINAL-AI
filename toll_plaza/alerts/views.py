from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.conf import settings
from detection.models import DetectedVehicle
from alerts.models import AlertEvidence
import os
import uuid


# ================= DASHBOARD =================

@login_required
def dashboard(request):

    profile = request.user.userprofile

    # ================= ADMIN =================
    if profile.role == "ADMIN":
        detections = DetectedVehicle.objects.all().order_by("-detection_time")
        alerts = AlertEvidence.objects.all().order_by("-alert_time")

    # ================= OPERATOR =================
    else:
        detections = DetectedVehicle.objects.filter(
            user=request.user
        ).order_by("-detection_time")

        alerts = AlertEvidence.objects.filter(
            detected_vehicle__user=request.user
        ).order_by("-alert_time")

    return render(request, "alerts/dashboard.html", {
        "detections": detections,
        "alerts": alerts,
        "profile": profile
    })


# ================= VIDEO UPLOAD =================




@login_required
def upload_video(request):

    if request.method == "POST":
        video = request.FILES.get("video")

        if video:

            temp_folder = os.path.join(settings.MEDIA_ROOT, "temp_videos")

            if not os.path.exists(temp_folder):
                os.makedirs(temp_folder)

            unique_name = str(uuid.uuid4())

            # 🔥 Add username in filename
            filename = f"{request.user.username}__{unique_name}.mp4"

            temp_path = os.path.join(temp_folder, filename)

            with open(temp_path, "wb+") as destination:
                for chunk in video.chunks():
                    destination.write(chunk)

        return redirect("dashboard")

    return redirect("dashboard")