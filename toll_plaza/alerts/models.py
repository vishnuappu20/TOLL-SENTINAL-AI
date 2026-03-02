from django.db import models
from detection.models import DetectedVehicle


class AlertEvidence(models.Model):

    detected_vehicle = models.ForeignKey(
        DetectedVehicle,
        on_delete=models.CASCADE
    )

    plate_number = models.CharField(max_length=20)

    alert_reason = models.CharField(max_length=100)

    alert_time = models.DateTimeField(auto_now_add=True)

    toll_id = models.CharField(max_length=50)
    toll_name = models.CharField(max_length=100)
    lane_number = models.CharField(max_length=20)

    owner_name = models.CharField(max_length=100, null=True, blank=True)
    owner_contact = models.CharField(max_length=100, null=True, blank=True)

    image = models.ImageField(upload_to='alert_images/', null=True, blank=True)

    def __str__(self):
        return f"{self.plate_number} - {self.alert_reason}"
    
from django.contrib.auth import get_user_model
User = get_user_model()

class UploadedVideo(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    video = models.FileField(upload_to="temp_videos/")
    processed = models.BooleanField(default=False)
    uploaded_at = models.DateTimeField(auto_now_add=True)