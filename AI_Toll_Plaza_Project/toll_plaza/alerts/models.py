from django.db import models


class AlertEvidence(models.Model):
    plate_number = models.CharField(max_length=20)

    detected_color = models.CharField(max_length=30)
    database_color = models.CharField(max_length=30, null=True, blank=True)

    detected_vehicle_type = models.CharField(
        max_length=30, null=True, blank=True
    )
    database_vehicle_type = models.CharField(
        max_length=30, null=True, blank=True
    )

    alert_type = models.CharField(max_length=50)
    detected_time = models.DateTimeField(auto_now_add=True)

    image = models.ImageField(
        upload_to='alert_images/', null=True, blank=True
    )

    def __str__(self):
        return f"{self.plate_number} - {self.alert_type}"


class UploadedVideo(models.Model):
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('PROCESSING', 'Processing'),
        ('DONE', 'Done'),
        ('ERROR', 'Error'),
    ]

    video = models.FileField(upload_to='videos/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='PENDING'
    )

    def __str__(self):
        return f"Video {self.id} - {self.status}"
