
from django.db import models

class VehicleMaster(models.Model):
    plate_number = models.CharField(max_length=20, unique=True)
    vehicle_type = models.CharField(max_length=50)
    registered_color = models.CharField(max_length=30)
    owner_name = models.CharField(max_length=100)
    contact_info = models.CharField(max_length=255)
    status = models.CharField(max_length=20, choices=[
        ('NORMAL', 'Normal'),
        ('STOLEN', 'Stolen')
    ], default='NORMAL')

    def __str__(self):
        return self.plate_number

from django.db import models
from django.contrib.auth.models import User

class DetectedVehicle(models.Model):

    user = models.ForeignKey(User, on_delete=models.CASCADE)

    plate_number = models.CharField(max_length=20)
    detected_color = models.CharField(max_length=30)
    detected_vehicle_type = models.CharField(max_length=30)

    detection_time = models.DateTimeField(auto_now_add=True)

    toll_id = models.CharField(max_length=50)
    toll_name = models.CharField(max_length=100)
    lane_number = models.CharField(max_length=20)



    def __str__(self):
        return f"{self.plate_number} - {self.toll_name}"