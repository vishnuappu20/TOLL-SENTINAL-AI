
from django.db import models

class VehicleMaster(models.Model):
    plate_number = models.CharField(max_length=20, unique=True)
    vehicle_type = models.CharField(max_length=50)
    registered_color = models.CharField(max_length=30)
    owner_name = models.CharField(max_length=100)
    status = models.CharField(max_length=20, choices=[
        ('NORMAL', 'Normal'),
        ('STOLEN', 'Stolen')
    ], default='NORMAL')

    def __str__(self):
        return self.plate_number


# Create your models here.
