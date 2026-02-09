from django.urls import path
from detection.views import verify_vehicle_api

urlpatterns = [
    path("verify/", verify_vehicle_api, name="verify_vehicle_api"),
]
