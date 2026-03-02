from django.urls import path
from detection.views import verify_vehicle_api

urlpatterns = [
    path("verify/", verify_vehicle_api, name="verify_vehicle_api"),
]
from django.urls import path
from .views import live_ai_view

urlpatterns = [
    path('live/<str:filename>/', live_ai_view, name='live_ai_view'),
]
