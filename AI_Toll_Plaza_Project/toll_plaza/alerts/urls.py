from django.urls import path
from .views import alert_dashboard, upload_video

urlpatterns = [
    path('dashboard/', alert_dashboard, name='alert_dashboard'),
    path('upload/', upload_video, name='upload_video'),
]
