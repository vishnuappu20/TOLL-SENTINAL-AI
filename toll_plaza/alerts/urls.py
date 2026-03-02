from django.urls import path
from . import views



urlpatterns = [
    path("dashboard/", views.dashboard, name="dashboard"),
    path("upload/", views.upload_video, name="upload_video"),

]