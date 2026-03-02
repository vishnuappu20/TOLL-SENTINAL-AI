from django.urls import path
from .views import CustomLoginView
from . import views

urlpatterns = [
    path('', CustomLoginView.as_view(), name='login'),   # 👈 THIS IS IMPORTANT
    path('login/', CustomLoginView.as_view(), name='login'),
   
]