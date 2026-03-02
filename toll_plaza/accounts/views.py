from django.shortcuts import render
from django.contrib.auth.views import LoginView
from django.urls import reverse_lazy

class CustomLoginView(LoginView):
    template_name = "login.html"

    def get_success_url(self):
        if self.request.user.is_staff:
            return reverse_lazy("admin:index")
        else:
            return reverse_lazy("dashboard")
# Create your views here.
