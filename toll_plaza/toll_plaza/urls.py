from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from accounts.views import CustomLoginView
from django.contrib.auth import views as auth_views

urlpatterns = [
    # Admin panel
    path('admin/', admin.site.urls),
    path('', include('accounts.urls')),
    path('', include('detection.urls')),

    # Login page (Default page)
    path("login/", CustomLoginView.as_view(), name="login"), 

    # Logout
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),

    # Alerts app
    path('alerts/', include('alerts.urls')),

    # Detection API
    path("api/", include("detection.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


