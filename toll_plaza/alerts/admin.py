from django.contrib import admin
from .models import AlertEvidence, UploadedVideo


@admin.register(AlertEvidence)
class AlertEvidence(admin.ModelAdmin):
    list_display = (
        "detected_vehicle",
         "plate_number", 
         "alert_reason",

         "alert_time", 

         "toll_id",
         "toll_name",
         "lane_number", 

         "owner_name", 
         "owner_contact", 

         "image", 
    )

    # 👇 THIS MAKES THE ROW CLICKABLE VIA PLATE NUMBER
    list_display_links = ("plate_number",)

