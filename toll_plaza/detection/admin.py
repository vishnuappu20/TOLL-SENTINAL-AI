from django.contrib import admin
from .models import VehicleMaster,DetectedVehicle


@admin.register(VehicleMaster)
class VehicleMasterAdmin(admin.ModelAdmin):
    list_display = (
        "plate_number",
        "registered_color",
        "vehicle_type",
        "owner_name",
        "contact_info",
        "status",
    )

    # 👇 THIS MAKES THE ROW CLICKABLE VIA PLATE NUMBER
    list_display_links = ("plate_number",)

    search_fields = ("plate_number",)
    list_filter = ("vehicle_type", "status", "registered_color")
@admin.register(DetectedVehicle)
class DetectedVehicle(admin.ModelAdmin):
    list_display = (
     
       "user",
       "plate_number",
       "detected_color" ,
       "detected_vehicle_type",
        "detection_time",
        "toll_id" ,
        "toll_name",
        "lane_number" ,)

    # 👇 THIS MAKES THE ROW CLICKABLE VIA PLATE NUMBER
    list_display_links = ("plate_number",)
