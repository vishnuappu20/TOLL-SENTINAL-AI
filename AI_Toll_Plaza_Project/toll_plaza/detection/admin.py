from django.contrib import admin
from .models import VehicleMaster


@admin.register(VehicleMaster)
class VehicleMasterAdmin(admin.ModelAdmin):
    list_display = (
        "plate_number",
        "registered_color",
        "vehicle_type",
        "status",
    )

    # 👇 THIS MAKES THE ROW CLICKABLE VIA PLATE NUMBER
    list_display_links = ("plate_number",)

    search_fields = ("plate_number",)
    list_filter = ("vehicle_type", "status", "registered_color")
