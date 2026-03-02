from django.contrib import admin
from .models import UserProfile

@admin.register(UserProfile)
class Userprofile(admin.ModelAdmin):
    list_display = (
        
    "user", 
    "role",

    "toll_id",
    "toll_name",
    "lane_number",
    )

    # 👇 THIS MAKES THE ROW CLICKABLE VIA PLATE NUMBER
    list_display_links = ("user",)