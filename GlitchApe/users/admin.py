from django.contrib import admin
from .models import User

@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    list_display = ("id", "email", "username", "is_active", "date_joined")
    list_filter = ("is_active", "is_staff", "date_joined")
    search_fields = ("email", "username")
    ordering = ("-date_joined",)
