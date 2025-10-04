from django.contrib import admin
from .models import Design

@admin.register(Design)
class DesignAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "created_at")
    list_filter = ("created_at",)
    search_fields = ("user__email", "user__username")
    ordering = ("-created_at",)
