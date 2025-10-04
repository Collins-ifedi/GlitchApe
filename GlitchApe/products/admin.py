from django.contrib import admin
from .models import Product

@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "category", "created_at")
    list_filter = ("category", "created_at")
    search_fields = ("name",)
    ordering = ("-created_at",)
