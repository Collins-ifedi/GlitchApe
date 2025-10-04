from django.contrib import admin
from .models import Payment

@admin.register(Payment)
class PaymentAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "order", "stripe_payment_id", "amount", "status", "created_at")
    list_filter = ("status", "created_at")
    search_fields = ("user__email", "stripe_payment_id", "order__id")
    ordering = ("-created_at",)
    readonly_fields = ("created_at",)
