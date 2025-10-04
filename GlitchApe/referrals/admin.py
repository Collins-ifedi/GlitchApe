from django.contrib import admin
from .models import Referral, ReferralPayout

@admin.register(Referral)
class ReferralAdmin(admin.ModelAdmin):
    list_display = ("id", "referrer", "referred_user", "created_at")
    list_filter = ("created_at",)
    search_fields = ("referrer__email", "referred_user__email")
    ordering = ("-created_at",)

@admin.register(ReferralPayout)
class ReferralPayoutAdmin(admin.ModelAdmin):
    list_display = ("id", "referrer", "order", "amount", "status", "created_at")
    list_filter = ("status", "created_at")
    search_fields = ("referrer__email",)
    ordering = ("-created_at",)
