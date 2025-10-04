from rest_framework import serializers
from .models import Referral, ReferralPayout

class ReferralSerializer(serializers.ModelSerializer):
    class Meta:
        model = Referral
        fields = ["id", "referrer", "referred", "created_at"]

class ReferralPayoutSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReferralPayout
        fields = ["id", "order", "commission_amount", "status", "created_at"]