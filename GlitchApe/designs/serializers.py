from rest_framework import serializers
from .models import Design

class DesignSerializer(serializers.ModelSerializer):
    class Meta:
        model = Design
        fields = ["id", "user", "product_id", "design_url", "prompt", "mockup_urls", "created_at"]
        read_only_fields = ["id", "user", "created_at"]