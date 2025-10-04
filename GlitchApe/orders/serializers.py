from rest_framework import serializers
from .models import Order
from products.models import Product

class ProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = Product
        fields = ["id", "name", "base_price", "category"]

class OrderSerializer(serializers.ModelSerializer):
    product = ProductSerializer(read_only=True)

    class Meta:
        model = Order
        fields = [
            "id",
            "created_at",
            "status",
            "total_price",
            "shipping_address",
            "product",
        ]