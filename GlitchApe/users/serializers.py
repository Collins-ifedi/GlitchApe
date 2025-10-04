from rest_framework import serializers
from django.contrib.auth import authenticate
from .models import User

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ["id", "username", "email", "stripe_account_id", "referred_by"]

class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, min_length=8)
    referral_code = serializers.CharField(write_only=True, required=False)

    class Meta:
        model = User
        fields = ["username", "email", "password", "referral_code"]

    def create(self, validated_data):
        referral_code = validated_data.pop("referral_code", None)
        password = validated_data.pop("password")
        user = User(**validated_data)
        user.set_password(password)

        if referral_code:
            referrer = User.objects.filter(id=referral_code).first()
            if referrer:
                user.referred_by = referrer

        user.save()
        return user

class LoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True)

    def validate(self, data):
        email = data.get("email")
        password = data.get("password")
        user = authenticate(username=email, password=password)
        if not user:
            raise serializers.ValidationError("Invalid credentials")
        data["user"] = user
        return data