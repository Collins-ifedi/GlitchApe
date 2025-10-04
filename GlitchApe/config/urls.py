from django.contrib import admin
from django.urls import path, include
from payments.views import stripe_webhook  # will be created

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("api.urls")),
    path("stripe/webhook/", stripe_webhook, name="stripe-webhook"),
    path("", include("core.urls")),
]
