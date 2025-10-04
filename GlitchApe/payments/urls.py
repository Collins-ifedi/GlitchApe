# payments/urls.py
from django.urls import path
from . import views

urlpatterns = [
    # Create Stripe checkout session
    path("checkout/", views.create_checkout_session, name="create_checkout_session"),

    # Stripe webhook for payment updates
    path("webhook/", views.stripe_webhook, name="stripe_webhook"),

    # (Optional) API endpoints for frontend to fetch user payments
    # e.g., /payments/user/ to list a user’s payments
    path("user/", views.user_payments, name="user_payments"),
]