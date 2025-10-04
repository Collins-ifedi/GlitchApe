import os
import json
import stripe
from decimal import Decimal
from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.db import transaction
from django.contrib.auth.decorators import login_required

from users.models import User
from orders.models import Order
from .models import Payment
from referrals.models import ReferralPayout

stripe.api_key = getattr(settings, "STRIPE_SECRET_KEY", os.getenv("STRIPE_SECRET_KEY"))


@csrf_exempt
@require_POST
def create_checkout_session(request):
    """Create Stripe checkout session"""
    data = json.loads(request.body.decode("utf-8") or "{}")
    amount = int((data.get("amount", 0)) * 100)  # cents
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{
                "price_data": {
                    "currency": "usd",
                    "product_data": {"name": "GlitchApe order"},
                    "unit_amount": amount,
                },
                "quantity": 1,
            }],
            mode="payment",
            success_url=data.get("success_url") or "https://example.com/success",
            cancel_url=data.get("cancel_url") or "https://example.com/cancel",
        )
        return JsonResponse({"sessionId": session["id"]})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)


@csrf_exempt
def stripe_webhook(request):
    """Handle Stripe webhook events"""
    payload = request.body
    sig_header = request.META.get("HTTP_STRIPE_SIGNATURE", "")
    endpoint_secret = getattr(settings, "STRIPE_WEBHOOK_SECRET", None)
    if not endpoint_secret:
        return HttpResponse(status=500)

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
    except ValueError:
        return HttpResponse(status=400)
    except stripe.error.SignatureVerificationError:
        return HttpResponse(status=400)

    event_type = event["type"]
    data = event["data"]["object"]

    try:
        with transaction.atomic():
            if event_type == "checkout.session.completed":
                session = data
                stripe_pi = session.get("payment_intent") or session.get("id")
                amount_total = session.get("amount_total") or 0
                email = session.get("customer_email")
                amount = Decimal(int(amount_total or 0) / 100)

                user = User.objects.filter(email=email).first() if email else None

                payment = Payment.objects.filter(stripe_payment_id=stripe_pi).first()
                if not payment:
                    Payment.objects.create(
                        user=user,
                        amount=amount,
                        stripe_payment_id=stripe_pi,
                        status="succeeded",
                    )
            elif event_type == "payment_intent.payment_failed":
                pass
            elif event_type in ("charge.refunded", "charge.refund.updated"):
                pass
    except Exception:
        return HttpResponse(status=500)

    return HttpResponse(status=200)


@login_required
def user_payments(request):
    """Return logged-in user's payments"""
    payments = Payment.objects.filter(user=request.user).order_by("-created_at")
    data = [
        {
            "id": str(p.id),
            "amount": float(p.amount),
            "status": p.status,
            "created_at": p.created_at.isoformat(),
        }
        for p in payments
    ]
    return JsonResponse({"payments": data})