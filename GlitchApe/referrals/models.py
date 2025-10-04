from django.db import models
from users.models import User
from orders.models import Order

class Referral(models.Model):
    referrer = models.ForeignKey(User, on_delete=models.CASCADE, related_name="referrals_made")
    referred_user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="referral")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.referrer.email} referred {self.referred_user.email}"

class ReferralPayout(models.Model):
    referrer = models.ForeignKey(User, on_delete=models.CASCADE, related_name="referral_payouts")
    order = models.ForeignKey(Order, on_delete=models.CASCADE, related_name="referral_payouts")
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    status = models.CharField(max_length=20, default="pending")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Payout {self.id} to {self.referrer.email}"
