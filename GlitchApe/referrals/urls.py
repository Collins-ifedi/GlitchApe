from django.urls import path
from . import views

urlpatterns = [
    path("stats/", views.ReferralStatsView.as_view(), name="referral_stats"),
    path("payouts/", views.ReferralPayoutsView.as_view(), name="referral_payouts"),
]