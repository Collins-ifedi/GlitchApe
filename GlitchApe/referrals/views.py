from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from django.views import View
from .models import Referral, ReferralPayout
from .serializers import ReferralSerializer, ReferralPayoutSerializer

@method_decorator(login_required, name="dispatch")
class ReferralStatsView(View):
    """GET /api/referrals/stats/ → referral stats for logged-in user"""

    def get(self, request):
        referrals = Referral.objects.filter(referrer=request.user)
        payouts = ReferralPayout.objects.filter(referrer=request.user)

        stats = {
            "referral_code": str(request.user.id),
            "total_referrals": referrals.count(),
            "total_earnings": sum(p.commission_amount for p in payouts if p.status == "paid"),
            "pending_payouts": sum(p.commission_amount for p in payouts if p.status != "paid"),
        }
        return JsonResponse(stats)

@method_decorator(login_required, name="dispatch")
class ReferralPayoutsView(View):
    """GET /api/referrals/payouts/ → payout history"""

    def get(self, request):
        payouts = ReferralPayout.objects.filter(referrer=request.user).select_related("order")
        serializer = ReferralPayoutSerializer(payouts, many=True)
        return JsonResponse(serializer.data, safe=False)