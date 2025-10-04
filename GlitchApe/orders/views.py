from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
from .models import Order
from .serializers import OrderSerializer

@login_required
def order_history(request):
    """Server-rendered view (optional for admin/internal use)."""
    orders = Order.objects.filter(user=request.user)
    return render(request, "orders/order_history.html", {"orders": orders})

@method_decorator(login_required, name="dispatch")
class UserOrdersView(View):
    """API endpoint for frontend → JSON response"""

    def get(self, request):
        orders = Order.objects.filter(user=request.user).select_related("product")
        serializer = OrderSerializer(orders, many=True)
        return JsonResponse(serializer.data, safe=False)