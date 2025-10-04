from django.http import JsonResponse
from django.views import View
from .models import Product
from .serializers import ProductSerializer

class ProductListView(View):
    """GET /api/products/ → list all products"""

    def get(self, request):
        products = Product.objects.all()[:50]
        serializer = ProductSerializer(products, many=True)
        return JsonResponse(serializer.data, safe=False)

class ProductDetailView(View):
    """GET /api/products/<id>/ → product details"""

    def get(self, request, pk):
        product = Product.objects.filter(pk=pk).first()
        if not product:
            return JsonResponse({"error": "Product not found"}, status=404)
        serializer = ProductSerializer(product)
        return JsonResponse(serializer.data, safe=False)