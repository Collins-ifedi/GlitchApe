from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('users/', include('users.urls')),
    path('products/', include('products.urls')),
    path('orders/', include('orders.urls')),
    path('payments/', include('payments.urls')),
    path('designs/', include('designs.urls')),
    path('referrals/', include('referrals.urls')),
    path("users/", include("account.urls")),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)