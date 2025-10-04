from django.urls import path
from . import views

urlpatterns = [
    path("history/", views.order_history, name="order_history"),
    path("user/", views.UserOrdersView.as_view(), name="user_orders"),
]