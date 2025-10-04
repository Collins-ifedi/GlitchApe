# account/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("login/", views.api_login, name="api_login"),
    path("signup/", views.api_signup, name="api_signup"),
    path("logout/", views.api_logout, name="api_logout"),
    path("profile/", views.api_profile, name="api_profile"),
]