# designs/urls.py
from django.urls import path
from . import views

urlpatterns = [
    # AI generation endpoint (Leonardo + LLaMA)
    path("generate/", views.generate_design, name="generate_design"),

    # CRUD for designs
    path("", views.DesignListView.as_view(), name="design_list"),
]