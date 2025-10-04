from django.db import models

class Product(models.Model):
    CATEGORY_CHOICES = [
        ("jacket", "Jacket"),
        ("trouser", "Trouser"),
        ("shirt", "Shirt"),
        ("polo", "Polo"),
        ("sneaker", "Sneaker"),
    ]

    name = models.CharField(max_length=255)
    slug = models.SlugField(max_length=255, unique=True, null=True, blank=True)
    category = models.CharField(max_length=50, choices=CATEGORY_CHOICES)
    base_price = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    image_url = models.URLField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name
