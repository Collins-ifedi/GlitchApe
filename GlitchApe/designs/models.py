# designs/models.py
from django.db import models
from users.models import User

class Design(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="designs")
    product_id = models.IntegerField(null=True, blank=True)  # keep optional for backward compatibility
    design_url = models.URLField()  # main preview image
    prompt = models.TextField(blank=True, null=True)  # refined prompt (LLaMA)
    mockup_urls = models.JSONField(default=dict, blank=True)  # multiple sections: {"front": "...", "back": "..."}
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Design {self.id} by {self.user.email}"