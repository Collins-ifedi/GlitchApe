# designs/views.py
import os
import requests
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator
from django.views import View
from .models import Design
from .serializers import DesignSerializer

# -------------------------------
# API keys
# -------------------------------
LLAMA_API_URL = os.getenv("LLAMA_API_URL")
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PRINTFUL_API_KEY = os.getenv("PRINTFUL_API_KEY")

# -------------------------------
# LLaMA prompt refinement
# -------------------------------
def refine_prompt_with_llama(prompt: str) -> str:
    if not LLAMA_API_URL or not LLAMA_API_KEY:
        return prompt
    try:
        resp = requests.post(
            LLAMA_API_URL,
            headers={"Authorization": f"Bearer {LLAMA_API_KEY}", "Content-Type": "application/json"},
            json={"prompt": prompt, "max_tokens": 150},
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("output", prompt)
    except Exception as e:
        print("LLaMA error:", e)
        return prompt

# -------------------------------
# Gemini AI image editing (Google official endpoint)
# -------------------------------
def edit_with_gemini(base_image_url: str, mask_url: str, prompt: str):
    if not GEMINI_API_KEY:
        raise Exception("Gemini API key missing")
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateImage?key={GEMINI_API_KEY}"
        resp = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json={
                "prompt": {
                    "text": prompt
                },
                "image": {
                    "baseImage": base_image_url,
                    "maskImage": mask_url or None
                },
                "parameters": {
                    "size": "1024x1024",
                    "n": 1
                }
            },
            timeout=60
        )
        resp.raise_for_status()
        data = resp.json()
        # The response format from Gemini might differ; adjust as needed
        return data.get("candidates", [{}])[0].get("content", [{}])[0].get("url")
    except Exception as e:
        raise Exception(f"Gemini editing error: {e}")

# -------------------------------
# Printful mockup fetch
# -------------------------------
def get_printful_mockups(product_id: int):
    """
    Fetch mockup templates from Printful API for a product.
    Returns dict with base image URLs and masks for all placements.
    """
    headers = {"Authorization": f"Bearer {PRINTFUL_API_KEY}", "Content-Type": "application/json"}
    try:
        resp = requests.get(
            f"https://api.printful.com/mockup-generator/templates/{product_id}",
            headers=headers,
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json().get("result", [])

        mockups = {}
        for item in data:
            placement = item.get("placement")  # front, back, left, right, collar, etc.
            if placement:
                mockups[placement] = {
                    "base": item.get("mockup_url"),
                    "mask": item.get("mask_url")
                }
        return mockups
    except Exception as e:
        print("Printful error:", e)
        return {}

# -------------------------------
# Dynamic per-section prompts
# -------------------------------
def generate_section_prompts_dynamic(refined_prompt: str, placements: list) -> dict:
    section_prompts = {}
    for placement in placements:
        section_prompts[placement] = (
            f"Design the {placement} of this apparel: {refined_prompt}. "
            "Make it high quality, print-ready, and stylish."
        )
    return section_prompts

# -------------------------------
# API Endpoint: Generate Design
# -------------------------------
@csrf_exempt
@login_required
@require_http_methods(["POST"])
def generate_design(request):
    """
    POST /api/designs/generate/
    Body: { "prompt": "...", "product_id": 1 }
    """
    try:
        body = json.loads(request.body)
        prompt = body.get("prompt")
        product_id = body.get("product_id")

        if not prompt or not product_id:
            return JsonResponse({"error": "Missing prompt or product_id"}, status=400)

        # 1️⃣ Refine prompt
        refined_prompt = refine_prompt_with_llama(prompt)

        # 2️⃣ Get Printful templates
        templates = get_printful_mockups(product_id)
        if not templates:
            return JsonResponse({"error": "No mockup templates available from Printful"}, status=400)

        placements = list(templates.keys())
        section_prompts = generate_section_prompts_dynamic(refined_prompt, placements)

        # 3️⃣ Generate edited mockups
        edited_urls = {}
        for section, urls in templates.items():
            if not urls.get("base"):
                continue
            prompt_for_section = section_prompts.get(section, refined_prompt)
            edited_urls[section] = edit_with_gemini(
                base_image_url=urls["base"],
                mask_url=urls.get("mask", ""),
                prompt=prompt_for_section
            )

        if not edited_urls:
            return JsonResponse({"error": "Failed to generate mockups"}, status=500)

        # 4️⃣ Save in DB
        design = Design.objects.create(
            user=request.user,
            product_id=product_id,
            design_url=edited_urls.get("front") or next(iter(edited_urls.values())),
            prompt=refined_prompt,
            mockup_urls=edited_urls
        )
        serializer = DesignSerializer(design)

        return JsonResponse({
            "success": True,
            "design": serializer.data,
            "mockup_urls": edited_urls
        }, status=201)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

# -------------------------------
# API Endpoint: List + Save Designs
# -------------------------------
@method_decorator(login_required, name="dispatch")
class DesignListView(View):
    def get(self, request):
        designs = Design.objects.filter(user=request.user).order_by("-created_at")
        serializer = DesignSerializer(designs, many=True)
        return JsonResponse(serializer.data, safe=False)

    def post(self, request):
        try:
            body = json.loads(request.body)
            product_id = body.get("product_id")
            design_url = body.get("design_url")
            prompt = body.get("prompt", "")

            if not product_id or not design_url:
                return JsonResponse({"error": "Missing product_id or design_url"}, status=400)

            design = Design.objects.create(
                user=request.user,
                product_id=product_id,
                design_url=design_url,
                prompt=prompt
            )
            serializer = DesignSerializer(design)
            return JsonResponse(serializer.data, status=201)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)