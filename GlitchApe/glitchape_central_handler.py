# glitchape_central_handler.py
"""
Central Brain and API Orchestrator for GlitchApe.

This module replaces the individual routers from ai.py and order.py.
It provides a unified API interface and contains the central business
logic orchestrator, the 'GlitchApeCentralHandler' class.
"""

import os
import uuid
import json
import httpx
import stripe
import asyncio
import shutil
import logging
import base64 # <-- Added for image encoding
from io import BytesIO
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from fastapi import (
    APIRouter, Depends, HTTPException, Request,
    UploadFile, File, Form, status
)
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from PIL import Image

# Import core components from interface.py
from interface import (
    get_db,
    get_current_user,
    User,
    ChatSession,
    ChatMessage,
    OrderRecord,
    OrderPayment
)

# --- Logging Setup ---
log = logging.getLogger(__name__)

# ===================================================================
# CONFIGURATION
# ===================================================================

# --- LLaMA 4 & Vision Model (OpenRouter) Configuration ---
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
if not LLAMA_API_KEY:
    log.critical("LLAMA_API_KEY environment variable not set.")
    raise RuntimeError("LLAMA_API_KEY environment variable not set")
LLAMA_API_URL = "https://openrouter.ai/api/v1/chat/completions" # Same endpoint for both models
LLAMA_MODEL = "meta-llama/llama-4-maverick:free" # Text model for intent/response
VISION_MODEL = "google/gemma-3-4b-it:free" # <-- UPDATED: Free vision model via OpenRouter

# --- Public URL for external models to access images ---
GLITCHAPE_PUBLIC_URL = os.getenv("GLITCHAPE_PUBLIC_URL", "http://localhost:8000") # Replace with your Render URL in production
if GLITCHAPE_PUBLIC_URL.endswith('/'):
    GLITCHAPE_PUBLIC_URL = GLITCHAPE_PUBLIC_URL[:-1]

# --- Stable Diffusion (HuggingFace) Configuration ---
HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    log.critical("HF_API_KEY environment variable not set.")
    raise RuntimeError("HF_API_KEY environment variable not set")
HF_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
HF_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

# --- LLaMA System Prompt (Modified to handle analysis output) ---
LLAMA_SYSTEM_PROMPT = """
You are "Marvick," the central AI brain for GlitchApe, a futuristic clothing design platform.
Your primary role is to be a helpful and creative design assistant.
You MUST respond in a conversational, slightly cyberpunk, and enthusiastic tone.

**CRITICAL RULE:** If the user's prompt is prefixed with "[IMAGE ANALYSIS: ...]", use the content inside the brackets as the detailed description of the design they uploaded, provided by a vision model. Your task is to extract the core design prompt from this analysis and the user's request, then decide the next action.

You have three main tasks:
1.  **Conversational Chat:** Talk to the user, help them brainstorm, and answer questions.
2.  **Intent Detection:** Analyze the user's prompt (potentially including image analysis) and chat history to understand their goal.
3.  **Tool Orchestration:** Decide which tool to use next based on the intent.

**INTENTS & ACTIONS:**
You must classify the user's intent and decide the next action. Your response MUST contain a JSON object with the intent/action details.

1.  **Intent: `general_chat`**
    * Description: The user is just talking, asking questions, or greeting you. No image analysis involved.
    * Action: `{"intent": "general_chat", "response_text": "Your conversational reply here."}`

2.  **Intent: `design_request`**
    * Description: The user wants to create a new design. They MUST describe what they want. If the prompt contains an image analysis ([IMAGE ANALYSIS: ...]), base the `design_prompt` on that analysis plus any new user instructions.
    * Action: `{"intent": "design_request", "clothing_type": "t-shirt", "design_prompt": "a neon-cyberpunk ape drinking coffee based on uploaded image analysis", "response_text": "Got the analysis! Firing up the image generators for your design based on that..."}`
    * Valid `clothing_type` values: "t-shirt", "hoodie", "jacket", "trousers", "pants", "jeans". If unsure, default to "t-shirt".

3.  **Intent: `image_analysis`**
    * Description: The user has uploaded an image and wants you to analyze it or use it as inspiration. The visual analysis is already complete and included in the prompt ([IMAGE ANALYSIS: ...]). Use this intent to confirm receipt and perhaps ask for refinement or next steps if the user's request was vague (e.g., just "use this image").
    * Action: `{"intent": "image_analysis", "response_text": "I've processed the design! I see a [Brief Summary of Analysis from prompt]. What kind of clothing item should we put this awesome design on?", "analysis_prompt": "User wants to use their uploaded image, analysis provided."}`

4.  **Intent: `order_request`**
    * Description: The user is happy with the design and wants to buy it.
    * Action: `{"intent": "order_request", "response_text": "Great! I'm prepping the design for the order. Please proceed to checkout when ready."}`

5.  **Intent: `design_revision`**
    * Description: The user wants to change the *previous* design.
    * Action: `{"intent": "design_revision", "revision_prompt": "make the ape purple", "response_text": "Got it! Rerunning the design with that change..."}`

**RULES:**
-   **You MUST include a single, valid JSON object in your response, which should be the LAST thing you output.** Extract relevant info from `[IMAGE ANALYSIS: ...]` if present.
-   `response_text` is mandatory in all responses.
-   Be creative and on-brand (cyberpunk, glitchy, cool).
-   If the user's design prompt is vague (e.g., "make a shirt"), ask for more details (`general_chat` intent) instead of `design_request`.
"""


# --- Printful ---
PRINTFUL_API_KEY = os.getenv("PRINTFUL_API_KEY")
if not PRINTFUL_API_KEY:
    log.critical("PRINTFUL_API_KEY environment variable not set.")
    raise RuntimeError("PRINTFUL_API_KEY environment variable not set")
PRINTFUL_API_URL = "https://api.printful.com"

# --- Stripe ---
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
if not STRIPE_SECRET_KEY:
    log.critical("STRIPE_SECRET_KEY environment variable not set.")
    raise RuntimeError("STRIPE_SECRET_KEY environment variable not set")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
if not STRIPE_WEBHOOK_SECRET:
    log.critical("STRIPE_WEBHOOK_SECRET environment variable not set.")
    raise RuntimeError("STRIPE_WEBHOOK_SECRET environment variable not set")
stripe.api_key = STRIPE_SECRET_KEY

# --- File Storage ---
UPLOAD_DIR = "temp_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# --- MIME Type Mapping for reliable serving (FIX for broken images) ---
MIME_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}


# ===================================================================
# PYDANTIC SCHEMAS (Consolidated)
# ===================================================================

# --- AI Schemas ---
# NOTE: AIRequest is no longer used by the /ai/chat endpoint,
# but kept here in case other internal services use it.
class AIRequest(BaseModel):
    prompt: str
    session_id: str
    uploaded_image_url: Optional[str] = None

class ImagePreview(BaseModel):
    view_name: str
    url: str # Relative URL like /api/ai/image/xyz.png

class AIResponse(BaseModel): # Kept for potential internal use
    session_id: str
    ai_message: str
    images: List[ImagePreview] = []
    conversation_state: str = "awaiting_feedback"

# --- ADDED: Schemas for Chat History Endpoint ---
class ChatMessageResponse(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    image_url: Optional[str] = None # Still singular for individual DB messages
    created_at: datetime

    class Config:
        from_attributes = True # Replaces orm_mode in Pydantic v2

class ChatHistoryResponse(BaseModel):
    messages: List[ChatMessageResponse]

# --- UPDATED: New response model for the live chat endpoint (now supports multiple images) ---
class LiveChatResponse(BaseModel):
    session_id: str
    response_text: str
    image_urls: Optional[List[str]] = None # Changed from image_url (singular)

# --- Order Schemas ---
class Recipient(BaseModel):
    name: str
    address1: str
    address2: Optional[str] = None
    city: str
    state_code: Optional[str] = None
    country_code: str
    zip: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None

class OrderItem(BaseModel):
    variant_id: int
    product_name: str
    quantity: int = 1

class CheckoutRequest(BaseModel):
    session_id: str
    item: OrderItem
    recipient: Recipient

class CheckoutResponse(BaseModel):
    order_id: str
    payment_intent_id: str
    client_secret: str
    total_cost: float
    currency: str


# ===================================================================
# CORE LOGIC HELPERS (Private Module Functions)
# ===================================================================

# --- AI Logic Helpers ---

async def _cleanup_expired_images():
    """Deletes images from UPLOAD_DIR older than 48 hours."""
    now = datetime.utcnow()
    for filename in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.isfile(file_path):
            try:
                file_time = datetime.utcfromtimestamp(os.path.getmtime(file_path))
                if now - file_time > timedelta(hours=48):
                    os.remove(file_path)
                    log.info(f"Cleaned up expired image: {filename}")
            except OSError as e:
                log.warning(f"Failed to cleanup image {filename}: {e}")
                pass

async def _save_temp_image(file: UploadFile) -> str:
    """Saves an uploaded image temporarily, validates size/type, and checks integrity. Returns the full local file path."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name provided.")

    # Determine extension and MIME type early for validation and base64 encoding
    original_filename = file.filename
    ext = original_filename.split(".")[-1].lower()
    if ext not in ["png", "jpg", "jpeg", "webp"]:
        ext = "png" # Default to PNG if extension is unknown/unsafe
        file_mime_type = "image/png"
    else:
        file_mime_type = MIME_TYPES.get(ext, "image/jpeg") # Default to jpeg if somehow ext is valid but not in MIME_TYPES

    if not file.content_type or not file.content_type.startswith("image/"):
        # Double check MIME type from header
        log.warning(f"File {original_filename} has potentially incorrect header MIME type: {file.content_type}. Proceeding based on extension.")
        # Allow if extension seems valid
        if ext not in ["png", "jpg", "jpeg", "webp"]:
             raise HTTPException(status_code=400, detail="Invalid file type. Only images are allowed.")


    file_id = f"{uuid.uuid4()}.{ext}"
    file_path = os.path.join(UPLOAD_DIR, file_id)

    try:
        file_size = 0
        with open(file_path, "wb") as f:
            while chunk := await file.read(8192): # Read in 8KB chunks
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE_BYTES:
                    # Clean up partial file
                    f.close()
                    os.remove(file_path)
                    raise HTTPException(status_code=413, detail=f"File is too large (max {MAX_FILE_SIZE_MB}MB).")
                f.write(chunk)

        # --- Image Integrity Check (Prevents 400s later) ---
        try:
            # Re-open the saved file and verify it's a valid image
            with open(file_path, "rb") as f:
                img = Image.open(f)
                img.verify() # Checks for basic integrity without fully loading pixel data
        except Exception as image_check_error:
            # If PIL fails, delete the saved file and raise 400 Bad Request
            log.error(f"Image integrity check failed for {file_path}: {image_check_error}")
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="Uploaded file is corrupted or not a valid image format.")
        # --- END OF INTEGRITY CHECK ---

        # Return path and determined MIME type for base64 encoding later
        return file_path, file_mime_type

    except HTTPException:
        # Re-raise validation errors
        raise
    except Exception as e:
        log.error(f"Could not save temp image: {e}")
        # Attempt to clean up if something went wrong
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail="Could not save file.")
    finally:
        await file.close()


# --- UPDATED: Function to analyze image using Gemma 3 Vision Model ---
async def _analyze_image_with_vision_model(
    local_image_path: str,
    image_mime_type: str, # Get MIME type from _save_temp_image
    user_prompt: str
) -> str:
    """
    Uses google/gemma-3-4b-it:free (via OpenRouter) to describe the image content
    based on the local file path. Encodes image to base64.
    """
    log.info(f"Calling {VISION_MODEL} via OpenRouter for vision analysis on {local_image_path}...")
    if not LLAMA_API_KEY: # Use the same API key for OpenRouter models
        return "ERROR: OpenRouter API key (LLAMA_API_KEY) is missing. Cannot analyze image."

    try:
        # 1. Read the image file and encode to base64
        with open(local_image_path, "rb") as f:
            image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        
        # 2. Prepare the multimodal chat completion payload for OpenRouter
        vision_analysis_prompt = (
            f"Analyze this image which shows a clothing design (either on a person or as a flat design). "
            f"Describe the clothing item type (e.g., T-shirt, hoodie, jacket), primary colors, any patterns or graphics, "
            f"visible fabric texture, and overall style (e.g., minimalist, graphic print, streetwear, formal). "
            f"Be detailed and objective. Also consider the user's request: '{user_prompt}'"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            # Use data URI format required by multimodal chat completion APIs
                            "url": f"data:{image_mime_type};base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": vision_analysis_prompt
                    }
                ]
            }
        ]

        payload = {
            "model": VISION_MODEL, # Use the Gemma 3 model ID
            "messages": messages,
            "max_tokens": 1024, # Allow sufficient length for detailed analysis
            "temperature": 0.3, # Lower temperature for more objective description
        }
        
        headers = {
            "Authorization": f"Bearer {LLAMA_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://glitchape.fun", # Your site URL
            "X-Title": "GlitchApe Vision" # Your app name
        }

        async with httpx.AsyncClient(timeout=120) as client: # Increased timeout for vision model
            response = await client.post(
                LLAMA_API_URL, # Use the common OpenRouter endpoint
                headers=headers,
                json=payload
            )

            response.raise_for_status() # Raise exceptions for 4xx/5xx errors

            response_data = response.json()
            analysis_text = response_data['choices'][0]['message']['content']
            
            if not analysis_text:
                log.warning(f"{VISION_MODEL} returned empty analysis.")
                return "Vision model returned an empty analysis."
                
            log.info(f"Vision analysis successful using {VISION_MODEL}.")
            return analysis_text # Return the raw analysis text

    except httpx.HTTPStatusError as e:
        # Log the specific error from OpenRouter/Provider
        log.error(f"{VISION_MODEL} API Error {e.response.status_code}: {e.response.text}")
        error_detail = e.response.text[:200] # Limit error detail length
        # Check specifically for 403 errors which might be moderation flags
        if e.response.status_code == 403:
             return f"ERROR: Vision Model request rejected (HTTP 403 Forbidden). This might be due to content moderation filters on the image or prompt. Details: {error_detail}"
        return f"ERROR: Vision Model Error (HTTP {e.response.status_code}). Details: {error_detail}"
    except Exception as e:
        log.error(f"Unexpected error in {VISION_MODEL} call: {e}", exc_info=True)
        return f"ERROR: Internal Error during vision analysis: {str(e)[:100]}"


async def _generate_design_view(prompt: str, view: str) -> str:
    # ... (rest of the _generate_design_view function remains the same) ...
    """Generates a single, backgroundless 2D image view for a design using Stable Diffusion."""
    full_prompt = (
        f"Generate a single, realistic, 2D product image of the {view} view "
        f"of a {prompt}. The image must have a transparent background (PNG format). "
        f"Only show the clothing item, no models, mannequins, or shadows. "
        "The output must be a high-resolution, print-quality file."
    )
    payload = {"inputs": full_prompt}
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(HF_API_URL, headers=HF_HEADERS, json=payload)
        if response.status_code != 200:
            error_text = response.text
            log.warning(f"HuggingFace API Error: {response.status_code} - {error_text}")
            if response.status_code == 503:
                raise HTTPException(status_code=503, detail="The AI image model is still loading. Please try again in a moment.")
            error_detail = f"HuggingFace API Error: {response.status_code}"
            raise HTTPException(status_code=502, detail=error_detail)
        image_data = response.content
        try:
            img = Image.open(BytesIO(image_data))
        except Exception:
            log.error("AI returned invalid image data (PIL could not open).")
            raise HTTPException(status_code=500, detail="AI returned invalid image data.")
        file_id = f"{uuid.uuid4()}.png"
        file_path = os.path.join(UPLOAD_DIR, file_id)
        with open(file_path, "wb") as f:
            f.write(image_data)
        return file_path
    except httpx.ReadTimeout:
        log.warning("HuggingFace image generation timed out.")
        raise HTTPException(status_code=504, detail="Image generation timed out. Please try again.")
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        log.error(f"Unhandled AI image generation error: {e}")
        raise HTTPException(status_code=502, detail=f"AI image generation error: {e}")


def _get_required_views(clothing_type: str) -> List[str]:
    # ... (rest of the _get_required_views function remains the same) ...
    """Determines which clothing views to generate based on the clothing type."""
    clothing_type_lower = clothing_type.lower()
    if "hoodie" in clothing_type_lower or "shirt" in clothing_type_lower or "t-shirt" in clothing_type_lower or "jacket" in clothing_type_lower:
        return ["front", "back", "left_sleeve", "right_sleeve"]
    if "trousers" in clothing_type_lower or "pants" in clothing_type_lower or "jeans" in clothing_type_lower:
        return ["front", "back"]
    return ["main_design"]


def _place_uploaded_image(base_path: str, overlay_path: str, position: str) -> str:
    # ... (rest of the _place_uploaded_image function remains the same) ...
    """Places an uploaded image (overlay) on top of a generated image (base)."""
    try:
        base = Image.open(base_path).convert("RGBA")
        overlay = Image.open(overlay_path).convert("RGBA")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image files: {e}")
    w, h = base.size
    overlay_width = w // 4
    overlay_height = int(overlay.height * (overlay_width / overlay.width))
    overlay = overlay.resize((overlay_width, overlay_height), Image.LANCZOS)
    positions = {
        "center": ((w - overlay.width) // 2, (h - overlay.height) // 2),
        "top": ((w - overlay.width) // 2, h // 10),
        "bottom": ((w - overlay.width) // 2, h - overlay.height - (h // 10)),
        "left": (w // 10, (h - overlay.height) // 2),
        "right": (w - overlay.width - (w // 10), (h - overlay.height) // 2),
    }
    pos = positions.get(position, positions["center"])
    base.paste(overlay, pos, overlay)
    result_id = f"{uuid.uuid4()}.png"
    result_path = os.path.join(UPLOAD_DIR, result_id)
    base.save(result_path, "PNG")
    return result_path


# --- Order Logic Helpers (Unchanged - Ensure image_url path logic in _upload_to_printful is correct) ---
async def _get_design_images_from_session(
# ... (rest of the _get_design_images_from_session function remains the same)
    session_id: uuid.UUID, user_id: uuid.UUID, db: AsyncSession
) -> List[ChatMessage]:
    """Finds the *latest set* of generated design images from the chat session."""
    q = select(ChatMessage).where(
        ChatMessage.session_id == session_id, ChatMessage.user_id == user_id,
        ChatMessage.role == "ai", ChatMessage.image_url.isnot(None),
        ChatMessage.content.like("Generated image: %")
    ).order_by(ChatMessage.created_at.desc())
    r = await db.execute(q)
    all_images = r.scalars().all()
    latest_images: Dict[str, ChatMessage] = {}
    for img in all_images:
        view_name = img.content.replace("Generated image: ", "").strip()
        if view_name not in latest_images: latest_images[view_name] = img
    if not latest_images:
        raise HTTPException(status_code=404, detail="No valid AI-generated design images found in this session.")
    return list(latest_images.values())


async def _upload_to_printful(image_msg: ChatMessage, headers: Dict[str, str]) -> Dict[str, Any]:
# ... (rest of the _upload_to_printful function remains the same - path logic verified)
    """Uploads a single local file to Printful."""
    relative_url = image_msg.image_url
    if not relative_url:
         raise HTTPException(status_code=500, detail=f"Database record for image {image_msg.id} is missing URL.")
    filename = os.path.basename(relative_url) # Extracts 'xyz.png' from '/api/ai/image/xyz.png'
    local_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(local_path):
        raise HTTPException(status_code=404, detail=f"Image file not found on server at: {local_path}")
    view_name = image_msg.content.replace("Generated image: ", "").strip()
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            with open(local_path, "rb") as f:
                # Infer MIME type for upload
                ext = filename.split(".")[-1].lower()
                mime_type = MIME_TYPES.get(ext, "image/png") # Default to PNG
                files = {"file": (filename, f, mime_type)}
                resp = await client.post(f"{PRINTFUL_API_URL}/files", headers=headers, files=files)
            resp.raise_for_status()
            data = resp.json().get("result")
            if not data or not data.get("id"):
                log.error(f"Printful API did not return file ID. Response: {resp.text}")
                raise Exception("Printful API did not return a file ID.")
            return {"view": view_name, "id": data["id"]}
    except Exception as e:
        log.error(f"Printful file upload failed for {view_name}: {e}")
        raise HTTPException(status_code=502, detail=f"Printful file upload failed for {view_name}: {e}")


async def _get_printful_costs(
# ... (rest of the _get_printful_costs function remains the same)
    item: OrderItem, recipient: Recipient, headers: Dict[str, str]
) -> Dict[str, Any]:
    """Gets item and shipping costs from Printful."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            prod_resp = await client.get(f"{PRINTFUL_API_URL}/products/variant/{item.variant_id}", headers=headers)
            prod_resp.raise_for_status()
            prod_data = prod_resp.json().get("result")
            if not prod_data: raise Exception("Printful product API returned no result.")
            item_cost = float(prod_data.get("price", 0)) * item.quantity
            currency = prod_data.get("currency", "usd").lower()
            shipping_payload = { "recipient": recipient.model_dump(exclude_none=True), "items": [{"variant_id": item.variant_id, "quantity": item.quantity}]}
            ship_resp = await client.post(f"{PRINTFUL_API_URL}/shipping/rates", headers=headers, json=shipping_payload)
            ship_resp.raise_for_status()
            rate_data = ship_resp.json().get("result")
            if not rate_data or not isinstance(rate_data, list) or len(rate_data) == 0:
                raise Exception("Printful shipping API returned no rates.")
            shipping_cost = float(rate_data[0].get("rate", 0))
            total = item_cost + shipping_cost
            if total <= 0: raise Exception("Calculated total cost is zero or negative.")
            return {"total_cents": int(total * 100), "currency": currency}
    except Exception as e:
        log.error(f"Failed to calculate Printful costs: {e}. Response: {e.response.text if hasattr(e, 'response') else 'N/A'}")
        raise HTTPException(status_code=502, detail=f"Failed to calculate Printful costs: {e}")

async def _submit_order_to_printful(
# ... (rest of the _submit_order_to_printful function remains the same)
    payment: OrderPayment, order: OrderRecord, headers: Dict[str, str], db: AsyncSession
):
    """Submits the final, confirmed order to Printful."""
    try:
        recipient = json.loads(payment.recipient_json)
        file_id_map = json.loads(payment.printful_file_ids_json)
        printful_files = [{"type": view, "id": file_id} for view, file_id in file_id_map.items()]
        order_item = {"variant_id": payment.variant_id, "quantity": 1, "files": printful_files, "name": order.product_name, "external_id": str(order.id)}
        payload = {"recipient": recipient, "items": [order_item], "external_id": str(order.id)}
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(f"{PRINTFUL_API_URL}/orders", headers=headers, json=payload)
            resp.raise_for_status()
            draft_order = resp.json().get("result")
            printful_order_id = draft_order.get("id")
            confirm_resp = await client.post(f"{PRINTFUL_API_URL}/orders/{printful_order_id}/confirm", headers=headers)
            confirm_resp.raise_for_status()
            confirmed_order = confirm_resp.json().get("result")
            order.printful_order_id = str(confirmed_order.get("id"))
            payment.status = "submitted_to_printful"
            await db.commit()
            log.info(f"Successfully submitted order {order.id} to Printful (Printful ID: {order.printful_order_id})")
    except Exception as e:
        log.error(f"Printful submission failed for order {payment.order_id}: {e}")
        payment.status = "error"; payment.error_message = f"Printful submission failed: {e}"; await db.commit()


# ===================================================================
# GLITCHAPE CENTRAL HANDLER (The "Brain" Class)
# ===================================================================

class GlitchApeCentralHandler:
    """
    Orchestrates all business logic for AI, orders, and payments. Includes vision analysis step.
    """
    def __init__(self, db: AsyncSession, user: Optional[User] = None):
        self.db = db
        self.user = user

    # --- Helper to get chat history for LLaMA context ---
    async def _get_chat_history_for_context(self, session_id: uuid.UUID, limit: int = 10) -> List[Dict[str, str]]:
        # ... (implementation remains the same - maps 'ai' to 'assistant') ...
        """
        Fetches recent chat history formatted for the LLaMA API.
        CRITICAL FIX: Maps local role 'ai' to API role 'assistant'.
        """
        q = select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at.desc()).limit(limit)
        r = await self.db.execute(q)
        messages = r.scalars().all()
        llama_history = []
        for msg in reversed(messages):
            api_role = msg.role if msg.role != "ai" else "assistant"
            # Exclude potentially long analysis text from history context to save tokens
            content = msg.content
            if content.startswith("[IMAGE ANALYSIS:") and len(content) > 500: # Heuristic length check
                 content = "[Image analysis provided previously] " + content.split("USER REQUEST:", 1)[-1].strip()

            llama_history.append({"role": api_role, "content": content})
        return llama_history

    # --- Helper to call LLaMA 4 (Marvick) ---
    async def _call_llama_brain(
        self,
        prompt: str, # This prompt might contain the vision analysis prefix
        history: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        # ... (implementation remains largely the same - handles JSON parsing) ...
        """Calls the OpenRouter LLaMA model to get intent and response, using vision analysis if provided in prompt."""
        headers = { "Authorization": f"Bearer {LLAMA_API_KEY}", "Content-Type": "application/json", "HTTP-Referer": "https://glitchape.fun", "X-Title": "GlitchApe"}
        messages = [ {"role": "system", "content": LLAMA_SYSTEM_PROMPT}, *history ]
        messages.append({"role": "user", "content": prompt}) # Pass the (potentially analysis-prefixed) prompt
        payload = { "model": LLAMA_MODEL, "messages": messages }
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(LLAMA_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            llama_response_content = data.get("choices", [{}])[0].get("message", {}).get("content")
            if not llama_response_content:
                log.error(f"LLaMA response content was empty. Data: {data}")
                raise HTTPException(status_code=502, detail="LLaMA returned empty content.")
            clean_content = llama_response_content.strip()
            start = clean_content.find('{'); end = clean_content.rfind('}')
            if start != -1 and end != -1 and end > start: json_string_to_parse = clean_content[start:end+1]
            else: log.error(f"LLaMA failed to include JSON. Raw: {llama_response_content}"); json_string_to_parse = clean_content
            try: action_json = json.loads(json_string_to_parse); return action_json
            except json.JSONDecodeError as e:
                log.error(f"LLaMA invalid JSON. Error: {e}. Attempted: {json_string_to_parse}. Original: {llama_response_content}")
                raise HTTPException(status_code=502, detail=f"LLaMA returned malformed JSON: {e}")
        except httpx.HTTPStatusError as e:
            log.error(f"LLaMA API Error: {e.response.status_code} - {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=f"LLaMA API Error: {e.response.text}")
        except Exception as e:
            if isinstance(e, HTTPException): raise e
            log.error(f"Error calling LLaMA brain: {e}")
            raise HTTPException(status_code=502, detail=f"Error calling LLaMA brain: {e}")


    # --- UPDATED: Main chat handler incorporating vision analysis step ---
    async def handle_chat_message(
        self,
        session_id: uuid.UUID,
        prompt: str,
        uploaded_image: Optional[UploadFile] = None
    ) -> LiveChatResponse: # Returns updated response model
        """Orchestrates the AI chat and design generation flow, including vision analysis."""

        # 1. Verify session ownership
        if not self.user: raise HTTPException(status_code=401, detail="User not authenticated")
        session = await self.db.get(ChatSession, session_id)
        if not session or session.user_id != self.user.id: raise HTTPException(status_code=404, detail="Chat session not found or access denied")

        # 2. Handle file upload and Vision Analysis Stage
        local_image_path: Optional[str] = None
        image_analysis_text: Optional[str] = None
        analysis_error_message: Optional[str] = None
        db_image_url: Optional[str] = None # Relative URL for DB record

        if uploaded_image and uploaded_image.filename:
            try:
                local_image_path, image_mime_type = await _save_temp_image(uploaded_image)
                filename = os.path.basename(local_image_path)
                db_image_url = f"/api/ai/image/{filename}" # Relative URL for DB
                log.info(f"User {self.user.id} uploaded '{filename}'. Analyzing with {VISION_MODEL}...")

                # Call the Vision Model (Gemma 3)
                image_analysis_text = await _analyze_image_with_vision_model(local_image_path, image_mime_type, prompt)

                # Check if analysis itself returned an error message
                if image_analysis_text.startswith("ERROR:"):
                    analysis_error_message = image_analysis_text # Store error to inform user/LLaMA
                    log.warning(f"Vision analysis failed for {filename}: {analysis_error_message}")
                    image_analysis_text = None # Don't pass error text as analysis
                else:
                    log.info(f"Vision analysis successful for {filename}.")

            except HTTPException as e: # Handle file save errors
                log.warning(f"File upload/save failed for user {self.user.id}: {e.detail}")
                raise e # Propagate file errors immediately
            except Exception as e: # Handle unexpected errors during save/analysis call setup
                log.error(f"Critical error during file handling or vision analysis setup: {e}", exc_info=True)
                analysis_error_message = "System Error: Failed during image processing setup."

        # 3. Prepare the final prompt for LLaMA
        final_prompt_for_llama = prompt
        if image_analysis_text:
             # Prepend successful analysis to the user's prompt
             final_prompt_for_llama = f"[IMAGE ANALYSIS: {image_analysis_text}] User request: {prompt}"
        elif analysis_error_message:
             # Prepend error note if analysis failed
             final_prompt_for_llama = f"[IMAGE ANALYSIS FAILED: {analysis_error_message}] User request: {prompt}"
        elif db_image_url and not image_analysis_text and not analysis_error_message:
             # If image uploaded but analysis step somehow skipped (shouldn't happen), add note
             final_prompt_for_llama = f"[IMAGE UPLOADED: {db_image_url} - Analysis Skipped] User request: {prompt}"


        # 4. Save user's message (with potentially modified prompt) to DB
        user_msg = ChatMessage(
            session_id=session_id, user_id=self.user.id, role="user", content=final_prompt_for_llama,
            image_url=db_image_url # Save relative URL if image was uploaded
        )
        self.db.add(user_msg)
        await self.db.flush() # Ensure saved before history fetch

        # 5. Get context (including the latest message) and call LLaMA brain
        history = await self._get_chat_history_for_context(session_id)

        try:
            action_json = await self._call_llama_brain(final_prompt_for_llama, history)
        except HTTPException as e:
            # Handle LLaMA API call errors
            ai_text_response = f"Sorry, my AI brain (LLaMA) had a glitch... ({e.detail}). Can you try again?"
            ai_text_msg = ChatMessage(session_id=session_id, user_id=self.user.id, role="ai", content=ai_text_response)
            self.db.add(ai_text_msg); await self.db.commit()
            return LiveChatResponse(session_id=str(session_id), response_text=ai_text_response, image_urls=None)

        # 6. Process LLaMA's decision
        intent = action_json.get("intent")
        ai_text_response = action_json.get("response_text", "I received the request but I'm unsure how to respond.")
        image_previews: List[ImagePreview] = [] # Stores generated image previews

        try:
            if intent == "design_request":
                clothing_type = action_json.get("clothing_type", "t-shirt")
                design_prompt_from_llama = action_json.get("design_prompt")

                if not design_prompt_from_llama:
                    ai_text_response = "You want a design, but LLaMA couldn't extract a clear prompt from the analysis! Please describe it more clearly."
                else:
                    views_to_gen = _get_required_views(clothing_type)
                    tasks = [_generate_design_view(design_prompt_from_llama, view) for view in views_to_gen]
                    generated_paths = await asyncio.gather(*tasks)

                    for i, local_path in enumerate(generated_paths):
                        filename = os.path.basename(local_path)
                        public_url = f"/api/ai/image/{filename}" # Relative URL
                        view_name = views_to_gen[i]
                        image_previews.append(ImagePreview(view_name=view_name, url=public_url))

                        # Save individual image message to DB for history
                        ai_img_msg = ChatMessage( session_id=session_id, user_id=self.user.id, role="ai", content=f"Generated image: {view_name}", image_url=public_url)
                        self.db.add(ai_img_msg)

            # ... (rest of intent handling: design_revision, order_request, general_chat, image_analysis, unknown) ...
            elif intent == "design_revision": ai_text_response = action_json.get("response_text", "OK, I'll revise that.")
            elif intent == "order_request": ai_text_response = action_json.get("response_text", "Great! Let's get that ordered.")
            elif intent == "general_chat" or intent == "image_analysis": pass # Just use the text response
            else: log.warning(f"LLaMA returned unknown intent: {intent}"); ai_text_response = action_json.get("response_text", "Got it.")

        except Exception as e:
            log.error(f"Error during intent processing '{intent}': {e}", exc_info=True)
            ai_text_response = "I tried to process your request, but an error occurred during execution."
            if isinstance(e, HTTPException): ai_text_response = f"I tried, but hit a snag: {e.detail}"


        # 7. Extract ALL generated image URLs for the response payload
        all_generated_image_urls = [preview.url for preview in image_previews] if image_previews else None

        # 8. Save the main AI text response to DB
        # Note: ChatMessage.image_url still only stores one URL, maybe the first generated one for history simplicity
        main_db_image_url = all_generated_image_urls[0] if all_generated_image_urls else None
        ai_text_msg = ChatMessage(session_id=session_id, user_id=self.user.id, role="ai", content=ai_text_response, image_url=main_db_image_url)
        self.db.add(ai_text_msg)
        await self.db.commit() # Commit all DB changes

        # 9. Return the consolidated response with potentially multiple image URLs
        return LiveChatResponse(
            session_id=str(session_id),
            response_text=ai_text_response,
            image_urls=all_generated_image_urls # Pass the list of generated URLs
        )


    # --- Other Handler Methods (initiate_checkout, handle_stripe_webhook, etc.) ---
    async def initiate_checkout(self, req: CheckoutRequest) -> CheckoutResponse:
        # ... (implementation remains the same, including 400 validation checks) ...
        """Orchestrates the checkout initiation flow."""
        if not self.user: raise HTTPException(status_code=401, detail="User not authenticated")
        try: session_uuid = uuid.UUID(req.session_id)
        except ValueError: raise HTTPException(status_code=400, detail="Invalid session_id format.")
        printful_headers = {"Authorization": f"Bearer {PRINTFUL_API_KEY}"}
        recipient_data = req.recipient.model_dump(exclude_none=True)
        required_fields = ["name", "address1", "city", "country_code"]
        missing_fields = [field for field in required_fields if not recipient_data.get(field)]
        if missing_fields: raise HTTPException(status_code=400, detail=f"Missing required recipient information: {', '.join(missing_fields)}.")
        if req.item.quantity <= 0: raise HTTPException(status_code=400, detail="Order quantity must be a positive number.")
        images = await _get_design_images_from_session(session_uuid, self.user.id, self.db)
        upload_tasks = [_upload_to_printful(img, printful_headers) for img in images]
        upload_results = await asyncio.gather(*upload_tasks)
        file_id_map = {res["view"]: res["id"] for res in upload_results}; file_ids_json = json.dumps(file_id_map)
        costs = await _get_printful_costs(req.item, req.recipient, printful_headers)
        async with self.db.begin_nested():
            new_order = OrderRecord(user_id=self.user.id, product_name=req.item.product_name, image_id=images[0].id)
            self.db.add(new_order); await self.db.flush()
            try: intent = stripe.PaymentIntent.create( amount=costs["total_cents"], currency=costs["currency"], automatic_payment_methods={"enabled": True}, metadata={"order_id": str(new_order.id), "user_id": str(self.user.id)})
            except Exception as e: log.error(f"Stripe Payment Intent creation failed: {e}"); raise HTTPException(status_code=502, detail=f"Stripe Payment Intent creation failed: {e}")
            new_payment = OrderPayment( order_id=new_order.id, user_id=self.user.id, payment_intent_id=intent.id, total_cost_cents=costs["total_cents"], currency=costs["currency"], recipient_json=req.recipient.model_dump_json(), printful_file_ids_json=file_ids_json, variant_id=req.item.variant_id, status="pending_payment")
            self.db.add(new_payment); await self.db.flush()
            stripe.PaymentIntent.modify( intent.id, metadata={ "order_id": str(new_order.id), "order_payment_id": str(new_payment.id), "user_id": str(self.user.id)})
            await self.db.commit()
        return CheckoutResponse( order_id=str(new_order.id), payment_intent_id=intent.id, client_secret=intent.client_secret, total_cost=round(costs["total_cents"] / 100, 2), currency=costs["currency"])


    async def handle_stripe_webhook(self, payload: bytes, sig_header: str) -> dict:
        # ... (implementation remains the same) ...
        """Orchestrates the payment confirmation and fulfillment trigger."""
        if not STRIPE_WEBHOOK_SECRET: log.error("Stripe webhook secret not configured."); raise HTTPException(status_code=500, detail="Stripe webhook secret not configured.")
        try: event = stripe.Webhook.construct_event( payload=payload, sig_header=sig_header, secret=STRIPE_WEBHOOK_SECRET)
        except ValueError: log.warning("Stripe webhook invalid payload."); raise HTTPException(status_code=400, detail="Invalid payload")
        except stripe.error.SignatureVerificationError: log.warning("Stripe webhook invalid signature."); raise HTTPException(status_code=400, detail="Invalid signature")
        if event.type == "payment_intent.succeeded":
            intent = event.data.object; payment_id_str = intent.metadata.get("order_payment_id")
            if not payment_id_str: log.error(f"Stripe event {event.id} missing 'order_payment_id'."); return {"status": "error", "reason": "Missing metadata"}
            try: payment_id = uuid.UUID(payment_id_str)
            except ValueError: log.error(f"Stripe event {event.id} invalid UUID: {payment_id_str}"); return {"status": "error", "reason": "Invalid payment_id format"}
            payment = await self.db.get(OrderPayment, payment_id)
            if not payment: log.error(f"Stripe event {event.id} referenced non-existent OrderPayment: {payment_id}"); return {"status": "error", "reason": "Payment record not found"}
            if payment.status not in ["pending_payment", "failed"]: log.info(f"Stripe event {event.id} already processed. Status: {payment.status}"); return {"status": "ok", "message": "Already processed"}
            payment.status = "succeeded"; await self.db.flush()
            order = await self.db.get(OrderRecord, payment.order_id)
            if not order: payment.status = "error"; payment.error_message = f"OrderRecord {payment.order_id} not found."; await self.db.commit(); log.error(f"Stripe event {event.id} succeeded but OrderRecord {payment.order_id} not found."); return {"status": "error", "reason": "OrderRecord not found"}
            printful_headers = {"Authorization": f"Bearer {PRINTFUL_API_KEY}"}
            asyncio.create_task(_submit_order_to_printful(payment, order, printful_headers, self.db))
            log.info(f"Stripe event {event.id}: Payment {payment_id} succeeded. Queued Printful submission.")
        elif event.type == "payment_intent.payment_failed":
            intent = event.data.object; payment_id_str = intent.metadata.get("order_payment_id")
            if payment_id_str:
                try:
                    payment = await self.db.get(OrderPayment, uuid.UUID(payment_id_str))
                    if payment: payment.status = "failed"; payment.error_message = intent.last_payment_error.message if intent.last_payment_error else "Unknown"; await self.db.commit(); log.warning(f"Stripe event {event.id}: Payment {payment_id_str} failed: {payment.error_message}")
                except Exception as e: log.error(f"Failed to process payment_intent.payment_failed event: {e}")
        return {"status": "received"}


    async def handle_image_upload(self, file: UploadFile) -> dict:
        # ... (implementation remains the same) ...
        """Handles user-uploaded images for placement."""
        # Note: _save_temp_image now returns path AND mime_type, but we only need path here
        path, _ = await _save_temp_image(file)
        filename = os.path.basename(path)
        return {"status": "success", "filename": filename, "file_url": f"/api/ai/image/{filename}"}


    async def handle_image_placement(
        self, base_filename: str, overlay_filename: str, position: str
    ) -> dict:
        # ... (implementation remains the same, including 400 validation checks) ...
        """Combines an uploaded image with a generated design."""
        def is_safe_filename(name): return name and not (".." in name or "/" in name or "\\" in name)
        if not is_safe_filename(base_filename) or not is_safe_filename(overlay_filename): raise HTTPException(status_code=400, detail="Invalid or unsafe filename provided.")
        base_path = os.path.join(UPLOAD_DIR, base_filename); overlay_path = os.path.join(UPLOAD_DIR, overlay_filename)
        if not os.path.exists(base_path) or not os.path.exists(overlay_path): raise HTTPException(status_code=404, detail="One or both image files not found.")
        try:
            result_path = _place_uploaded_image(base_path, overlay_path, position)
            filename = os.path.basename(result_path)
            return { "status": "success", "filename": filename, "image_url": f"/api/ai/image/{filename}"}
        except HTTPException as e: raise e
        except Exception as e: log.error(f"Error placing image: {e}"); raise HTTPException(status_code=500, detail=f"Error processing image placement: {e}")


# ===================================================================
# FASTAPI ROUTER (Thin Wrappers)
# ===================================================================

# This single router will be imported and mounted by server.py
router = APIRouter(tags=["GlitchApe Central"])


# --- ADDED: Endpoint to create a new chat session ---
@router.post("/chat/start", status_code=status.HTTP_201_CREATED)
async def start_new_chat_session(
    db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)
):
    # ... (implementation remains the same) ...
    """Starts a new chat session for the authenticated user and returns the session ID."""
    try:
        new_session = ChatSession( user_id=current_user.id, title=f"New Chat - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        db.add(new_session); await db.commit(); await db.refresh(new_session)
        log.info(f"User {current_user.id} started new session {new_session.id}")
        return {"session_id": str(new_session.id)}
    except Exception as e: log.error(f"Failed to create new chat session for user {current_user.id}: {e}"); await db.rollback(); raise HTTPException(status_code=500, detail=f"Failed to create new chat session: {e}")


# --- *** UPDATED: This endpoint now accepts multipart/form-data and returns multiple URLs *** ---
@router.post("/ai/chat", response_model=LiveChatResponse) # Uses updated response model
async def chat_with_ai(
    prompt: str = Form(...), session_id: uuid.UUID = Form(...), uploaded_image: Optional[UploadFile] = File(None),
    db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user),
):
    # ... (implementation remains the same) ...
    """Main AI chat endpoint. Accepts form-data."""
    asyncio.create_task(_cleanup_expired_images())
    handler = GlitchApeCentralHandler(db=db, user=current_user)
    return await handler.handle_chat_message( session_id=session_id, prompt=prompt, uploaded_image=uploaded_image)


# --- ADDED: Endpoint to get chat history for a session ---
@router.get("/chat/history/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(
    session_id: uuid.UUID, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)
):
    # ... (implementation remains the same) ...
    """Retrieves the chat history for a specific session, verifying user ownership."""
    session = await db.get(ChatSession, session_id)
    if not session: raise HTTPException(status_code=404, detail="Chat session not found")
    if session.user_id != current_user.id: raise HTTPException(status_code=403, detail="Access denied to this chat session")
    q = select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at.asc())
    r = await db.execute(q); messages = r.scalars().all()
    return ChatHistoryResponse(messages=messages)


@router.post("/orders/initiate-checkout", response_model=CheckoutResponse)
async def initiate_checkout(
    req: CheckoutRequest, db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)
):
    # ... (implementation remains the same) ...
    """Starts the checkout process."""
    handler = GlitchApeCentralHandler(db=db, user=user)
    return await handler.initiate_checkout(req)


@router.post("/orders/stripe-webhook")
async def stripe_webhook(
    request: Request, db: AsyncSession = Depends(get_db)
):
    # ... (implementation remains the same) ...
    """Handles incoming webhooks from Stripe."""
    handler = GlitchApeCentralHandler(db=db)
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    return await handler.handle_stripe_webhook(payload, sig_header)


@router.post("/ai/upload-image")
async def upload_user_image(
    file: UploadFile = File(...), db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user),
):
    # ... (implementation remains the same) ...
    """Handles user-uploaded images (e.g., a logo)."""
    handler = GlitchApeCentralHandler(db=db, user=current_user)
    return await handler.handle_image_upload(file)


@router.post("/ai/place-image")
async def place_image_on_design(
    base_filename: str = Form(...), overlay_filename: str = Form(...), position: str = Form("center"),
    db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user),
):
    # ... (implementation remains the same) ...
    """Combines a user-uploaded image with a generated outfit image."""
    handler = GlitchApeCentralHandler(db=db, user=current_user)
    return await handler.handle_image_placement(base_filename, overlay_filename, position)


# --- UPDATED: get_image endpoint with explicit MIME type fix ---
@router.get("/ai/image/{filename}")
async def get_image(filename: str):
    # ... (implementation remains the same - MIME type fix is included) ...
    """Serves generated or uploaded images with explicit MIME type."""
    if ".." in filename or filename.startswith(("/", "\\")): raise HTTPException(status_code=400, detail="Invalid filename.")
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path): raise HTTPException(status_code=404, detail="Image not found")
    ext = filename.split(".")[-1].lower()
    media_type = MIME_TYPES.get(ext, "application/octet-stream")
    return FileResponse(file_path, media_type=media_type)
