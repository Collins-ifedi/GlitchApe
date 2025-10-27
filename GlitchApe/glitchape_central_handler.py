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
import logging # <-- ADDED: For production logging
from io import BytesIO
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from fastapi import (
    APIRouter, Depends, HTTPException, Request,
    UploadFile, File, Form, status
)
from fastapi.responses import FileResponse
from pantic import BaseModel
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

# --- ADDED: Logging Setup ---
log = logging.getLogger(__name__)

# ===================================================================
# CONFIGURATION
# ===================================================================

# --- NEW: LLaMA 4 (OpenRouter) Configuration ---
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
if not LLAMA_API_KEY:
    log.critical("LLAMA_API_KEY environment variable not set.")
    raise RuntimeError("LLAMA_API_KEY environment variable not set")
LLAMA_API_URL = "https://openrouter.ai/api/v1/chat/completions"
LLAMA_MODEL = "meta-llama/llama-4-maverick:free" # Using LLaMA 4 Marvick as requested

# --- NEW: Stable Diffusion (HuggingFace) Configuration ---
HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    log.critical("HF_API_KEY environment variable not set.")
    raise RuntimeError("HF_API_KEY environment variable not set")
HF_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
HF_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

# --- LLaMA System Prompt ---
LLAMA_SYSTEM_PROMPT = """
You are "Marvick," the central AI brain for GlitchApe, a futuristic clothing design platform.
Your primary role is to be a helpful and creative design assistant.
You MUST respond in a conversational, slightly cyberpunk, and enthusiastic tone.

You have three main tasks:
1.  **Conversational Chat:** Talk to the user, help them brainstorm, and answer questions.
2.  **Intent Detection:** Analyze the user's prompt and chat history to understand their goal.
3.  **Tool Orchestration:** Decide which tool to use next based on the intent.

**INTENTS & ACTIONS:**
You must classify the user's intent and decide the next action. Your response MUST be a JSON object.

1.  **Intent: `general_chat`**
    * Description: The user is just talking, asking questions, or greeting you.
    * Action: `{"intent": "general_chat", "response_text": "Your conversational reply here."}`

2.  **Intent: `design_request`**
    * Description: The user wants to create a new design. They MUST describe what they want.
    * Action: `{"intent": "design_request", "clothing_type": "t-shirt", "design_prompt": "a neon-cyberpunk ape drinking coffee", "response_text": "Awesome! I'm firing up the image generators for your design..."}`
    * Valid `clothing_type` values: "t-shirt", "hoodie", "jacket", "trousers", "pants", "jeans". If unsure, default to "t-shirt".

3.  **Intent: `image_analysis`**
    * Description: The user has uploaded an image and wants you to analyze it or use it as inspiration. The prompt will contain a message like "use the image I just uploaded."
    * Action: `{"intent": "image_analysis", "response_text": "That's a slick image! What part should I use for inspiration?", "analysis_prompt": "User wants to use their uploaded image."}`
    * (Note: The backend will handle attaching the image to a future prompt if needed).

4.  **Intent: `order_request`**
    * Description: The user is happy with the design and wants to buy it. They might say "I want to order this," "let's buy it," or "proceed to checkout."
    * Action: `{"intent": "order_request", "response_text": "Great! I'm prepping the design for the order. Please proceed to checkout when ready."}`

5.  **Intent: `design_revision`**
    * Description: The user wants to change the *previous* design.
    * Action: `{"intent": "design_revision", "revision_prompt": "make the ape purple", "response_text": "Got it! Rerunning the design with that change..."}`
    * (Note: The backend will combine this with the previous design_prompt).

**RULES:**
-   You MUST ALWAYS respond with a valid JSON object.
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
    url: str

class AIResponse(BaseModel):
    session_id: str
    ai_message: str
    images: List[ImagePreview] = []
    conversation_state: str = "awaiting_feedback"

# --- ADDED: Schemas for Chat History Endpoint ---
class ChatMessageResponse(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    image_url: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True # Replaces orm_mode in Pydantic v2

class ChatHistoryResponse(BaseModel):
    messages: List[ChatMessageResponse]

# --- ADDED: New response model for the live chat endpoint ---
class LiveChatResponse(BaseModel):
    session_id: str
    response_text: str
    image_url: Optional[str] = None

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
    """Saves an uploaded image temporarily. Returns the full local file path."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name provided.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only images are allowed.")

    # Secure filename and path
    ext = file.filename.split(".")[-1]
    if ext.lower() not in ["png", "jpg", "jpeg", "webp"]:
        ext = "png" # Default to PNG if extension is unknown/unsafe

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

        return file_path

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

# --- UPDATE: Replaced Google GenAI with HuggingFace Stable Diffusion ---
async def _generate_design_view(prompt: str, view: str) -> str:
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

            # Handle specific 503 "model loading" error
            if response.status_code == 503:
                raise HTTPException(status_code=503, detail="The AI image model is still loading. Please try again in a moment.")

            error_detail = f"HuggingFace API Error: {response.status_code}"
            raise HTTPException(status_code=502, detail=error_detail)

        image_data = response.content

        try:
            # Use BytesIO to open the image data in memory
            img = Image.open(BytesIO(image_data))
        except Exception:
            log.error("AI returned invalid image data (PIL could not open).")
            raise HTTPException(status_code=500, detail="AI returned invalid image data.")

        file_id = f"{uuid.uuid4()}.png"
        file_path = os.path.join(UPLOAD_DIR, file_id)

        # Save the image data to a file
        with open(file_path, "wb") as f:
            f.write(image_data)

        return file_path

    except httpx.ReadTimeout:
        log.warning("HuggingFace image generation timed out.")
        raise HTTPException(status_code=504, detail="Image generation timed out. Please try again.")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        log.error(f"Unhandled AI image generation error: {e}")
        raise HTTPException(status_code=502, detail=f"AI image generation error: {e}")

def _get_required_views(clothing_type: str) -> List[str]:
    """Determines which clothing views to generate based on the clothing type."""
    clothing_type_lower = clothing_type.lower()
    if "hoodie" in clothing_type_lower or "shirt" in clothing_type_lower or "t-shirt" in clothing_type_lower or "jacket" in clothing_type_lower:
        return ["front", "back", "left_sleeve", "right_sleeve"]
    if "trousers" in clothing_type_lower or "pants" in clothing_type_lower or "jeans" in clothing_type_lower:
        return ["front", "back"]
    # Default for unknown types
    return ["main_design"]

def _place_uploaded_image(base_path: str, overlay_path: str, position: str) -> str:
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

# --- Order Logic Helpers (Unchanged) ---

async def _get_design_images_from_session(
    session_id: uuid.UUID, user_id: uuid.UUID, db: AsyncSession
) -> List[ChatMessage]:
    """Finds the *latest set* of generated design images from the chat session."""
    q = select(ChatMessage).where(
        ChatMessage.session_id == session_id,
        ChatMessage.user_id == user_id,
        ChatMessage.role == "ai",
        ChatMessage.image_url.isnot(None),
        ChatMessage.content.like("Generated image: %")
    ).order_by(ChatMessage.created_at.desc())

    r = await db.execute(q)
    all_images = r.scalars().all()

    latest_images: Dict[str, ChatMessage] = {}
    for img in all_images:
        view_name = img.content.replace("Generated image: ", "").strip()
        if view_name not in latest_images:
            latest_images[view_name] = img

    if not latest_images:
        raise HTTPException(
            status_code=404,
            detail="No valid AI-generated design images found in this session."
        )
    return list(latest_images.values())

async def _upload_to_printful(image_msg: ChatMessage, headers: Dict[str, str]) -> Dict[str, Any]:
    """Uploads a single local file to Printful."""
    local_path = os.path.join(UPLOAD_DIR, os.path.basename(image_msg.image_url))

    if not os.path.exists(local_path):
        raise HTTPException(status_code=404, detail=f"Image file not found: {local_path}")

    view_name = image_msg.content.replace("Generated image: ", "").strip()

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            with open(local_path, "rb") as f:
                files = {"file": (os.path.basename(local_path), f, "image/png")}
                resp = await client.post(
                    f"{PRINTFUL_API_URL}/files",
                    headers=headers,
                    files=files
                )
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
    item: OrderItem, recipient: Recipient, headers: Dict[str, str]
) -> Dict[str, Any]:
    """Gets item and shipping costs from Printful."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            prod_resp = await client.get(
                f"{PRINTFUL_API_URL}/products/variant/{item.variant_id}",
                headers=headers
            )
            prod_resp.raise_for_status()
            prod_data = prod_resp.json().get("result")
            if not prod_data:
                raise Exception("Printful product API returned no result.")
            item_cost = float(prod_data.get("price", 0)) * item.quantity
            currency = prod_data.get("currency", "usd").lower()

            shipping_payload = {
                "recipient": recipient.model_dump(exclude_none=True),
                "items": [{"variant_id": item.variant_id, "quantity": item.quantity}]
            }
            ship_resp = await client.post(
                f"{PRINTFUL_API_URL}/shipping/rates",
                headers=headers,
                json=shipping_payload
            )
            ship_resp.raise_for_status()

            rate_data = ship_resp.json().get("result")
            if not rate_data or not isinstance(rate_data, list) or len(rate_data) == 0:
                raise Exception("Printful shipping API returned no rates.")

            shipping_cost = float(rate_data[0].get("rate", 0))

            total = item_cost + shipping_cost
            if total <= 0:
                raise Exception("Calculated total cost is zero or negative.")

            return {"total_cents": int(total * 100), "currency": currency}
    except Exception as e:
        log.error(f"Failed to calculate Printful costs: {e}. Response: {e.response.text if hasattr(e, 'response') else 'N/A'}")
        raise HTTPException(status_code=502, detail=f"Failed to calculate Printful costs: {e}")

async def _submit_order_to_printful(
    payment: OrderPayment,
    order: OrderRecord,
    headers: Dict[str, str],
    db: AsyncSession
):
    """Submits the final, confirmed order to Printful."""
    try:
        recipient = json.loads(payment.recipient_json)
        file_id_map = json.loads(payment.printful_file_ids_json)
        printful_files = [{"type": view, "id": file_id} for view, file_id in file_id_map.items()]

        order_item = {
            "variant_id": payment.variant_id,
            "quantity": 1,
            "files": printful_files,
            "name": order.product_name,
            "external_id": str(order.id)
        }
        payload = {"recipient": recipient, "items": [order_item], "external_id": str(order.id)}

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(f"{PRINTFUL_API_URL}/orders", headers=headers, json=payload)
            resp.raise_for_status()
            draft_order = resp.json().get("result")
            printful_order_id = draft_order.get("id")

            confirm_resp = await client.post(
                f"{PRINTFUL_API_URL}/orders/{printful_order_id}/confirm",
                headers=headers
            )
            confirm_resp.raise_for_status()
            confirmed_order = confirm_resp.json().get("result")

            order.printful_order_id = str(confirmed_order.get("id"))
            payment.status = "submitted_to_printful"
            await db.commit()
            log.info(f"Successfully submitted order {order.id} to Printful (Printful ID: {order.printful_order_id})")
    except Exception as e:
        log.error(f"Printful submission failed for order {payment.order_id}: {e}")
        payment.status = "error"
        payment.error_message = f"Printful submission failed: {e}"
        await db.commit()


# ===================================================================
# GLITCHAPE CENTRAL HANDLER (The "Brain" Class)
# ===================================================================

class GlitchApeCentralHandler:
    """
    Orchestrates all business logic for AI, orders, and payments.
    """
    def __init__(self, db: AsyncSession, user: Optional[User] = None):
        self.db = db
        self.user = user

    # --- NEW: Helper to get chat history for LLaMA context ---
    async def _get_chat_history_for_context(self, session_id: uuid.UUID, limit: int = 10) -> List[Dict[str, str]]:
        """Fetches recent chat history formatted for the LLaMA API."""
        q = select(ChatMessage).where(
            ChatMessage.session_id == session_id
        ).order_by(ChatMessage.created_at.desc()).limit(limit)

        r = await self.db.execute(q)
        messages = r.scalars().all()

        # Format for LLaMA API (reverse to get chronological order)
        llama_history = []
        for msg in reversed(messages):
            # Ensure only 'role' and 'content' are included for the API
            llama_history.append({"role": msg.role, "content": msg.content})

        return llama_history

    # --- NEW: Helper to call LLaMA 4 (Marvick) ---
    async def _call_llama_brain(
        self,
        prompt: str,
        history: List[Dict[str, str]],
        uploaded_image_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Calls the OpenRouter LLaMA model to get intent and response."""

        headers = {
            "Authorization": f"Bearer {LLAMA_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://glitchape.fun", # Recommended by OpenRouter
            "X-Title": "GlitchApe" # Recommended by OpenRouter
        }

        messages = [
            {"role": "system", "content": LLAMA_SYSTEM_PROMPT},
            *history # Add past conversation (already cleaned by _get_chat_history_for_context)
        ]

        final_prompt_content = prompt
        if uploaded_image_url:
            final_prompt_content += f"\n\n[System Note: The user just uploaded an image for inspiration at: {uploaded_image_url}. Please acknowledge this if it seems relevant.]"

        messages.append({"role": "user", "content": final_prompt_content})

        payload = {
            "model": LLAMA_MODEL,
            "messages": messages,
            # Removed "response_format" based on previous errors
        }

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(LLAMA_API_URL, headers=headers, json=payload)

            response.raise_for_status() # Raise exception for 4xx/5xx errors

            data = response.json()
            llama_response_content = data.get("choices", [{}])[0].get("message", {}).get("content")

            if not llama_response_content:
                log.error(f"LLaMA response content was empty. Data: {data}")
                raise HTTPException(status_code=502, detail="LLaMA returned empty content.")

            # --- CRITICAL FIX: Aggressively clean the response content before JSON parsing ---
            clean_content = llama_response_content.strip()

            # Find the start and end of the JSON object
            start = clean_content.find('{')
            end = clean_content.rfind('}')

            if start != -1 and end != -1 and end > start:
                json_string_to_parse = clean_content[start:end+1]
            else:
                # If no clear JSON object found, maybe it's just text? Log and try anyway.
                log.warning(f"Could not reliably find JSON object braces '{{}}' in response. Attempting parse anyway. Raw: {llama_response_content}")
                json_string_to_parse = clean_content

            # Parse the cleaned/extracted JSON string from LLaMA
            try:
                action_json = json.loads(json_string_to_parse)
                return action_json
            except json.JSONDecodeError as e:
                # Log both the attempted string and the original for debugging
                log.error(f"LLaMA returned invalid JSON. Error: {e}. String attempted to parse: {json_string_to_parse}")
                log.error(f"Original full response from LLaMA: {llama_response_content}")
                raise HTTPException(status_code=502, detail=f"LLaMA returned malformed JSON: {e}")
            # --- END OF CRITICAL FIX ---

        except httpx.HTTPStatusError as e:
            log.error(f"LLaMA API Error: {e.response.status_code} - {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=f"LLaMA API Error: {e.response.text}")
        except Exception as e:
            log.error(f"Error calling LLaMA brain: {e}")
            raise HTTPException(status_code=502, detail=f"Error calling LLaMA brain: {e}")

    # --- MODIFIED: This function now accepts UploadFile ---
    async def handle_chat_message(
        self,
        session_id: uuid.UUID,
        prompt: str,
        uploaded_image: Optional[UploadFile] = None
    ) -> LiveChatResponse:
        """Orchestrates the AI chat and design generation flow using LLaMA."""

        # 1. Verify session ownership
        if not self.user:
            raise HTTPException(status_code=401, detail="User not authenticated")

        session = await self.db.get(ChatSession, session_id)
        if not session or session.user_id != self.user.id:
            raise HTTPException(status_code=404, detail="Chat session not found or access denied")

        # 2. Save user's message
        user_msg = ChatMessage(
            session_id=session_id, user_id=self.user.id, role="user", content=prompt
        )
        self.db.add(user_msg)
        await self.db.flush() # Flush to ensure user message is saved before history fetch

        # 3. Handle file upload
        uploaded_image_url: Optional[str] = None
        if uploaded_image and uploaded_image.filename:
            try:
                local_path = await _save_temp_image(uploaded_image)
                filename = os.path.basename(local_path)
                uploaded_image_url = f"/api/ai/image/{filename}" # Create relative URL
                log.info(f"User {self.user.id} uploaded '{filename}' for session {session_id}")
            except HTTPException as e:
                log.warning(f"File upload failed for user {self.user.id}: {e.detail}")
                raise e
            except Exception as e:
                log.error(f"Critical file save error for user {self.user.id}: {e}")
                raise HTTPException(status_code=500, detail="Failed to save uploaded image.")

        # 4. Get context and call LLaMA "brain"
        history = await self._get_chat_history_for_context(session_id)

        try:
            action_json = await self._call_llama_brain(prompt, history, uploaded_image_url)
        except HTTPException as e:
            # If LLaMA fails, return a simple error message
            ai_text_response = f"Sorry, my AI brain had a glitch... ({e.detail}). Can you try that again?"
            ai_text_msg = ChatMessage(
                session_id=session_id, user_id=self.user.id, role="ai", content=ai_text_response
            )
            self.db.add(ai_text_msg)
            await self.db.commit() # Commit the error message to DB
            return LiveChatResponse(
                session_id=str(session_id),
                response_text=ai_text_response,
                image_url=None
            )

        # 5. Process LLaMA's decision
        intent = action_json.get("intent")
        ai_text_response = action_json.get("response_text", "I'm not sure what to say!")
        image_previews: List[ImagePreview] = []

        try:
            if intent == "design_request":
                clothing_type = action_json.get("clothing_type", "t-shirt")
                design_prompt = action_json.get("design_prompt")

                if not design_prompt:
                    ai_text_response = "You want a design, but I'm not sure what of! Please describe it."
                else:
                    views_to_gen = _get_required_views(clothing_type)
                    tasks = [_generate_design_view(design_prompt, view) for view in views_to_gen]
                    generated_paths = await asyncio.gather(*tasks)

                    for i, local_path in enumerate(generated_paths):
                        filename = os.path.basename(local_path)
                        public_url = f"/api/ai/image/{filename}"
                        view_name = views_to_gen[i]
                        image_previews.append(ImagePreview(view_name=view_name, url=public_url))

                        # Save image message to DB
                        ai_img_msg = ChatMessage(
                            session_id=session_id, user_id=self.user.id, role="ai",
                            content=f"Generated image: {view_name}", image_url=public_url
                        )
                        self.db.add(ai_img_msg)

            elif intent == "design_revision":
                # TODO: Implement revision logic
                ai_text_response = action_json.get("response_text", "OK, I'll revise that.")
            elif intent == "order_request":
                # TODO: Update session state if needed
                ai_text_response = action_json.get("response_text", "Great! Let's get that ordered.")
            elif intent == "general_chat" or intent == "image_analysis":
                pass # Just use the text response
            else:
                log.warning(f"LLaMA returned unknown intent: {intent}")
                ai_text_response = action_json.get("response_text", "Got it.")

        except Exception as e:
            log.error(f"Error during intent processing '{intent}': {e}")
            ai_text_response = "I tried to process your request, but an error occurred during execution."
            if isinstance(e, HTTPException):
                ai_text_response = f"I tried, but hit a snag: {e.detail}"

        # Extract the primary image URL for the response
        main_image_url = image_previews[0].url if image_previews else None

        # 6. Save the main AI text response
        ai_text_msg = ChatMessage(
            session_id=session_id, user_id=self.user.id, role="ai", content=ai_text_response,
            image_url=main_image_url
        )
        self.db.add(ai_text_msg)
        await self.db.commit() # Commit all DB changes for this interaction

        # 7. Return the consolidated response
        return LiveChatResponse(
            session_id=str(session_id),
            response_text=ai_text_response,
            image_url=main_image_url
        )

    async def initiate_checkout(self, req: CheckoutRequest) -> CheckoutResponse:
        """Orchestrates the checkout initiation flow."""
        if not self.user:
            raise HTTPException(status_code=401, detail="User not authenticated")

        try:
            session_uuid = uuid.UUID(req.session_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid session_id format.")

        printful_headers = {"Authorization": f"Bearer {PRINTFUL_API_KEY}"}

        # 1. Get Images
        images = await _get_design_images_from_session(session_uuid, self.user.id, self.db)

        # 2. Upload to Printful
        upload_tasks = [_upload_to_printful(img, printful_headers) for img in images]
        upload_results = await asyncio.gather(*upload_tasks)
        file_id_map = {res["view"]: res["id"] for res in upload_results}
        file_ids_json = json.dumps(file_id_map)

        # 3. Get Costs
        costs = await _get_printful_costs(req.item, req.recipient, printful_headers)

        # 4. Create local DB records & Stripe Intent
        async with self.db.begin_nested():
            new_order = OrderRecord(
                user_id=self.user.id,
                product_name=req.item.product_name,
                image_id=images[0].id
            )
            self.db.add(new_order)
            await self.db.flush()

            try:
                intent = stripe.PaymentIntent.create(
                    amount=costs["total_cents"],
                    currency=costs["currency"],
                    automatic_payment_methods={"enabled": True},
                    metadata={"order_id": str(new_order.id), "user_id": str(self.user.id)}
                )
            except Exception as e:
                log.error(f"Stripe Payment Intent creation failed: {e}")
                raise HTTPException(status_code=502, detail=f"Stripe Payment Intent creation failed: {e}")

            new_payment = OrderPayment(
                order_id=new_order.id,
                user_id=self.user.id,
                payment_intent_id=intent.id,
                total_cost_cents=costs["total_cents"],
                currency=costs["currency"],
                recipient_json=req.recipient.model_dump_json(),
                printful_file_ids_json=file_ids_json,
                variant_id=req.item.variant_id,
                status="pending_payment"
            )
            self.db.add(new_payment)
            await self.db.flush()

            stripe.PaymentIntent.modify(
                intent.id,
                metadata={
                    "order_id": str(new_order.id),
                    "order_payment_id": str(new_payment.id), # Crucial link
                    "user_id": str(self.user.id)
                }
            )
            await self.db.commit()

        # 6. Return response
        return CheckoutResponse(
            order_id=str(new_order.id),
            payment_intent_id=intent.id,
            client_secret=intent.client_secret,
            total_cost=round(costs["total_cents"] / 100, 2),
            currency=costs["currency"]
        )

    async def handle_stripe_webhook(self, payload: bytes, sig_header: str) -> dict:
        """Orchestrates the payment confirmation and fulfillment trigger."""
        if not STRIPE_WEBHOOK_SECRET:
            log.error("Stripe webhook received but STRIPE_WEBHOOK_SECRET is not configured.")
            raise HTTPException(status_code=500, detail="Stripe webhook secret is not configured.")

        try:
            event = stripe.Webhook.construct_event(
                payload=payload, sig_header=sig_header, secret=STRIPE_WEBHOOK_SECRET
            )
        except ValueError:
            log.warning("Stripe webhook received with invalid payload.")
            raise HTTPException(status_code=400, detail="Invalid payload")
        except stripe.error.SignatureVerificationError:
            log.warning("Stripe webhook received with invalid signature.")
            raise HTTPException(status_code=400, detail="Invalid signature")

        if event.type == "payment_intent.succeeded":
            intent = event.data.object
            payment_id_str = intent.metadata.get("order_payment_id")
            if not payment_id_str:
                log.error(f"Stripe event {event.id} (payment_intent.succeeded) missing 'order_payment_id' in metadata.")
                return {"status": "error", "reason": "Missing metadata"}

            try:
                payment_id = uuid.UUID(payment_id_str)
            except ValueError:
                log.error(f"Stripe event {event.id} had invalid 'order_payment_id' UUID: {payment_id_str}")
                return {"status": "error", "reason": "Invalid payment_id format"}

            payment = await self.db.get(OrderPayment, payment_id)
            if not payment:
                log.error(f"Stripe event {event.id} referenced non-existent OrderPayment: {payment_id}")
                return {"status": "error", "reason": "Payment record not found"}

            if payment.status not in ["pending_payment", "failed"]:
                log.info(f"Stripe event {event.id} for payment {payment_id} already processed. Status: {payment.status}")
                return {"status": "ok", "message": "Already processed"}

            payment.status = "succeeded"
            await self.db.flush()

            order = await self.db.get(OrderRecord, payment.order_id)
            if not order:
                 payment.status = "error"
                 payment.error_message = f"Associated OrderRecord {payment.order_id} not found."
                 await self.db.commit()
                 log.error(f"Stripe event {event.id} succeeded for payment {payment_id}, but OrderRecord {payment.order_id} not found.")
                 return {"status": "error", "reason": "OrderRecord not found"}

            printful_headers = {"Authorization": f"Bearer {PRINTFUL_API_KEY}"}
            # Run Printful submission in the background to not block the webhook response
            asyncio.create_task(_submit_order_to_printful(payment, order, printful_headers, self.db))
            log.info(f"Stripe event {event.id}: Payment {payment_id} succeeded. Queued Printful submission.")

        elif event.type == "payment_intent.payment_failed":
            intent = event.data.object
            payment_id_str = intent.metadata.get("order_payment_id")
            if payment_id_str:
                try:
                    payment = await self.db.get(OrderPayment, uuid.UUID(payment_id_str))
                    if payment:
                        payment.status = "failed"
                        payment.error_message = intent.last_payment_error.message if intent.last_payment_error else "Unknown payment failure"
                        await self.db.commit()
                        log.warning(f"Stripe event {event.id}: Payment {payment_id_str} failed: {payment.error_message}")
                except Exception as e:
                    log.error(f"Failed to process payment_intent.payment_failed event: {e}")

        return {"status": "received"}

    async def handle_image_upload(self, file: UploadFile) -> dict:
        """Handles user-uploaded images for placement."""
        path = await _save_temp_image(file) # Use the robust save function
        filename = os.path.basename(path)
        return {
            "status": "success",
            "filename": filename,
            "file_url": f"/api/ai/image/{filename}"
        }

    async def handle_image_placement(
        self, base_filename: str, overlay_filename: str, position: str
    ) -> dict:
        """Combines an uploaded image with a generated design."""
        base_path = os.path.join(UPLOAD_DIR, base_filename)
        overlay_path = os.path.join(UPLOAD_DIR, overlay_filename)

        if not os.path.exists(base_path) or not os.path.exists(overlay_path):
            raise HTTPException(status_code=404, detail="One or both image files not found.")

        result_path = _place_uploaded_image(base_path, overlay_path, position)
        filename = os.path.basename(result_path)
        return {
            "status": "success",
            "filename": filename,
            "image_url": f"/api/ai/image/{filename}"
        }

# ===================================================================
# FASTAPI ROUTER (Thin Wrappers)
# ===================================================================

# This single router will be imported and mounted by server.py
router = APIRouter(tags=["GlitchApe Central"])


# --- ADDED: Endpoint to create a new chat session ---
@router.post("/chat/start", status_code=status.HTTP_201_CREATED)
async def start_new_chat_session(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Starts a new chat session for the authenticated user and returns the session ID.
    """
    try:
        new_session = ChatSession(
            user_id=current_user.id,
            title=f"New Chat - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        db.add(new_session)
        await db.commit()
        await db.refresh(new_session)
        log.info(f"User {current_user.id} started new session {new_session.id}")
        return {"session_id": str(new_session.id)}
    except Exception as e:
        log.error(f"Failed to create new chat session for user {current_user.id}: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create new chat session: {e}")


# --- *** MODIFIED: This endpoint now accepts multipart/form-data *** ---
@router.post("/ai/chat", response_model=LiveChatResponse) # <-- Uses correct response
async def chat_with_ai(
    # Use Form() for text fields
    prompt: str = Form(...),
    # ***MODIFIED: Let FastAPI validate the UUID directly from the form field***
    session_id: uuid.UUID = Form(...),
    # Use File() for optional file upload
    uploaded_image: Optional[UploadFile] = File(None),

    # Standard dependencies
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Main AI chat endpoint. Accepts form-data."""
    # Run cleanup in background
    asyncio.create_task(_cleanup_expired_images())

    handler = GlitchApeCentralHandler(db=db, user=current_user)

    # Pass all Form/File data to the handler
    return await handler.handle_chat_message(
        session_id=session_id, # <-- Pass the validated UUID object directly
        prompt=prompt,
        uploaded_image=uploaded_image # Pass the UploadFile object directly
    )


# --- ADDED: Endpoint to get chat history for a session ---
@router.get("/chat/history/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieves the chat history for a specific session, verifying user ownership.
    """
    session = await db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    if session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied to this chat session")

    q = select(ChatMessage).where(
        ChatMessage.session_id == session_id
    ).order_by(ChatMessage.created_at.asc())

    r = await db.execute(q)
    messages = r.scalars().all()

    return ChatHistoryResponse(messages=messages)


@router.post("/orders/initiate-checkout", response_model=CheckoutResponse)
async def initiate_checkout(
    req: CheckoutRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """Starts the checkout process."""
    handler = GlitchApeCentralHandler(db=db, user=user)
    return await handler.initiate_checkout(req)


@router.post("/orders/stripe-webhook")
async def stripe_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """Handles incoming webhooks from Stripe."""
    handler = GlitchApeCentralHandler(db=db)
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    return await handler.handle_stripe_webhook(payload, sig_header)


@router.post("/ai/upload-image")
async def upload_user_image(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Handles user-uploaded images (e.g., a logo)."""
    handler = GlitchApeCentralHandler(db=db, user=current_user)
    return await handler.handle_image_upload(file)


@router.post("/ai/place-image")
async def place_image_on_design(
    base_filename: str = Form(...),
    overlay_filename: str = Form(...),
    position: str = Form("center"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Combines a user-uploaded image with a generated outfit image."""
    handler = GlitchApeCentralHandler(db=db, user=current_user)
    return await handler.handle_image_placement(base_filename, overlay_filename, position)


@router.get("/ai/image/{filename}")
async def get_image(filename: str):
    """Serves generated or uploaded images."""
    # Security: Prevent path traversal attacks
    if ".." in filename or filename.startswith(("/", "\\")):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    file_path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(file_path)