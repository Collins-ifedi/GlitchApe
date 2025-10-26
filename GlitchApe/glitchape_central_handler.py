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
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from fastapi import (
    APIRouter, Depends, HTTPException, Request,
    UploadFile, File, Form, status
)
from fastapi.responses import FileResponse
# --- UPDATED: Added Field for validation ---
from pydantic import BaseModel, Field
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

# ===================================================================
# CONFIGURATION
# ===================================================================

# --- AI (Chat Model: OpenRouter/Meta-Llama 4 Maverick) ---
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
if not LLAMA_API_KEY:
    raise RuntimeError("LLAMA_API_KEY environment variable not set for chat model")

# The OpenRouter API Endpoint
LLAMA_CHAT_API_URL = "https://openrouter.ai/api/v1/chat/completions"
LLAMA_CHAT_MODEL = "meta-llama/llama-4-maverick" # The model ID for Llama 4 Maverick

LLAMA_HEADERS = {
    "Authorization": f"Bearer {LLAMA_API_KEY}",
    "Content-Type": "application/json",
    # OpenRouter recommends a referrer; use an env var or a default.
    "HTTP-Referer": os.getenv("APP_DOMAIN", "https_glitchape_onrender_com"),
    "X-Title": "GlitchApe Central Handler"
}

# --- AI (Image Model: Hugging Face/Stable Diffusion) ---
# NOTE: Retaining the HF configuration for the image generation model.
HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    raise RuntimeError("HF_API_KEY environment variable not set for image model")

HF_STABLE_DIFFUSION_URL = os.getenv(
    "HF_STABLE_DIFFUSION_URL",
    "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
)
HF_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}


# --- Printful ---
PRINTFUL_API_KEY = os.getenv("PRINTFUL_API_KEY")
if not PRINTFUL_API_KEY:
    raise RuntimeError("PRINTFUL_API_KEY environment variable not set")
PRINTFUL_API_URL = "https://api.printful.com"

# --- Stripe ---
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
if not STRIPE_SECRET_KEY:
    raise RuntimeError("STRIPE_SECRET_KEY environment variable not set")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
if not STRIPE_WEBHOOK_SECRET:
    raise RuntimeError("STRIPE_WEBHOOK_SECRET environment variable not set")
stripe.api_key = STRIPE_SECRET_KEY

# --- File Storage ---
UPLOAD_DIR = "temp_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ===================================================================
# PYDANTIC SCHEMAS (Consolidated)
# ===================================================================

# --- AI Schemas ---
class AIRequest(BaseModel):
    # --- FIX: Changed min_length to 0 to allow empty string for new sessions ---
    session_id: str = Field(..., min_length=0)
    prompt: str = Field(..., min_length=1)

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

# --- Order Schemas ---
# UPDATED: Added min_length validation to required fields to prevent 422 errors
# caused by empty strings
class Recipient(BaseModel):
    name: str = Field(..., min_length=1)
    address1: str = Field(..., min_length=1)
    address2: Optional[str] = None
    city: str = Field(..., min_length=1)
    state_code: Optional[str] = None
    country_code: str = Field(..., min_length=2)
    zip: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None

class OrderItem(BaseModel):
    variant_id: int
    product_name: str
    quantity: int = 1

class CheckoutRequest(BaseModel):
    session_id: str = Field(..., min_length=1) # Checkout requires a non-empty session ID
    item: OrderItem
    recipient: Recipient

class CheckoutResponse(BaseModel):
    order_id: str
    payment_intent_id: str
    client_secret: str
    total_cost: float
    currency: str


# ===================================================================
# LLAMA TOOL DEFINITION
# ===================================================================

LLAMA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_design",
            "description": (
                "Call this function when the user definitively provides a detailed prompt for a new clothing design and expects a design preview. "
                "The user must be explicitly asking for a new design or confirming a design idea. "
                "DO NOT call for casual conversation or if the prompt is vague. This triggers the image generation model."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "design_prompt": {
                        "type": "string",
                        "description": "A refined, detailed, single-string prompt (e.g., 'a black hoodie with a cyberpunk glitch art graphic of an ape's head on the chest') for the image generation model. You must combine all design details into this one string."
                    }
                },
                "required": ["design_prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "initiate_order",
            "description": "Call this function when the user explicitly expresses intent to purchase or place an order for the *current design*. Use this when they say 'I want to order' or 'let's checkout'.",
            "parameters": {
                "type": "object",
                "properties": {},
            }
        }
    }
]

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
            except OSError:
                pass

async def _save_temp_image(file: UploadFile) -> str:
    """Saves an uploaded image temporarily. Returns the full local file path."""
    try:
        ext = file.filename.split(".")[-1]
        if ext.lower() not in ["png", "jpg", "jpeg", "webp"]:
            ext = "png"
        file_id = f"{uuid.uuid4()}.{ext}"
        file_path = os.path.join(UPLOAD_DIR, file_id)

        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        return file_path
    finally:
        file.file.close()

async def _call_llama_maverick(db: AsyncSession, session_id: uuid.UUID) -> Dict[str, Any]:
    """
    Calls the Llama 4 Maverick model via OpenRouter with the chat history
    and available tools to determine the next action.
    """

    # 1. Fetch chat history (max 15 messages for context/token limit)
    q = select(ChatMessage).where(
        ChatMessage.session_id == session_id
    ).order_by(ChatMessage.created_at.desc()).limit(15)

    r = await db.execute(q)
    history = r.scalars().all()

    # 2. Format history for Llama (OpenAI Chat API format)
    messages = []
    # Loop backwards to maintain chronological order
    for msg in reversed(history):
        # Only pass text messages to Llama for decision-making
        if msg.role in ["user", "ai"]:
            messages.append({"role": msg.role, "content": msg.content})

    # 3. Define the System Prompt
    system_prompt = (
        "You are GlitchApe, an AI clothing design assistant. Your role is to guide the user through designing and ordering custom apparel. "
        "You have two tools: `generate_design` (for image generation) and `initiate_order` (for checkout). "
        "1. Use `generate_design` ONLY when the user gives a detailed prompt for a new design. After generating a design, ask the user if they want to order or redesign. "
        "2. Use `initiate_order` ONLY when the user explicitly asks to order or checkout. "
        "3. For all other queries (greetings, questions, casual chat, feedback), respond conversationally and encourage them to refine their design or ask to see options. "
        "Keep responses brief, friendly, and focused on the design/order process. Always use the provided tools when appropriate."
    )

    api_messages = [
        {"role": "system", "content": system_prompt}
    ] + messages # Prepend system prompt to chat history

    payload = {
        "model": LLAMA_CHAT_MODEL, # <-- Use new model ID
        "messages": api_messages,
        "tools": LLAMA_TOOLS,
        "tool_choice": "auto" # Let Llama decide if a tool is needed
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                LLAMA_CHAT_API_URL, # <-- Use new API URL
                headers=LLAMA_HEADERS, # <-- Use new headers
                json=payload
            )
            response.raise_for_status()

            response_data = response.json()
            choice = response_data["choices"][0]

            # Check for tool call
            if choice.get("finish_reason") == "tool_calls":
                # Assuming single tool call for simplicity
                tool_call = choice["message"]["tool_calls"][0]
                function_name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"])
                return {"action": function_name, "args": arguments}

            # Check for text response
            return {"action": "chat", "response": choice["message"]["content"]}

    except httpx.HTTPStatusError as e:
        error_detail = e.response.json().get("error", {}).get("message", e.response.text)
        raise HTTPException(status_code=502, detail=f"LLM API failed ({e.response.status_code}): {error_detail}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM communication error: {e}")

# --- UPDATE: Swapped Gemini for Hugging Face Stable Diffusion REST API (Remains the same as last update) ---
async def _generate_design_view(prompt: str, view: str) -> str:
    """Generates a single 2D image view for a design using Stable Diffusion."""

    # NOTE: Stable Diffusion is not guaranteed to produce a transparent PNG.
    # This prompt attempts to get a clean image on a white background.
    full_prompt = (
        f"A single, realistic, 2D product image of the {view} view "
        f"of a {prompt}. Photorealistic, studio lighting, "
        f"on a pure white background. "
        f"Only show the clothing item, no models, mannequins, or shadows. "
        "High-resolution, print-quality file."
    )

    payload = {"inputs": full_prompt}

    try:
        # Increase timeout for image generation models
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                HF_STABLE_DIFFUSION_URL,
                headers=HF_HEADERS,
                json=payload
            )

        # Handle API errors
        if response.status_code != 200:
            error_detail = response.text
            try:
                # Try to parse HF error message
                error_data = response.json()
                detail = error_data.get("error", response.text)
                # Check for the common "model is loading" error
                if isinstance(detail, str) and "is currently loading" in detail:
                    raise HTTPException(status_code=503, detail="AI model is loading, please try again in a few moments.")
                error_detail = detail
            except:
                pass # Keep the original response.text if JSON parsing fails
            raise HTTPException(status_code=502, detail=f"AI image generation failed: {error_detail}")

        # Check if we got an image
        if not response.content or not response.headers.get('content-type', '').startswith('image/'):
             raise HTTPException(status_code=500, detail="AI returned an invalid response (not an image).")

        # Determine file extension (default to jpg)
        content_type = response.headers.get('content-type', 'image/jpeg')
        ext = content_type.split('/')[-1]
        if ext not in ['jpeg', 'jpg', 'png', 'webp']:
            ext = 'jpg'

        # Save the image data to a file
        file_id = f"{uuid.uuid4()}.{ext}"
        file_path = os.path.join(UPLOAD_DIR, file_id)
        with open(file_path, "wb") as f:
            f.write(response.content)

        return file_path

    except httpx.ReadTimeout:
        raise HTTPException(status_code=504, detail="AI image generation timed out.")
    except Exception as e:
        # Re-raise HTTPException if we threw it, otherwise wrap the error
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"AI image generation error: {e}")

def _get_required_views(prompt: str) -> List[str]:
    """Determines which clothing views to generate based on the prompt."""
    prompt_lower = prompt.lower()
    if "hoodie" in prompt_lower or "shirt" in prompt_lower or "t-shirt" in prompt_lower or "jacket" in prompt_lower:
        return ["front", "back", "left_sleeve", "right_sleeve"]
    if "trousers" in prompt_lower or "pants" in prompt_lower or "jeans" in prompt_lower:
        return ["front", "back"]
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

# --- Order Logic Helpers ---

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
    # Note: image_url is relative, e.g., /api/ai/image/uuid.png
    # We need the local path.
    local_path = os.path.join(UPLOAD_DIR, os.path.basename(image_msg.image_url))

    if not os.path.exists(local_path):
        raise HTTPException(status_code=404, detail=f"Image file not found: {local_path}")

    view_name = image_msg.content.replace("Generated image: ", "").strip()

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            with open(local_path, "rb") as f:
                # Dynamically detect content type for printful
                file_ext = local_path.split('.')[-1]
                mime_type = f"image/{file_ext}" if file_ext in ['png', 'jpg', 'jpeg'] else 'image/jpeg'
                files = {"file": (os.path.basename(local_path), f, mime_type)}
                resp = await client.post(
                    f"{PRINTFUL_API_URL}/files",
                    headers=headers,
                    files=files
                )
            resp.raise_for_status()
            data = resp.json().get("result")
            if not data or not data.get("id"):
                raise Exception("Printful API did not return a file ID.")
            return {"view": view_name, "id": data["id"]}
    except Exception as e:
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
            item_cost = float(prod_data.get("price")) * item.quantity
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
            rate_data = ship_resp.json().get("result")[0]
            shipping_cost = float(rate_data.get("rate"))

            total = item_cost + shipping_cost
            return {"total_cents": int(total * 100), "currency": currency}
    except Exception as e:
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
    except Exception as e:
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

    async def handle_chat_message(self, session_id: uuid.UUID, prompt: str) -> AIResponse:
        """
        Orchestrates the AI chat and design generation flow by delegating the decision
        (chat, generate image, or initiate order) to Llama Maverick.
        """

        # 1. Verify session ownership and save user message
        if not self.user:
            raise HTTPException(status_code=401, detail="User not authenticated")

        session = await self.db.get(ChatSession, session_id)
        if not session or session.user_id != self.user.id:
            raise HTTPException(status_code=404, detail="Chat session not found or access denied")

        # Save user's message so it is included in the history passed to Llama
        user_msg = ChatMessage(
            session_id=session_id, user_id=self.user.id, role="user", content=prompt
        )
        self.db.add(user_msg)
        await self.db.flush() # Make sure the new message is in the DB session for history fetch

        ai_text_response = ""
        image_previews: List[ImagePreview] = []
        # --- FIX: Read the current state from the session object ---
        new_state = session.state

        try:
            # 2. Call Llama Maverick to determine the action
            llama_result = await _call_llama_maverick(self.db, session_id)

            action = llama_result["action"]

            if action == "generate_design":
                # --- Action: Generate Design ---
                design_prompt = llama_result["args"]["design_prompt"]

                # Input validation
                if not design_prompt or len(design_prompt) < 10:
                     ai_text_response = "I'm sorry, I couldn't interpret a clear design request. Could you be more specific about the clothing type and design you want?"
                     new_state = "designing"
                else:
                    views_to_gen = _get_required_views(design_prompt)
                    ai_text_response = f"Got it! I'm generating the {', '.join(views_to_gen)} views for your '{design_prompt}'. Here's the preview:"

                    tasks = [_generate_design_view(design_prompt, view) for view in views_to_gen]
                    generated_paths = await asyncio.gather(*tasks)

                    for i, local_path in enumerate(generated_paths):
                        filename = os.path.basename(local_path)
                        public_url = f"/api/ai/image/{filename}"
                        view_name = views_to_gen[i]

                        image_previews.append(ImagePreview(view_name=view_name, url=public_url))

                        ai_img_msg = ChatMessage(
                            session_id=session_id, user_id=self.user.id, role="ai",
                            content=f"Generated image: {view_name}", image_url=public_url
                        )
                        self.db.add(ai_img_msg)

                    ai_text_response += "\n\nDo you want to **place an order** for this design, or would you like to **redesign** a section?"
                    new_state = "awaiting_feedback"

            elif action == "initiate_order":
                # --- Action: Initiate Order ---
                ai_text_response = "Great! I'm preparing your most recent designs for the order. Please proceed to checkout by filling out the order form."
                new_state = "finalized"

            elif action == "chat":
                # --- Action: Casual Chat / Conversational ---
                ai_text_response = llama_result["response"]
                # The state (new_state) is already set to the session's current state, so no change is needed.

        except Exception as e:
            # If the LLM call or image generation fails, log and respond gracefully
            ai_text_response = f"I tried to process your request, but an internal error occurred: {e}"
            new_state = "designing" # Revert to a safe state on error

        # 3. Save the main AI text response
        ai_text_msg = ChatMessage(
            session_id=session_id, user_id=self.user.id, role="ai", content=ai_text_response
        )
        self.db.add(ai_text_msg)

        # 4. Commit all changes
        # --- FIX: Persist the new state to the database session ---
        session.state = new_state
        await self.db.commit()

        # 5. Return the consolidated response
        return AIResponse(
            session_id=str(session_id),
            ai_message=ai_text_response,
            images=image_previews,
            conversation_state=new_state
        )

    async def initiate_checkout(self, req: CheckoutRequest) -> CheckoutResponse:
        """Orchestrates the checkout initiation flow."""
        if not self.user:
            raise HTTPException(status_code=401, detail="User not authenticated")

        try:
            # Pydantic validation on CheckoutRequest ensures session_id is not empty
            session_uuid = uuid.UUID(req.session_id)
        except ValueError:
            # If the string is present but not a valid UUID, raise 400
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
            raise HTTPException(status_code=500, detail="Stripe webhook secret is not configured.")

        try:
            event = stripe.Webhook.construct_event(
                payload=payload, sig_header=sig_header, secret=STRIPE_WEBHOOK_SECRET
            )
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid payload")
        except stripe.error.SignatureVerificationError:
            raise HTTPException(status_code=400, detail="Invalid signature")

        if event.type == "payment_intent.succeeded":
            intent = event.data.object
            payment_id = intent.metadata.get("order_payment_id")
            if not payment_id:
                return {"status": "error", "reason": "Missing metadata"}

            payment = await self.db.get(OrderPayment, uuid.UUID(payment_id))
            if not payment:
                return {"status": "error", "reason": "Payment record not found"}

            if payment.status not in ["pending_payment", "failed"]:
                return {"status": "ok", "message": "Already processed"}

            payment.status = "succeeded"
            await self.db.flush()

            order = await self.db.get(OrderRecord, payment.order_id)
            if not order:
                 payment.status = "error"
                 payment.error_message = f"Associated OrderRecord {payment.order_id} not found."
                 await self.db.commit()
                 return {"status": "error", "reason": "OrderRecord not found"}

            printful_headers = {"Authorization": f"Bearer {PRINTFUL_API_KEY}"}
            await _submit_order_to_printful(payment, order, printful_headers, self.db)

        elif event.type == "payment_intent.payment_failed":
            intent = event.data.object
            payment_id = intent.metadata.get("order_payment_id")
            if payment_id:
                payment = await self.db.get(OrderPayment, uuid.UUID(payment_id))
                if payment:
                    payment.status = "failed"
                    payment.error_message = intent.last_payment_error.message
                    await self.db.commit()

        return {"status": "received"}

    async def handle_image_upload(self, file: UploadFile) -> dict:
        """Handles user-uploaded images for placement."""
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type.")

        path = await _save_temp_image(file)
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
    (Fix for 500 DB error is here)
    """
    try:
        # Create a new session
        new_session = ChatSession(
            user_id=current_user.id,
            title=f"New Chat - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            state="designing" # <-- FIX: Added default state to prevent DB IntegrityError
        )
        db.add(new_session)
        await db.commit()
        await db.refresh(new_session)

        return {"session_id": str(new_session.id)}
    except Exception as e:
        await db.rollback()
        # NOTE: A 422 here would likely be due to a client sending an invalid request body
        # or an upstream dependency failure (401), but we catch everything as 500 for DB failure.
        raise HTTPException(status_code=500, detail=f"Failed to create new chat session: {e}")


@router.post("/ai/chat", response_model=AIResponse)
async def chat_with_ai(
    request: AIRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Main AI chat endpoint."""
    # Run cleanup in background
    asyncio.create_task(_cleanup_expired_images())

    handler = GlitchApeCentralHandler(db=db, user=current_user)
    try:
        # Pydantic validation handles 422 for missing/malformed body fields (prompt, session_id)
        # Handle the case where session_id might be an empty string, convert to UUID if valid
        if request.session_id:
            session_uuid = uuid.UUID(request.session_id)
        else:
            # If session_id is empty, it implies a new session. 
            # We rely on handle_chat_message to manage this case, but still need a UUID placeholder.
            # Ideally, the frontend should call /chat/start first.
            # For robustness, we might want to create a session here if empty, but 
            # let's assume the frontend flow is correct for now.
            # If the Pydantic model allows empty string, we can raise a 400 here if it's empty
            # as the chat handler expects a valid UUID.
            raise HTTPException(status_code=400, detail="Session ID cannot be empty for chat.")

    except ValueError:
        # Catch Bad UUID format, which would be a 400 Bad Request
        raise HTTPException(status_code=400, detail="Invalid session_id format.")
    except HTTPException as e: # Re-raise known HTTPExceptions
        raise e
    except Exception as e: # Catch unexpected errors during UUID conversion
        raise HTTPException(status_code=500, detail=f"Error processing session ID: {e}")


    return await handler.handle_chat_message(session_uuid, request.prompt)


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
    # 1. Verify session exists and belongs to the user
    session = await db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    if session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied to this chat session")

    # 2. Query for messages
    q = select(ChatMessage).where(
        ChatMessage.session_id == session_id
    ).order_by(ChatMessage.created_at.asc()) # History should be in ascending order

    r = await db.execute(q)
    messages = r.scalars().all()

    # 3. Format and return response
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
    base_filename: str = Form(..., min_length=1),
    overlay_filename: str = Form(..., min_length=1),
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
    file_path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(file_path) or ".." in filename or filename.startswith("/"):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(file_path)