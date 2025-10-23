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
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from PIL import Image
# --- FIX: Correct import for the Google GenAI library ---
import google.generativeai as genai

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

# --- AI ---
AI_API_KEY = os.getenv("AI_API_KEY")
if not AI_API_KEY:
    raise RuntimeError("AI_API_KEY environment variable not set")
# --- FIX: This line will now work with the correct import ---
genai.configure(api_key=AI_API_KEY)
# --- UPDATE: Using gemini-1.5-flash-latest as requested ---
ai_client = genai.GenerativeModel("gemini-1.5-flash-latest")

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
    prompt: str
    session_id: str

class ImagePreview(BaseModel):
    view_name: str
    url: str

class AIResponse(BaseModel):
    session_id: str
    ai_message: str
    images: List[ImagePreview] = []
    conversation_state: str = "awaiting_feedback"

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

# --- UPDATE: Replaced placeholder with production-ready image generation logic ---
async def _generate_design_view(prompt: str, view: str) -> str:
    """Generates a single, backgroundless 2D image view for a design."""
    full_prompt = (
        f"Generate a single, realistic, 2D product image of the {view} view "
        f"of a {prompt}. The image must have a transparent background (PNG format). "
        f"Only show the clothing item, no models, mannequins, or shadows. "
        "The output must be a high-resolution, print-quality file."
    )
    try:
        # Generate the content using the specified model
        response = await ai_client.generate_content_async(full_prompt)
        
        # Robustly check for the image part in the response
        if not response.parts:
            raise HTTPException(status_code=500, detail="AI returned an empty response.")

        # Find the first part that is an image
        image_part = next((p for p in response.parts if p.mime_type.startswith("image/")), None)

        if not image_part:
            # If no image is found, the AI might have responded with text (e.g., a safety refusal)
            text_part = response.text or "[No text content returned]"
            raise HTTPException(status_code=500, detail=f"AI failed to generate an image. Response: {text_part}")
        
        # Get the raw image bytes from the inline_data field
        image_data = image_part.inline_data.data
        
        # Save the image data to a file
        file_id = f"{uuid.uuid4()}.png"
        file_path = os.path.join(UPLOAD_DIR, file_id)
        with open(file_path, "wb") as f:
            f.write(image_data)
        
        return file_path
        
    except Exception as e:
        # Re-raise HTTPException if we threw it, otherwise wrap the error
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=502, detail=f"AI image generation error: {e}")

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
                files = {"file": (os.path.basename(local_path), f, "image/png")}
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
        """Orchestrates the AI chat and design generation flow."""
        
        # 1. Verify session ownership (User must be set)
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

        # 3. Process request and determine AI action
        ai_text_response = ""
        image_previews: List[ImagePreview] = []
        new_state = "awaiting_feedback"
        prompt_lower = prompt.lower()

        try:
            if "order" in prompt_lower or "place order" in prompt_lower:
                ai_text_response = "Great! I'm preparing your designs for the order. Please proceed to checkout."
                new_state = "finalized"
            
            elif "redesign" in prompt_lower or "change" in prompt_lower:
                ai_text_response = "Okay, what part would you like to change, and what should it look like?"
                new_state = "designing"

            else:
                views_to_gen = _get_required_views(prompt_lower)
                ai_text_response = f"I'm generating the {', '.join(views_to_gen)} views for your design. Here's the preview:"
                
                tasks = [_generate_design_view(prompt, view) for view in views_to_gen]
                generated_paths = await asyncio.gather(*tasks)
                
                for i, local_path in enumerate(generated_paths):
                    filename = os.path.basename(local_path)
                    public_url = f"/api/ai/image/{filename}" # The full path frontend will use
                    view_name = views_to_gen[i]
                    
                    image_previews.append(ImagePreview(view_name=view_name, url=public_url))
                    
                    ai_img_msg = ChatMessage(
                        session_id=session_id, user_id=self.user.id, role="ai",
                        content=f"Generated image: {view_name}", image_url=public_url
                    )
                    self.db.add(ai_img_msg)

                ai_text_response += "\n\nIs it OK to place an order, or would you like to redesign a section?"
                new_state = "awaiting_feedback"

        except Exception as e:
            ai_text_response = f"I tried to process your request, but an error occurred: {e}"
            new_state = "designing"

        # 4. Save the main AI text response
        ai_text_msg = ChatMessage(
            session_id=session_id, user_id=self.user.id, role="ai", content=ai_text_response
        )
        self.db.add(ai_text_msg)
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
        session_uuid = uuid.UUID(request.session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session_id format.")
        
    return await handler.handle_chat_message(session_uuid, request.prompt)


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
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    if not os.path.exists(file_path) or ".." in filename or filename.startswith("/"):
        raise HTTPException(status_code=404, detail="Image not found")
        
    return FileResponse(file_path)
