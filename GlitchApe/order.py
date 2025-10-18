# order.py
"""
Handles Order Initiation, Printful API, and Stripe Checkout.

This module provides endpoints to:
1.  Initiate a checkout:
    - Gathers the latest designs from ai.py's session.
    - Uploads designs to Printful.
    - Calculates shipping and product costs from Printful.
    - Creates a Stripe Payment Intent.
    - Saves a local 'OrderRecord' and 'OrderPayment' (new model).
2.  Handle Stripe webhooks:
    - Verifies 'payment_intent.succeeded'.
    - Submits the paid-for order to Printful for fulfillment.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import uuid
import json
from pathlib import Path
import httpx
import stripe
import asyncio
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import Column, String, Integer, Text, select
from sqlalchemy.dialects.postgresql import UUID as SA_UUID
from sqlalchemy.sql.sqltypes import ForeignKey

# Import core components from server.py
from server import (
    get_db,
    get_current_user,
    User,
    ChatSession,
    ChatMessage,
    OrderRecord,
    Base  # Import Base to declare new model
)

# ===================================================================
# CONFIGURATION
# ===================================================================

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

# --- File Storage (Must match ai.py) ---
AI_IMAGE_DIR = "temp_images"

# --- Router ---
router = APIRouter(prefix="/orders", tags=["Orders"])


# ===================================================================
# NEW DATABASE MODEL
# ===================================================================

class OrderPayment(Base):
    """
    Augments the base OrderRecord with payment, state, and fulfillment data.
    This model is necessary because the original OrderRecord is too simple.
    
    *** IMPORTANT ***
    You must ensure server.py's `Base.metadata.create_all` call
    is aware of this model so the 'order_payments' table is created.
    """
    __tablename__ = "order_payments"
    id = Column(SA_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Links to the original OrderRecord
    order_id = Column(SA_UUID(as_uuid=True), ForeignKey("orders.id"), nullable=False, unique=True)
    user_id = Column(SA_UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    
    # Stripe Info
    payment_intent_id = Column(String(255), nullable=False, index=True, unique=True)
    total_cost_cents = Column(Integer, nullable=False)
    currency = Column(String(10), nullable=False, default="usd")
    
    # State Management
    status = Column(String(50), nullable=False, default="pending_payment", index=True) 
    # Values: pending_payment, succeeded, failed, submitted_to_printful, error
    
    # Fulfillment Data
    recipient_json = Column(Text, nullable=False)           # JSON blob of Recipient schema
    printful_file_ids_json = Column(Text, nullable=False) # JSON map: {"front": 123, "back": 456}
    variant_id = Column(Integer, nullable=False)          # Printful product variant ID
    error_message = Column(Text, nullable=True)           # For logging failures


# ===================================================================
# PYDANTIC SCHEMAS
# ===================================================================

class Recipient(BaseModel):
    """Printful-compatible recipient schema."""
    name: str
    address1: str
    address2: Optional[str] = None
    city: str
    state_code: Optional[str] = None # Required for US/CA
    country_code: str # e.g., "US"
    zip: Optional[str] = None # Required for US/CA
    email: Optional[str] = None
    phone: Optional[str] = None

class OrderItem(BaseModel):
    """Printful-compatible item schema."""
    variant_id: int # e.g., 7671 (Printful's ID for "Bella + Canvas 3001", Black, M)
    product_name: str # Our internal name, e.g., "User's Custom Hoodie"
    quantity: int = 1

class CheckoutRequest(BaseModel):
    """Request from frontend to start checkout."""
    session_id: str # The chat session UUID
    item: OrderItem
    recipient: Recipient

class CheckoutResponse(BaseModel):
    """Response sent to frontend to initialize Stripe.js."""
    order_id: str
    payment_intent_id: str
    client_secret: str
    total_cost: float # In dollars (e.g., 25.99)
    currency: str


# ===================================================================
# PRINTFUL API HELPERS
# ===================================================================

async def _get_design_images_from_session(
    session_id: uuid.UUID, user_id: uuid.UUID, db: AsyncSession
) -> List[ChatMessage]:
    """
    Finds the *latest set* of generated design images from the chat session.
    It looks for 'ai' messages with image_urls and "Generated image: ..." content.
    """
    q = select(ChatMessage).where(
        ChatMessage.session_id == session_id,
        ChatMessage.user_id == user_id,
        ChatMessage.role == "ai",
        ChatMessage.image_url.isnot(None),
        ChatMessage.content.like("Generated image: %")
    ).order_by(ChatMessage.created_at.desc())
    
    r = await db.execute(q)
    all_images = r.scalars().all()
    
    # De-duplicate, keeping only the most recent for each view name
    latest_images: Dict[str, ChatMessage] = {}
    for img in all_images:
        # e.g., "Generated image: front" -> "front"
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
    """Uploads a single local file from ai.py's temp_images to Printful."""
    local_path = os.path.join(AI_IMAGE_DIR, os.path.basename(image_msg.image_url))
    
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
                
            resp.raise_for_status() # Raise exception for 4xx/5xx
            
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
            # 1. Get Product Cost
            prod_resp = await client.get(
                f"{PRINTFUL_API_URL}/products/variant/{item.variant_id}", 
                headers=headers
            )
            prod_resp.raise_for_status()
            prod_data = prod_resp.json().get("result")
            item_cost = float(prod_data.get("price")) * item.quantity
            currency = prod_data.get("currency", "usd").lower()

            # 2. Get Shipping Cost
            shipping_payload = {
                "recipient": recipient.model_dump(exclude_none=True),
                "items": [{
                    "variant_id": item.variant_id,
                    "quantity": item.quantity
                }]
            }
            ship_resp = await client.post(
                f"{PRINTFUL_API_URL}/shipping/rates",
                headers=headers,
                json=shipping_payload
            )
            ship_resp.raise_for_status()
            # Select the first (usually cheapest) shipping rate
            rate_data = ship_resp.json().get("result")[0]
            shipping_cost = float(rate_data.get("rate"))

            total = item_cost + shipping_cost
            
            return {
                "total_cents": int(total * 100),
                "currency": currency,
            }
            
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to calculate Printful costs: {e}")


async def _submit_order_to_printful(
    payment: OrderPayment, 
    order: OrderRecord, 
    headers: Dict[str, str], 
    db: AsyncSession
):
    """Submits the final, confirmed order to Printful for fulfillment."""
    try:
        recipient = json.loads(payment.recipient_json)
        file_id_map = json.loads(payment.printful_file_ids_json)
        
        # Convert {"front": 123} to [{"type": "front", "id": 123}]
        printful_files = [
            {"type": view, "id": file_id} for view, file_id in file_id_map.items()
        ]

        # Build the final order payload
        order_item = {
            "variant_id": payment.variant_id,
            "quantity": 1, # Assuming 1 for now, enhance OrderItem/Payment if needed
            "files": printful_files,
            "name": order.product_name,
            "external_id": str(order.id) # Link to our internal OrderRecord
        }
        
        payload = {
            "recipient": recipient,
            "items": [order_item],
            "external_id": str(order.id)
        }
        
        async with httpx.AsyncClient(timeout=60) as client:
            # Note: We do NOT confirm. We create a draft, then confirm.
            # This is safer.
            resp = await client.post(
                f"{PRINTFUL_API_URL}/orders",
                headers=headers,
                json=payload
            )
            resp.raise_for_status()
            draft_order = resp.json().get("result")
            printful_order_id = draft_order.get("id")

            # 2. Confirm the draft order
            confirm_resp = await client.post(
                f"{PRINTFUL_API_URL}/orders/{printful_order_id}/confirm",
                headers=headers
            )
            confirm_resp.raise_for_status()
            confirmed_order = confirm_resp.json().get("result")
            
            # 3. Update our local records
            order.printful_order_id = str(confirmed_order.get("id"))
            payment.status = "submitted_to_printful"
            await db.commit()

    except Exception as e:
        # Log error and mark payment for manual review
        payment.status = "error"
        payment.error_message = f"Printful submission failed: {e}"
        await db.commit()
        # Optionally, send an alert to admins here


# ===================================================================
# ENDPOINT 1: INITIATE CHECKOUT
# ===================================================================

@router.post("/initiate-checkout", response_model=CheckoutResponse)
async def initiate_checkout(
    req: CheckoutRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """
    Starts the checkout process.
    1. Gets design images from the session.
    2. Uploads them to Printful.
    3. Calculates total cost from Printful.
    4. Creates a Stripe Payment Intent.
    5. Saves local DB records.
    """
    try:
        session_uuid = uuid.UUID(req.session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session_id format.")

    printful_headers = {"Authorization": f"Bearer {PRINTFUL_API_KEY}"}
    
    # 1. Get Images from Chat
    images = await _get_design_images_from_session(session_uuid, user.id, db)
    
    # 2. Upload to Printful in parallel
    upload_tasks = [_upload_to_printful(img, printful_headers) for img in images]
    upload_results = await asyncio.gather(*upload_tasks)
    
    # Map of {"front": 12345, "back": 67890}
    file_id_map = {res["view"]: res["id"] for res in upload_results}
    file_ids_json = json.dumps(file_id_map)

    # 3. Get Costs from Printful
    costs = await _get_printful_costs(req.item, req.recipient, printful_headers)
    
    # 4. Create local DB records
    async with db.begin_nested(): # Use savepoint for transaction
        # Create base OrderRecord (from server.py)
        new_order = OrderRecord(
            user_id=user.id,
            product_name=req.item.product_name,
            # This field is flawed, but we'll populate it with the first ChatMessage ID
            image_id=images[0].id 
        )
        db.add(new_order)
        await db.flush() # Get the new_order.id
        
        # 5. Create Stripe Payment Intent
        try:
            intent = stripe.PaymentIntent.create(
                amount=costs["total_cents"],
                currency=costs["currency"],
                automatic_payment_methods={"enabled": True},
                metadata={
                    "order_id": str(new_order.id), # Link to OrderRecord
                    "user_id": str(user.id)
                }
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Stripe Payment Intent creation failed: {e}")

        # Create our new OrderPayment record
        new_payment = OrderPayment(
            order_id=new_order.id,
            user_id=user.id,
            payment_intent_id=intent.id,
            total_cost_cents=costs["total_cents"],
            currency=costs["currency"],
            recipient_json=req.recipient.model_dump_json(),
            printful_file_ids_json=file_ids_json,
            variant_id=req.item.variant_id,
            status="pending_payment"
        )
        db.add(new_payment)
        
        # We must link the payment_intent_id to our *OrderPayment* record
        # so the webhook can find it.
        stripe.PaymentIntent.modify(
            intent.id,
            metadata={
                "order_id": str(new_order.id),
                "order_payment_id": str(new_payment.id), # The crucial link
                "user_id": str(user.id)
            }
        )

        await db.commit() # Commit the transaction

    # 6. Return response to frontend
    return CheckoutResponse(
        order_id=str(new_order.id),
        payment_intent_id=intent.id,
        client_secret=intent.client_secret,
        total_cost=round(costs["total_cents"] / 100, 2),
        currency=costs["currency"]
    )


# ===================================================================
# ENDPOINT 2: STRIPE WEBHOOK
# ===================================================================

@router.post("/stripe-webhook")
async def stripe_webhook(
    request: Request, 
    db: AsyncSession = Depends(get_db)
):
    """
    Handles incoming webhooks from Stripe to confirm payment
    and trigger order fulfillment.
    """
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="Stripe webhook secret is not configured.")
        
    try:
        event = stripe.Webhook.construct_event(
            payload=payload, sig_header=sig_header, secret=STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        # Invalid payload
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        # Invalid signature
        raise HTTPException(status_code=400, detail="Invalid signature")

    # Handle the event
    if event.type == "payment_intent.succeeded":
        intent = event.data.object
        payment_id = intent.metadata.get("order_payment_id")
        
        if not payment_id:
            print(f"Webhook Error: payment_intent.succeeded {intent.id} has no order_payment_id in metadata.")
            return {"status": "error", "reason": "Missing metadata"}
            
        payment = await db.get(OrderPayment, uuid.UUID(payment_id))
        
        if not payment:
            print(f"Webhook Error: OrderPayment record {payment_id} not found.")
            return {"status": "error", "reason": "Payment record not found"}

        # Idempotency check:
        if payment.status not in ["pending_payment", "failed"]:
            print(f"Webhook Info: OrderPayment {payment_id} already processed. Status: {payment.status}")
            return {"status": "ok", "message": "Already processed"}

        # Payment successful! Update status and submit to Printful
        payment.status = "succeeded"
        await db.flush()
        
        order = await db.get(OrderRecord, payment.order_id)
        if not order:
             payment.status = "error"
             payment.error_message = f"Associated OrderRecord {payment.order_id} not found."
             await db.commit()
             return {"status": "error", "reason": "OrderRecord not found"}

        # This is the magic! Submit to Printful *after* payment is confirmed.
        printful_headers = {"Authorization": f"Bearer {PRINTFUL_API_KEY}"}
        await _submit_order_to_printful(payment, order, printful_headers, db)
        
    elif event.type == "payment_intent.payment_failed":
        intent = event.data.object
        payment_id = intent.metadata.get("order_payment_id")
        
        if payment_id:
            payment = await db.get(OrderPayment, uuid.UUID(payment_id))
            if payment:
                payment.status = "failed"
                payment.error_message = intent.last_payment_error.message
                await db.commit()

    return {"status": "received"}
