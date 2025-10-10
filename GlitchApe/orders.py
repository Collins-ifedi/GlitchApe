# orders.py
import stripe
import httpx
import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request, Header, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from datetime import datetime

from db import get_db
from models import Order, OrderItem, User, Design, ReferralPayout, OrderStatus
from settings import settings
from auth import get_current_user

# --- Configuration & Setup ---
router = APIRouter(prefix="/orders", tags=["Orders"])
stripe.api_key = settings.STRIPE_SECRET_KEY

# --- Pydantic Schemas for Data Validation ---

class CheckoutItemIn(BaseModel):
    design_id: uuid.UUID
    quantity: int

class CheckoutSessionIn(BaseModel):
    items: List[CheckoutItemIn]
    shipping_address_id: uuid.UUID

class CheckoutSessionOut(BaseModel):
    checkout_url: str

class OrderItemOut(BaseModel):
    design_id: uuid.UUID | None
    quantity: int
    price_per_item: float

    class Config:
        orm_mode = True

class OrderOut(BaseModel):
    id: uuid.UUID
    status: OrderStatus
    amount: float
    created_at: datetime
    items: List[OrderItemOut]

    class Config:
        orm_mode = True


# --- Order Creation Endpoint ---

@router.post("/checkout", response_model=CheckoutSessionOut, status_code=status.HTTP_201_CREATED)
async def create_checkout_session(
    payload: CheckoutSessionIn,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Creates a Stripe Checkout session for a cart of items.
    1. Validates the designs and calculates the total price.
    2. Creates a 'pending' Order and associated OrderItems in the database.
    3. Creates a Stripe session, embedding our internal order ID in the metadata.
    4. Returns the Stripe URL for the client to redirect to.
    """
    if not settings.STRIPE_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Stripe is not configured.")
    if not payload.items:
        raise HTTPException(status_code=400, detail="Checkout must contain at least one item.")

    line_items = []
    order_items = []
    total_amount = 0

    # Create a pending order first
    new_order = Order(
        id=uuid.uuid4(),
        user_id=current_user.id,
        shipping_address_id=payload.shipping_address_id,
        status=OrderStatus.PENDING,
        amount=0,  # Will be updated after calculation
    )

    for item in payload.items:
        design = await db.get(Design, item.design_id)
        if not design or design.user_id != current_user.id:
            raise HTTPException(status_code=404, detail=f"Design with ID {item.design_id} not found or does not belong to user.")
        
        # NOTE: In a real application, price should come from a trusted source
        # (e.g., a products table), not a hardcoded value.
        # This prevents users from manipulating prices.
        item_price = 25.00  # Example price: $25.00
        total_amount += item_price * item.quantity

        line_items.append({
            "price_data": {
                "currency": "usd",
                "product_data": {"name": f"Custom Design - {design.prompt[:20]}..."},
                "unit_amount": int(item_price * 100),  # Price in cents
            },
            "quantity": item.quantity,
        })
        order_items.append(OrderItem(
            order=new_order,
            design_id=design.id,
            quantity=item.quantity,
            price_per_item=item_price
        ))

    new_order.amount = total_amount
    db.add(new_order)
    db.add_all(order_items)
    
    try:
        checkout_session = await stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=line_items,
            mode="payment",
            success_url=f"{settings.FRONTEND_URL}/order/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{settings.FRONTEND_URL}/cart",
            metadata={
                "user_id": str(current_user.id),
                "order_id": str(new_order.id) # Use our internal UUID
            }
        )
        new_order.stripe_session_id = checkout_session.id
        await db.commit()
        return CheckoutSessionOut(checkout_url=checkout_session.url)

    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Stripe error: {str(e)}")


# --- Stripe Webhook Endpoint ---

@router.post("/webhook", include_in_schema=False)
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Handles incoming webhooks from Stripe.
    - Verifies the webhook signature for security.
    - Processes the 'checkout.session.completed' event.
    - Updates the order status, places the order with Printful, and triggers referral payouts.
    """
    if not stripe_signature:
        raise HTTPException(status_code=400, detail="Missing Stripe-Signature header.")
    
    try:
        payload = await request.body()
        event = stripe.Webhook.construct_event(
            payload=payload, sig_header=stripe_signature, secret=settings.STRIPE_WEBHOOK_SECRET
        )
    except ValueError as e: # Invalid payload
        raise HTTPException(status_code=400, detail=f"Invalid payload: {e}")
    except stripe.error.SignatureVerificationError as e:
        raise HTTPException(status_code=400, detail=f"Invalid signature: {e}")

    # Handle the checkout.session.completed event
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        order_id_str = session.get('metadata', {}).get('order_id')
        
        if not order_id_str:
            return {"status": "ignored", "reason": "No order_id in metadata"}
        
        order = await db.get(Order, uuid.UUID(order_id_str))
        if not order:
            return {"status": "error", "reason": f"Order {order_id_str} not found"}

        if order.status == OrderStatus.PENDING:
            order.status = OrderStatus.PAID
            order.paid_at = datetime.utcnow()
            order.stripe_payment_intent = session.get('payment_intent')
            await db.commit()

            # --- Post-Payment Actions ---
            # 1. Place the order with Printful
            await place_printful_order(order, db)
            # 2. Handle referral payout if applicable
            await handle_referral_payout(order, db)

    return {"status": "success"}


# --- Helper Functions for Post-Payment Actions ---

async def place_printful_order(order: Order, db: AsyncSession):
    """Places the confirmed order with the Printful API."""
    # (Implementation details for Printful API call)
    # This would involve building the request from order.items
    # and order.shipping_address, then updating order.printful_order_id
    pass

async def handle_referral_payout(order: Order, db: AsyncSession):
    """If the user was referred, creates a payout for the referrer."""
    user = order.user
    if not user.referred_by_id:
        return

    COMMISSION_RATE = 0.10  # 10%
    commission_amount = order.amount * COMMISSION_RATE
    
    payout = ReferralPayout(
        referrer_id=user.referred_by_id,
        referred_user_id=user.id,
        order_id=order.id,
        amount=commission_amount,
        status="pending"
    )
    db.add(payout)
    await db.commit()


# --- Order History Endpoint ---

@router.get("/my-orders", response_model=List[OrderOut])
async def get_my_orders(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Retrieves all orders for the currently logged-in user."""
    query = (
        select(Order)
        .where(Order.user_id == current_user.id)
        .options(selectinload(Order.items)) # Eager load items to avoid extra queries
        .order_by(Order.created_at.desc())
    )
    result = await db.execute(query)
    orders = result.scalars().unique().all()
    return orders