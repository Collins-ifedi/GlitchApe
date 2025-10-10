# referrals.py
import stripe
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func
from typing import List
from datetime import datetime
from pydantic import BaseModel, Field

from db import get_db
from models import User, ReferralPayout
from auth import get_current_user
from settings import settings

# --- Configuration & Setup ---
router = APIRouter(prefix="/referrals", tags=["Referrals"])

# Configure Stripe for asynchronous operations
stripe.api_key = settings.STRIPE_SECRET_KEY


# --- Pydantic Schemas for Data Validation ---

class ReferralStatsOut(BaseModel):
    """Defines the response structure for referral statistics."""
    total_referrals: int
    total_earnings: float
    pending_payouts: float
    referral_code: int

class ReferredUserOut(BaseModel):
    """Defines the response structure for a single referred user."""
    id: int
    email: str
    created_at: datetime

    class Config:
        orm_mode = True

class PayoutOut(BaseModel):
    """Defines the response structure for a single payout record."""
    id: int
    commission_amount: float = Field(alias="amount")
    status: str
    created_at: datetime

    class Config:
        orm_mode = True
        allow_population_by_field_name = True


# --- Core Referral Logic ---

async def assign_referrer(new_user: User, ref_id: int, db: AsyncSession):
    """
    Assigns a referrer to a new user during the signup process.
    This function is called from the /auth/register endpoint.
    """
    if ref_id == new_user.id:
        return  # A user cannot refer themselves.

    # Use await db.get() for a more direct primary key lookup
    referrer = await db.get(User, ref_id)
    if referrer:
        new_user.referred_by_id = referrer.id
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)


# --- API Endpoints ---

@router.get("/stats", response_model=ReferralStatsOut, summary="Get referral statistics for the current user")
async def get_referral_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Provides key statistics for the user's referral activity,
    including total referrals, earnings, and pending amounts.
    """
    # 1. Calculate total referrals
    referrals_count_query = select(func.count(User.id)).where(User.referred_by_id == current_user.id)
    referrals_result = await db.execute(referrals_count_query)
    total_referrals = referrals_result.scalar_one_or_none() or 0

    # 2. Calculate total earnings (sum of all non-failed payouts)
    total_earnings_query = select(func.sum(ReferralPayout.amount)).where(
        ReferralPayout.referrer_id == current_user.id,
        ReferralPayout.status != "failed"
    )
    total_earnings_result = await db.execute(total_earnings_query)
    total_earnings = total_earnings_result.scalar_one_or_none() or 0.0

    # 3. Calculate pending payouts
    pending_payouts_query = select(func.sum(ReferralPayout.amount)).where(
        ReferralPayout.referrer_id == current_user.id,
        ReferralPayout.status == "pending"
    )
    pending_payouts_result = await db.execute(pending_payouts_query)
    pending_payouts = pending_payouts_result.scalar_one_or_none() or 0.0

    return ReferralStatsOut(
        total_referrals=total_referrals,
        total_earnings=total_earnings,
        pending_payouts=pending_payouts,
        referral_code=current_user.id,
    )

@router.get("/my_referrals", response_model=List[ReferredUserOut], summary="List users referred by the current user")
async def get_my_referrals(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Retrieves a list of all users who signed up using the current user's referral code.
    """
    query = select(User).where(User.referred_by_id == current_user.id).order_by(User.created_at.desc())
    result = await db.execute(query)
    return result.scalars().all()


@router.get("/payouts", response_model=List[PayoutOut], summary="List all payouts for the current user")
async def get_my_payouts(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Retrieves the history of all referral payouts for the current user.
    This endpoint is now named /payouts to match frontend expectations.
    """
    query = select(ReferralPayout).where(ReferralPayout.referrer_id == current_user.id).order_by(ReferralPayout.created_at.desc())
    result = await db.execute(query)
    payouts = result.scalars().all()
    return payouts


@router.post("/payout/{payout_id}", summary="Process a specific pending payout via Stripe")
async def process_payout(
    payout_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Sends a pending referral payout to the current user's connected Stripe account.
    """
    query = select(ReferralPayout).where(
        ReferralPayout.id == payout_id,
        ReferralPayout.referrer_id == current_user.id
    )
    result = await db.execute(query)
    payout = result.scalars().first()

    if not payout:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Payout not found")

    if payout.status != "pending":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Payout status is '{payout.status}', not 'pending'.")

    if not current_user.stripe_account_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You must connect a Stripe account in your profile to receive payouts.",
        )

    try:
        # Asynchronously create a transfer to the referrer's connected Stripe account
        await stripe.Transfer.create(
            amount=int(payout.amount * 100),  # Amount in cents
            currency="usd",
            destination=current_user.stripe_account_id,
            description=f"GlitchApe Referral Payout ID: {payout.id}",
        )
        payout.status = "paid"
        await db.commit()
        await db.refresh(payout)
        return {"success": True, "payout_id": payout.id, "status": "paid"}

    except stripe.error.StripeError as e:
        payout.status = "failed"
        await db.commit()
        logger.error(f"Stripe payout failed for payout ID {payout.id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Stripe error: {e.user_message}")
    except Exception as e:
        payout.status = "failed"
        await db.commit()
        logger.exception(f"An unexpected error occurred during payout for ID {payout.id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.")