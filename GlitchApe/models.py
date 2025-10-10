# models.py
import uuid
import enum
from datetime import datetime

from sqlalchemy import (
    Column, String, ForeignKey, DateTime, Text, JSON,
    Enum as SAEnum, Numeric, Boolean, Integer
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, backref
from db import Base

# --- Enums for Status Fields ---
# Using Enums makes status fields type-safe and less prone to errors.

class OrderStatus(str, enum.Enum):
    PENDING = "pending"
    PAID = "paid"
    SHIPPED = "shipped"
    CANCELLED = "cancelled"

class PayoutStatus(str, enum.Enum):
    PENDING = "pending"
    PAID = "paid"
    FAILED = "failed"


# ===================================================================
# User, Auth & Address Models
# ===================================================================
class User(Base):
    __tablename__ = "users"

    # Using UUID for primary keys is a best practice for security and scalability.
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    stripe_account_id = Column(String(255), nullable=True)

    # --- Relationships ---
    referred_by_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True)
    referrals = relationship("User", backref=backref("referrer", remote_side=[id]))

    addresses = relationship("Address", back_populates="user", cascade="all, delete-orphan")
    designs = relationship("Design", back_populates="user", cascade="all, delete-orphan")
    orders = relationship("Order", back_populates="user", cascade="all, delete-orphan")
    
    payouts_received = relationship(
        "ReferralPayout", 
        foreign_keys="[ReferralPayout.referrer_id]", 
        back_populates="referrer", 
        cascade="all, delete-orphan"
    )
    payouts_generated = relationship(
        "ReferralPayout",
        foreign_keys="[ReferralPayout.referred_user_id]",
        back_populates="referred_user"
    )

class Address(Base):
    """A dedicated model for storing user addresses."""
    __tablename__ = "addresses"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    name = Column(String(255), nullable=False)
    address_line_1 = Column(String(255), nullable=False)
    address_line_2 = Column(String(255), nullable=True)
    city = Column(String(100), nullable=False)
    state = Column(String(100), nullable=False)
    postal_code = Column(String(20), nullable=False)
    country_code = Column(String(2), nullable=False) # ISO 3166-1 alpha-2

    user = relationship("User", back_populates="addresses")


# ===================================================================
# Design & Product Models
# ===================================================================
class Design(Base):
    __tablename__ = "designs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    product_id = Column(Integer, nullable=False) # From Printful
    
    design_url = Column(String, nullable=False)
    prompt = Column(Text, nullable=True)
    mockup_urls = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="designs")


# ===================================================================
# E-commerce Models
# ===================================================================
class Order(Base):
    __tablename__ = "orders"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Explicitly link to a shipping address for record-keeping.
    shipping_address_id = Column(UUID(as_uuid=True), ForeignKey("addresses.id", ondelete="SET NULL"), nullable=True)

    stripe_session_id = Column(String, nullable=True, unique=True, index=True)
    stripe_payment_intent = Column(String, nullable=True, unique=True, index=True)
    
    # Use Numeric/Decimal for financial values to avoid floating-point errors.
    amount = Column(Numeric(10, 2), nullable=False)
    currency = Column(String(10), default="usd", nullable=False)
    
    printful_order_id = Column(String, nullable=True)
    status = Column(SAEnum(OrderStatus), default=OrderStatus.PENDING, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    paid_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="orders")
    shipping_address = relationship("Address")
    items = relationship("OrderItem", back_populates="order", cascade="all, delete-orphan")

class OrderItem(Base):
    __tablename__ = "order_items"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    order_id = Column(UUID(as_uuid=True), ForeignKey("orders.id", ondelete="CASCADE"), nullable=False, index=True)
    design_id = Column(UUID(as_uuid=True), ForeignKey("designs.id", ondelete="SET NULL"), nullable=True, index=True)
    
    quantity = Column(Integer, nullable=False)
    price_per_item = Column(Numeric(10, 2), nullable=False)

    order = relationship("Order", back_populates="items")
    design = relationship("Design")


# ===================================================================
# Referral & Payout Models
# ===================================================================
class ReferralPayout(Base):
    __tablename__ = "referral_payouts"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    referrer_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    referred_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    order_id = Column(UUID(as_uuid=True), ForeignKey("orders.id", ondelete="SET NULL"), nullable=True, index=True)

    amount = Column(Numeric(10, 2), nullable=False)
    status = Column(SAEnum(PayoutStatus), default=PayoutStatus.PENDING, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    referrer = relationship("User", back_populates="payouts_received", foreign_keys=[referrer_id])
    referred_user = relationship("User", back_populates="payouts_generated", foreign_keys=[referred_user_id])
    order = relationship("Order")