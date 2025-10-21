# interface.py
"""
Core Interface Layer for the GlitchApe Backend.

This module consolidates all components related to:
1.  Database Connection and Session Management (engine, Base, get_db)
2.  Database Models (User, ChatSession, OrderRecord, OrderPayment, etc.)
3.  Security Utilities (password hashing, JWT logic, OAuth2 scheme)
4.  Core Dependencies (get_current_user)

This file is intended to be imported by the web layer (server.py)
and other modules (ai.py, order.py) to access data and auth.
"""

import os
import uuid
import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from jose import JWTError, jwt

from sqlalchemy import (
    Column, String, Boolean, DateTime, Text, select, func, Integer, ForeignKey
)
from sqlalchemy.dialects.postgresql import UUID as SA_UUID
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# --- Logging ---
log = logging.getLogger(__name__)

# -----------------------
# Configuration (env)
# -----------------------
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL env var required")

JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET env var required")

JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRY_MINUTES = int(os.getenv("JWT_EXPIRY_MINUTES", "60") or 60)

# -----------------------
# Database Setup
# -----------------------
Base = declarative_base()
engine = create_async_engine(DATABASE_URL, future=True, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db():
    """FastAPI dependency to get an async database session."""
    async with AsyncSessionLocal() as session:
        yield session

# -----------------------
# Security Helpers
# -----------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

def hash_password(password: str) -> str:
    """Hashes a plain-text password using bcrypt."""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain-text password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_minutes: Optional[int] = None) -> str:
    """Creates a new JWT access token."""
    to_encode = data.copy()
    expire_minutes = expires_minutes or JWT_EXPIRY_MINUTES
    expire = datetime.utcnow() + timedelta(minutes=expire_minutes)
    to_encode.update({"exp": expire})
    token = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token

def decode_token(token: str) -> dict:
    """
    Decodes a JWT token.
    Raises HTTPException 401 if the token is invalid.
    """
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError as e:
        log.warning(f"Invalid JWT decode attempt: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

# -----------------------
# Database Models
# -----------------------

class User(Base):
    __tablename__ = "users"
    id = Column(SA_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(320), unique=True, nullable=False, index=True)
    hashed_password = Column(String(512), nullable=False)
    is_verified = Column(Boolean, default=False)
    country_code = Column(String(16), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")
    orders = relationship("OrderRecord", back_populates="user", cascade="all, delete-orphan")
    payments = relationship("OrderPayment", back_populates="user", cascade="all, delete-orphan")


class VerificationToken(Base):
    __tablename__ = "verification_tokens"
    token = Column(String(128), primary_key=True, index=True)
    user_id = Column(SA_UUID(as_uuid=True), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)


class ResetToken(Base):
    __tablename__ = "reset_tokens"
    token = Column(String(128), primary_key=True, index=True)
    user_id = Column(SA_UUID(as_uuid=True), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(SA_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(SA_UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")
    user = relationship("User", back_populates="sessions")


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(SA_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(SA_UUID(as_uuid=True), ForeignKey("chat_sessions.id"), nullable=False, index=True)
    user_id = Column(SA_UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    role = Column(String(16), nullable=False)  # 'user' or 'ai'
    content = Column(Text, nullable=True)
    image_url = Column(String(1024), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    session = relationship("ChatSession", back_populates="messages")


class OrderRecord(Base):
    __tablename__ = "orders"
    id = Column(SA_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(SA_UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    printful_order_id = Column(String(255), nullable=True)
    image_id = Column(SA_UUID(as_uuid=True), nullable=True) # Refers to a ChatMessage ID
    product_name = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User", back_populates="orders")
    payment = relationship("OrderPayment", back_populates="order", uselist=False, cascade="all, delete-orphan")


class OrderPayment(Base):
    """
    Augments the base OrderRecord with payment, state, and fulfillment data.
    Moved from order.py to centralize all data models.
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

    # Relationships
    order = relationship("OrderRecord", back_populates="payment")
    user = relationship("User", back_populates="payments")


# -----------------------
# Core Dependency
# -----------------------

async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)) -> User:
    """
    FastAPI dependency to get the current user from a JWT token.
    Raises HTTPException 401 if the user is not found or token is invalid.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = decode_token(token)
        user_email = payload.get("email") or payload.get("sub")
        if user_email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    q = select(User).where(User.email == user_email)
    r = await db.execute(q)
    user = r.scalar_one_or_none()
    
    if not user:
        raise credentials_exception
        
    return user