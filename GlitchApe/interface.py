# interface.py
"""
Core Interface Layer for the GlitchApe Backend.

This module consolidates all components related to:
1.  Database Connection and Session Management (engine, Base, get_db)
2.  Database Models (User, ChatSession, OrderRecord, OrderPayment, etc.)
3.  Security Utilities (password hashing, JWT logic, OAuth2 scheme)
4.  Core Dependencies (get_current_user)

This file is intended to be imported by the web layer (server.py)
and other modules (glitchape_central_handler.py) to access data and auth.
"""

import os
import uuid
import logging
import random
from datetime import datetime, timedelta, timezone  # <-- ADDED timezone
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
    log.critical("DATABASE_URL environment variable not set. Exiting.")
    raise ValueError("DATABASE_URL not set")

JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    log.critical("JWT_SECRET environment variable not set. Exiting.")
    raise ValueError("JWT_SECRET not set")

JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRY_MINUTES = int(os.getenv("JWT_EXPIRY_MINUTES", "60") or 60)

# -----------------------
# Database Setup
# -----------------------
try:
    engine = create_async_engine(DATABASE_URL, echo=False)
    AsyncSessionLocal = sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    Base = declarative_base()
except Exception as e:
    log.critical(f"Failed to initialize database engine: {e}")
    raise

async def get_db():
    """FastAPI dependency to get an async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# -----------------------
# Security Helpers
# -----------------------
pwd_context = CryptContext(schemes=["scrypt", "bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def hash_password(password: str) -> str:
    """Hashes a plain-text password using the default scheme (scrypt)."""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain-text password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_minutes: Optional[int] = None) -> str:
    """Creates a new JWT access token."""
    to_encode = data.copy()
    if expires_minutes:
        expire = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes)
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRY_MINUTES)
    to_encode.update({"exp": expire})
    token = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token

def decode_token(token: str) -> dict:
    """Decodes a JWT token, handling exceptions."""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError as e:
        log.warning(f"Invalid JWT decode attempt: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

# --- ADDED: Verification Code Helper ---
def generate_verification_code() -> str:
    """Generates a random 6-digit verification code."""
    return str(random.randint(100000, 999999))


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

    # ADDED: Relationship to the verification token for cascading delete
    verification_token = relationship(
        "VerificationToken",
        back_populates="user",
        cascade="all, delete-orphan",
        uselist=False  # A user has one verification token at a time
    )
    
    # ADDED: Relationship to reset tokens
    reset_tokens = relationship(
        "ResetToken",
        back_populates="user",
        cascade="all, delete-orphan"
    )


class VerificationToken(Base):
    __tablename__ = "verification_tokens"
    
    # --- MODIFIED: Changed from long token to 6-digit code ---
    
    # Replaced 'token' (as PK) with a standard ID
    id = Column(SA_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Store the 6-digit code
    code = Column(String(6), index=True, nullable=False)
    
    # Added proper ForeignKey with cascade for user wipe
    user_id = Column(
        SA_UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False, 
        unique=True,  # A user can only have one token
        index=True
    )
    
    # Kept expires_at for the 10-minute expiry
    expires_at = Column(DateTime(timezone=True), nullable=False)
    
    # ADDED: Track last_sent_at for the 60-second resend delay
    last_sent_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # ADDED: Relationship back to User
    user = relationship("User", back_populates="verification_token")


class ResetToken(Base):
    __tablename__ = "reset_tokens"
    
    # --- FIX: Added a primary key (id) to resolve the mapper error ---
    id = Column(SA_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # The actual reset token string
    token = Column(String(128), unique=True, index=True, nullable=False)
    
    # Foreign key to the user
    user_id = Column(
        SA_UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False, 
        index=True
    )
    
    expires_at = Column(DateTime(timezone=True), nullable=False)
    
    # Relationship back to the User
    user = relationship("User", back_populates="reset_tokens")


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(SA_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(SA_UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), default="New Chat")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User", back_populates="sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(SA_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(SA_UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Also link to user for easier lookup, though session_id implies user
    user_id = Column(SA_UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    role = Column(String(32), nullable=False) # 'user' or 'ai'
    content = Column(Text, nullable=False)
    image_url = Column(String(1024), nullable=True) # For AI-generated images
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    session = relationship("ChatSession", back_populates="messages")


class OrderRecord(Base):
    __tablename__ = "orders"
    id = Column(SA_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(SA_UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    product_name = Column(String(255), nullable=False)
    
    # Foreign key to the ChatMessage that holds the primary design image
    image_id = Column(SA_UUID(as_uuid=True), ForeignKey("chat_messages.id"), nullable=False)
    
    printful_order_id = Column(String(128), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User", back_populates="orders")
    payments = relationship("OrderPayment", back_populates="order", cascade="all, delete-orphan", uselist=False) # One payment per order


class OrderPayment(Base):
    __tablename__ = "order_payments"
    id = Column(SA_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    order_id = Column(SA_UUID(as_uuid=True), ForeignKey("orders.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    user_id = Column(SA_UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    payment_intent_id = Column(String(255), unique=True, index=True, nullable=False)
    total_cost_cents = Column(Integer, nullable=False)
    currency = Column(String(8), nullable=False, default="usd")
    status = Column(String(64), nullable=False, default="pending_payment", index=True)
    error_message = Column(Text, nullable=True)
    
    variant_id = Column(Integer, nullable=False)
    recipient_json = Column(Text, nullable=False) # Store recipient address as JSON
    printful_file_ids_json = Column(Text, nullable=False) # Store Printful file IDs as JSON
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User", back_populates="payments")
    order = relationship("OrderRecord", back_populates="payments")


# -----------------------
# Core Dependency
# -----------------------

async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)) -> User:
    """
    Decodes JWT token, fetches user from DB.
    This is the core dependency for protected endpoints.
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