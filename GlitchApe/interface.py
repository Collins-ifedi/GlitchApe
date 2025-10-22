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
import random  # <-- ADDED for code generation
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
# ... (rest of configuration is unchanged) ...
JWT_EXPIRY_MINUTES = int(os.getenv("JWT_EXPIRY_MINUTES", "60") or 60)

# -----------------------
# Database Setup
# -----------------------
Base = declarative_base()
# ... (rest of database setup is unchanged) ...
async def get_db():
    """FastAPI dependency to get an async database session."""
    async with AsyncSessionLocal() as session:
        yield session

# -----------------------
# Security Helpers
# -----------------------
# ... (pwd_context and oauth2_scheme are unchanged) ...

def hash_password(password: str) -> str:
    """Hashes a plain-text password using the default scheme (scrypt)."""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain-text password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_minutes: Optional[int] = None) -> str:
    # ... (function is unchanged) ...
    return token

def decode_token(token: str) -> dict:
    # ... (function is unchanged) ...
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
    # ... (model is unchanged) ...


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    # ... (model is unchanged) ...


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    # ... (model is unchanged) ...


class OrderRecord(Base):
    __tablename__ = "orders"
    # ... (model is unchanged) ...


class OrderPayment(Base):
    __tablename__ = "order_payments"
    # ... (model is unchanged) ...


# -----------------------
# Core Dependency
# -----------------------

async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)) -> User:
    # ... (function is unchanged) ...
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