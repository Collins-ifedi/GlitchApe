# models.py
"""
Database models for GlitchApe.

This file defines all SQLAlchemy models used by the application,
providing a single source of truth for the database schema.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, Boolean, DateTime, Text, func, ForeignKey
)
from sqlalchemy.dialects.postgresql import UUID as SA_UUID
from sqlalchemy.orm import declarative_base, relationship

# --- Base ---
Base = declarative_base()

# -----------------------
# Models
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
    image_url = Column(String(1024), nullable=True) # Used by ai.py and order.py
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

# Note: The 'OrderPayment' model is defined in order.py
# and will be registered with Base.metadata when 'order.py' is imported
# by server.py.