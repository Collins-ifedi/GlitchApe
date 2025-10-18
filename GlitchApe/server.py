# server.py
import sys
import os
import uuid
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import Optional, List

import httpx
from fastapi import (
    FastAPI, Request, Form,
    Depends, HTTPException, status, BackgroundTasks
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from passlib.context import CryptContext
from jose import JWTError, jwt

from sqlalchemy import (
    Column, String, Boolean, DateTime, Text, select, func, Integer, ForeignKey
)
from sqlalchemy.dialects.postgresql import UUID as SA_UUID
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Logging Configuration ---
# Set up basic logging for production
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# --- Integration of AI and Order Modules ---

# Import ai router (ai.py)
try:
    from ai import router as ai_router
except ImportError:
    log.error("CRITICAL: ai.py not found or ai_router not exposed.")
    ai_router = None

# Import order router and models (order.py)
try:
    from order import router as order_router
    # This import is CRUCIAL so that Base.metadata knows about the OrderPayment table
    from order import OrderPayment
except ImportError:
    log.error("CRITICAL: order.py not found or order_router/OrderPayment not exposed.")
    order_router = None
    OrderPayment = None # This will cause startup to fail, which is intended.

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

BREVO_API_KEY = os.getenv("BREVO_API_KEY")
BREVO_SENDER_EMAIL = os.getenv("SENDER_EMAIL")
if not BREVO_API_KEY or not BREVO_SENDER_EMAIL:
    log.warning("BREVO_API_KEY or SENDER_EMAIL not set. Email services will be disabled.")

# Keys for other modules, checked here for completeness
if not os.getenv("AI_API_KEY"):
    raise RuntimeError("AI_API_KEY env var required for ai.py")
if not os.getenv("PRINTFUL_API_KEY"):
    raise RuntimeError("PRINTFUL_API_KEY env var required for order.py")
if not os.getenv("STRIPE_SECRET_KEY"):
    raise RuntimeError("STRIPE_SECRET_KEY env var required for order.py")
if not os.getenv("STRIPE_WEBHOOK_SECRET"):
    raise RuntimeError("STRIPE_WEBHOOK_SECRET env var required for order.py")

FRONTEND_URL = os.getenv("FRONTEND_URL")
if not FRONTEND_URL:
    raise RuntimeError("FRONTEND_URL env var required for CORS and email links")

# -----------------------
# App + Middleware
# -----------------------
app = FastAPI(title="GlitchApe Backend (Unified)")

# Production-ready CORS: Only allow the frontend URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL], # Restrict to frontend
    allow_credentials=True,
    allow_methods=["GET", "POST"], # Be explicit
    allow_headers=["*"],
)

# -----------------------
# Database (async)
# -----------------------
Base = declarative_base()
engine = create_async_engine(DATABASE_URL, future=True, echo=False) # echo=False for production
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

# -----------------------
# Security helpers
# -----------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login") # Path relative to frontend

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_minutes: Optional[int] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=(expires_minutes or JWT_EXPIRY_MINUTES))
    to_encode.update({"exp": expire})
    token = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError as e:
        log.warning(f"Invalid JWT decode attempt: {e}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


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
    # Removed 'images' relationship and 'ImageRecord' model


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

# Note: The 'OrderPayment' model is defined in order.py,
# and will be registered with Base.metadata when 'order.py' is imported.


# -----------------------
# Startup: create tables & mount routers
# -----------------------
@app.on_event("startup")
async def startup():
    if not OrderPayment:
        log.critical("OrderPayment model not imported from order.py. Database cannot be initialized.")
        raise RuntimeError("Failed to import OrderPayment model.")

    async with engine.begin() as conn:
        log.info("Initializing database...")
        await conn.run_sync(Base.metadata.create_all)
        log.info("Database tables created/verified.")
    
    # Attach AI router
    if ai_router:
        app.include_router(ai_router, prefix="/api")
        log.info("AI router (ai.py) loaded successfully at /api/ai.")
    else:
        log.error("AI router (ai.py) not found. AI endpoints will be unavailable.")
        
    # Attach Order router
    if order_router:
        app.include_router(order_router, prefix="/api")
        log.info("Order router (order.py) loaded successfully at /api/orders.")
    else:
        log.error("Order router (order.py) not found. Order endpoints will be unavailable.")


# -----------------------
# Email templates + Brevo helper
# -----------------------
EMAIL_TEMPLATES = {
    "verify": {
        "subject": "Verify Your GlitchApe Account",
        "html": """
        <div style="font-family:Arial;padding:24px;background:#0b1220;color:#e6eef6;border-radius:12px;">
          <h1 style="color:#00F5FF">GlitchApe</h1>
          <p>Welcome! Click to verify your account.</p>
          <div style="text-align:center;margin:24px;">
            <a href="{link}" style="padding:12px 20px;border-radius:8px;background:linear-gradient(90deg,#FF006E,#00F5FF);color:#0b1220;text-decoration:none;font-weight:700;">Verify Account</a>
          </div>
        </div>
        """
    },
    "resend_verify": {"subject": "Verify Your GlitchApe Account", "html": "<p>Verify your account: <a href='{link}'>click here</a></p>"},
    "forgot_password": {"subject": "Reset your GlitchApe password", "html": "<p>Reset password: <a href='{link}'>Reset</a></p>"},
    "success_reset": {"subject": "Your password was changed", "html": "<p>Your password was updated.</p>"},
}


async def _send_brevo_email(to_email: str, template_key: str, link: Optional[str] = None):
    if not BREVO_API_KEY:
        log.warning(f"Email sending skipped for {to_email}: BREVO_API_KEY not set.")
        return 
    
    tpl = EMAIL_TEMPLATES.get(template_key)
    if not tpl:
        log.error(f"Unknown email template key: {template_key}")
        return
        
    html = tpl["html"].format(link=link or "#")
    payload = {
        "sender": {"name": "GlitchApe", "email": BREVO_SENDER_EMAIL},
        "to": [{"email": to_email}],
        "subject": tpl["subject"],
        "htmlContent": html,
    }
    headers = {"api-key": BREVO_API_KEY, "Accept": "application/json", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post("https://api.brevo.com/v3/smtp/email", json=payload, headers=headers)
            if r.status_code >= 400:
                log.error(f"Brevo API error sending to {to_email}: {r.status_code} {r.text}")
            else:
                log.info(f"Email '{template_key}' sent successfully to {to_email}.")
    except Exception as e:
        log.error(f"Failed to send Brevo email to {to_email}: {e}")


# -----------------------
# Dependency: current user
# -----------------------
async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
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


# -----------------------
# Auth endpoints
# -----------------------
@app.post("/api/auth/register", status_code=201)
async def register_user(
    email: str = Form(...), 
    password: str = Form(...), 
    country_code: Optional[str] = Form(None), 
    db: AsyncSession = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    q = select(User).where(User.email == email)
    r = await db.execute(q)
    if r.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = hash_password(password)
    new_user = User(email=email, hashed_password=hashed_password, is_verified=False, country_code=country_code)
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    log.info(f"New user registered: {email} (ID: {new_user.id})")
    
    # create verification token
    token = uuid.uuid4().hex
    vt = VerificationToken(token=token, user_id=new_user.id, expires_at=datetime.utcnow() + timedelta(hours=24))
    db.add(vt)
    await db.commit()
    
    link = f"{FRONTEND_URL}/verify?token={token}"
    background_tasks.add_task(_send_brevo_email, new_user.email, "verify", link)
    
    return {"message": "Registered. Verification email sent."}


@app.get("/api/auth/verify")
async def verify_email(token: str, db: AsyncSession = Depends(get_db)):
    q_vt = select(VerificationToken).where(VerificationToken.token == token)
    vt = (await db.execute(q_vt)).scalar_one_or_none()

    if not vt or vt.expires_at < datetime.utcnow():
        log.warning(f"Invalid/expired verification token used: {token}")
        raise HTTPException(status_code=400, detail="Invalid or expired verification token")
    
    user = await db.get(User, vt.user_id)
    if not user:
        log.error(f"User not found for verification token {token} (User ID: {vt.user_id})")
        raise HTTPException(status_code=404, detail="User not found")
    
    user.is_verified = True
    await db.delete(vt)
    await db.commit()
    log.info(f"User email verified: {user.email}")
    
    html = f"<html><body style='text-align:center;padding:40px;'><h2>Verified ✅</h2><p>Your account is verified. You can now log in.</p><a href='{FRONTEND_URL}'>Open App</a></body></html>"
    return HTMLResponse(content=html)


@app.post("/api/auth/resend")
async def resend_verification(
    email: str = Form(...), 
    db: AsyncSession = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    q = select(User).where(User.email == email)
    r = await db.execute(q)
    user = r.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user.is_verified:
        raise HTTPException(status_code=400, detail="Email is already verified")

    token = uuid.uuid4().hex
    vt = VerificationToken(token=token, user_id=user.id, expires_at=datetime.utcnow() + timedelta(hours=24))
    db.add(vt)
    await db.commit()
    
    link = f"{FRONTEND_URL}/verify?token={token}"
    background_tasks.add_task(_send_brevo_email, user.email, "resend_verify", link)
    
    return {"message": "Verification email resent"}


@app.post("/api/auth/forgot-password")
async def forgot_password(
    email: str = Form(...), 
    db: AsyncSession = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    q = select(User).where(User.email == email)
    r = await db.execute(q)
    user = r.scalar_one_or_none()
    if not user:
        # Don't reveal if user exists
        log.info(f"Password reset requested for non-existent email: {email}")
        return {"message": "If an account with this email exists, a reset email has been sent."}
    
    token = uuid.uuid4().hex
    rt = ResetToken(token=token, user_id=user.id, expires_at=datetime.utcnow() + timedelta(hours=1))
    db.add(rt)
    await db.commit()
    
    link = f"{FRONTEND_URL}/reset-password?token={token}"
    background_tasks.add_task(_send_brevo_email, user.email, "forgot_password", link)
    
    return {"message": "If an account with this email exists, a reset email has been sent."}


@app.post("/api/auth/reset-password")
async def reset_password(
    token: str = Form(...), 
    new_password: str = Form(...), 
    db: AsyncSession = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    q_rt = select(ResetToken).where(ResetToken.token == token)
    rt = (await db.execute(q_rt)).scalar_one_or_none()

    if not rt or rt.expires_at < datetime.utcnow():
        log.warning(f"Invalid/expired password reset token used: {token}")
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")
    
    user = await db.get(User, rt.user_id)
    if not user:
         log.error(f"User not found for reset token {token} (User ID: {rt.user_id})")
         raise HTTPException(status_code=404, detail="User not found")

    user.hashed_password = hash_password(new_password)
    await db.delete(rt)
    await db.commit()
    log.info(f"Password reset successfully for user: {user.email}")
    
    background_tasks.add_task(_send_brevo_email, user.email, "success_reset", None)
    
    return {"message": "Password reset successful"}


@app.post("/api/auth/login")
async def login(form: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    q = select(User).where(User.email == form.username)
    r = await db.execute(q)
    user = r.scalar_one_or_none()
    
    if not user or not verify_password(form.password, user.hashed_password):
        log.warning(f"Failed login attempt for email: {form.username}")
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    if not user.is_verified:
        log.info(f"Login attempt by unverified user: {form.username}")
        raise HTTPException(status_code=403, detail="Please verify your email")
        
    access_token = create_access_token({"email": user.email, "sub": str(user.id)})
    log.info(f"User logged in: {user.email}")
    return {"access_token": access_token, "token_type": "bearer", "user": {"id": str(user.id), "email": user.email}}


@app.post("/api/auth/delete")
async def delete_account(current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    user_email = current_user.email
    # The get_current_user dependency already finds the user.
    # cascade="all, delete-orphan" in relationships handles deletion
    # of all associated sessions, messages, orders, etc.
    await db.delete(current_user)
    await db.commit()
    log.info(f"User account deleted: {user_email}")
    return {"message": "Account deleted"}


# -----------------------
# Chat session endpoints (Core)
# -----------------------
@app.post("/api/chat/start")
async def start_chat(name: Optional[str] = Form(None), current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    cs = ChatSession(user_id=current_user.id, name=name)
    db.add(cs)
    await db.commit()
    await db.refresh(cs)
    log.info(f"New chat session created: {cs.id} for user {current_user.email}")
    return {"session_id": str(cs.id), "created_at": cs.created_at.isoformat()}


@app.get("/api/chat/history/{session_id}")
async def get_history(session_id: str, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    try:
        session_uuid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session ID format")

    # This query SECURES the endpoint by checking user_id.
    # This prevents one user from seeing another user's chat.
    q = select(ChatSession).where(
        ChatSession.id == session_uuid, 
        ChatSession.user_id == current_user.id
    )
    r = await db.execute(q)
    s = r.scalar_one_or_none()
    
    if not s:
        log.warning(f"User {current_user.email} tried to access unknown session: {session_id}")
        raise HTTPException(status_code=404, detail="Session not found")
        
    m_q = select(ChatMessage).where(ChatMessage.session_id == s.id).order_by(ChatMessage.created_at)
    m_r = await db.execute(m_q)
    rows = m_r.scalars().all()
    
    out = []
    for m in rows:
        out.append({
            "id": str(m.id),
            "role": m.role,
            "content": m.content,
            "image_url": m.image_url,
            "created_at": m.created_at.isoformat() if m.created_at else None
        })
    return {"session_id": session_id, "messages": out}


# -----------------------
# Health & root
# -----------------------
@app.get("/api/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.get("/")
async def root():
    return {"message": "GlitchApe Backend", "frontend": FRONTEND_URL}
