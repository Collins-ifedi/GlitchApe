# server.py
import os
import uuid
import logging
from datetime import datetime, timedelta, timezone  # <-- ADDED timezone
from typing import Optional, List

import httpx
from fastapi import (
    FastAPI, Request, Form,
    Depends, HTTPException, status, BackgroundTasks
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse 
from fastapi.security import OAuth2PasswordRequestForm

from sqlalchemy import select, delete  # <-- ADDED delete
from sqlalchemy.ext.asyncio import AsyncSession

# --- Logging Configuration ---\n
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# --- Integration of Central Brain ---

# Import the new unified router from the central handler
try:
    from glitchape_central_handler import router as central_router
except ImportError:
    log.error("CRITICAL: glitchape_central_handler.py not found or router not exposed.")
    central_router = None

# --- Import Core Components from Interface ---
try:
    from interface import (
        Base,           # For startup
        engine,         # For startup
        get_db,         # Dependency
        get_current_user, # Dependency
        hash_password,  # Auth util
        verify_password, # Auth util
        create_access_token, # Auth util
        generate_verification_code, # <-- ADDED
        User,           # Model
        VerificationToken, # Model
        ResetToken,     # Model
        ChatSession,    # Model
        ChatMessage,    # Model
        OrderRecord,    # Model
        OrderPayment,   # Model
        JWT_EXPIRY_MINUTES # Config
    )
except ImportError as e:
    log.critical(f"Failed to import from interface.py: {e}")
    # Exit or raise if essential components are missing
    raise

# --- Stripe Webhook Secret ---
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
if not STRIPE_WEBHOOK_SECRET:
    log.warning("STRIPE_WEBHOOK_SECRET env var not set. Webhooks will fail.")
    # In production, you might want to raise an error
    # raise RuntimeError("STRIPE_WEBHOOK_SECRET env var required")

# --- App Initialization ---
app = FastAPI(
    title="GlitchApe Backend API",
    description="Main API server for GlitchApe, handling auth, AI, and orders.",
    version="1.0.0"
)

# --- CORS Middleware ---
origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Database Startup Event ---
@app.on_event("startup")
async def on_startup():
    """Create database tables on startup."""
    async with engine.begin() as conn:
        # await conn.run_sync(Base.metadata.drop_all) # Uncomment to wipe DB
        await conn.run_sync(Base.metadata.create_all)
    log.info("Database tables verified/created.")
    
    # --- ADDED: Scheduler setup for cleanup (optional) ---
    # To run the cleanup task, you need a scheduler.
    # e.g., using APScheduler:
    # from apscheduler.schedulers.asyncio import AsyncIOScheduler
    # scheduler = AsyncIOScheduler()
    # scheduler.add_job(
    #     run_cleanup, # A wrapper to get a DB session
    #     "interval", 
    #     hours=1
    # )
    # scheduler.start()
    # log.info("Started periodic cleanup task.")


# --- NEW: Helper Functions (Email & Cleanup) ---

def send_verification_email(email: str, code: str):
    """
    (Placeholder) Sends the verification code email.
    TODO: Replace this dummy function with your actual email service call.
    """
    log.info(f"EMAIL TASK: Sending code {code} to {email}")
    # In a real app, this would use a service like SendGrid, Mailgun, etc.
    # Simulating a send.
    print(f"--- DUMMY EMAIL --- \nTO: {email}\nCODE: {code}\n-------------------")
    return True

async def cleanup_unverified_users(db: AsyncSession):
    """
    Deletes users who are not verified and were created more than 2 hours ago.
    This should be run periodically by a scheduler.
    """
    CLEANUP_THRESHOLD = timedelta(hours=2) 
    cutoff_time = datetime.now(timezone.utc) - CLEANUP_THRESHOLD
    
    # The 'ondelete="CASCADE"' in interface.py ensures tokens are deleted too.
    q_users_to_delete = delete(User).where(
        User.is_verified == False,
        User.created_at < cutoff_time
    ).returning(User.id) # Use returning to log how many were deleted

    try:
        result = await db.execute(q_users_to_delete)
        deleted_user_ids = result.scalars().all()
        
        if deleted_user_ids:
            log.info(f"Cleanup task: Wiped {len(deleted_user_ids)} stale unverified users.")
            await db.commit()
        else:
            log.info("Cleanup task: No stale unverified users found.")
    except Exception as e:
        log.error(f"Error during cleanup_unverified_users: {e}")
        await db.rollback()

# Wrapper to run cleanup with its own session (if using a scheduler)
async def run_cleanup():
    """Wrapper to create a session for the scheduled cleanup job."""
    async with AsyncSessionLocal() as session:
        await cleanup_unverified_users(session)


# --- Include Central API Router ---
if central_router:
    app.include_router(central_router, prefix="/api", tags=["Central AI & Orders"])
else:
    log.error("Central router not included. AI/Order endpoints will be missing.")


# =======================================
# AUTHENTICATION ENDPOINTS
# =======================================

@app.post("/auth/register", status_code=status.HTTP_201_CREATED)
async def register(
    email: str = Form(...),
    password: str = Form(...),
    country_code: str = Form(None), # From interface.py User model
    db: AsyncSession = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Handles new user registration.
    - Hashes password
    - Creates a new user (unverified)
    - **Immediate Wipe**: Deletes existing *unverified* users with the same email.
    - Generates a 6-digit verification code (10-min expiry)
    - Sends verification code via email.
    """
    user_q = select(User).where(User.email == email)
    existing_user = (await db.execute(user_q)).scalar_one_or_none()

    if existing_user:
        if not existing_user.is_verified:
            # IMMEDIATE WIPE: User exists but isn't verified.
            # Delete them to allow a fresh registration attempt.
            log.info(f"Deleting stale unverified user for new registration: {email}")
            await db.delete(existing_user)
            await db.commit()
            # Continue to create the new user
        else:
            # User exists and is verified, this is a real conflict.
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="User with this email already exists."
            )

    # 1. Create New User
    hashed_password = hash_password(password)
    new_user = User(
        email=email,
        hashed_password=hashed_password,
        is_verified=False,
        country_code=country_code
        # created_at is set by server_default
    )
    db.add(new_user)
    await db.flush() # Flush to get the new_user.id

    # 2. Generate and Store Verification Code
    verification_code = generate_verification_code()
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(minutes=10) # 10 minute expiry
    
    new_token = VerificationToken(
        code=verification_code,
        user_id=new_user.id,
        expires_at=expires_at,
        last_sent_at=now  # Set send time
    )
    db.add(new_token)
    await db.commit()

    # 3. Send Email in Background
    background_tasks.add_task(send_verification_email, email, verification_code)
    
    return {
        "message": "Registration successful. Please check your email for a 6-digit verification code.", 
        "email": email
    }


@app.post("/auth/login")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """
    Standard OAuth2 password flow login.
    Returns a JWT access token.
    """
    q = select(User).where(User.email == form_data.username)
    r = await db.execute(q)
    user = r.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )

    if not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
        
    if not user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account not verified. Please check your email."
        )

    access_token = create_access_token(
        data={"email": user.email, "id": str(user.id)},
        expires_minutes=JWT_EXPIRY_MINUTES
    )

    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/auth/verify-code", status_code=status.HTTP_200_OK)
async def verify_code(
    email: str = Form(...),
    code: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Verifies a user's account using the 6-digit code.
    """
    # Find user by email
    user_q = select(User).where(User.email == email)
    user = (await db.execute(user_q)).scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    
    if user.is_verified:
        return {"message": "Account already verified."}

    # Find the matching token
    token_q = select(VerificationToken).where(
        VerificationToken.user_id == user.id,
        VerificationToken.code == code
    )
    token = (await db.execute(token_q)).scalar_one_or_none()

    if not token:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid verification code.")

    # Check Expiration (10 mins)
    if token.expires_at < datetime.now(timezone.utc):
        # Token is expired. Delete it.
        await db.delete(token)
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_410_GONE, # 410 Gone is semantically correct
            detail="Verification code has expired. Please request a new one."
        )

    # --- Success ---
    # Mark user as verified
    user.is_verified = True
    
    # Delete the token, it's been used
    await db.delete(token)
    
    await db.commit()

    return {"message": "Account successfully verified."}


@app.post("/auth/resend-code", status_code=status.HTTP_200_OK)
async def resend_code(
    email: str = Form(...),
    db: AsyncSession = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Resends a new verification code.
    Implements a 60-second delay between resend requests.
    """
    user_q = select(User).where(User.email == email)
    user = (await db.execute(user_q)).scalar_one_or_none()

    # For security, don't reveal if user exists or is already verified
    if not user or user.is_verified:
        log.warning(f"Resend code request for non-existent or verified user: {email}")
        return {"message": "If an unverified account exists for this email, a new code has been sent."}

    # Find the user's current token
    token_q = select(VerificationToken).where(VerificationToken.user_id == user.id)
    token = (await db.execute(token_q)).scalar_one_or_none()
    
    now = datetime.now(timezone.utc)
    RESEND_DELAY_SECONDS = 60
    TOKEN_EXPIRY_MINUTES = 10

    if token:
        # Check Resend Delay (60 seconds)
        can_resend_at = token.last_sent_at + timedelta(seconds=RESEND_DELAY_SECONDS)

        if now < can_resend_at:
            wait_seconds = int((can_resend_at - now).total_seconds()) + 1
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Please wait {wait_seconds} seconds before requesting a new code."
            )

    # If we are here, we can send a new code.
    new_code = generate_verification_code()
    new_expires_at = now + timedelta(minutes=TOKEN_EXPIRY_MINUTES)
    
    if token:
        # Update existing token
        token.code = new_code
        token.expires_at = new_expires_at
        token.last_sent_at = now
    else:
        # Create a new token if one didn't exist (e.g., it expired and was deleted)
        token = VerificationToken(
            user_id=user.id,
            code=new_code,
            expires_at=new_expires_at,
            last_sent_at=now
        )
        db.add(token)

    await db.commit()

    # Send new code in background
    background_tasks.add_task(send_verification_email, email, new_code)

    return {"message": "A new verification code has been sent to your email."}


# --- PASSWORD RESET (UNCHANGED) ---

@app.post("/auth/forgot-password")
async def forgot_password(
    email: str = Form(...),
    db: AsyncSession = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Handles a forgot password request.
    Generates a reset token and sends a reset email.
    """
    q = select(User).where(User.email == email)
    r = await db.execute(q)
    user = r.scalar_one_or_none()
    
    # Security: Don't reveal if user exists
    if not user:
        log.warning(f"Forgot password attempt for non-existent user: {email}")
        return {"message": "If an account with this email exists, a password reset link has been sent."}

    # Clear old tokens
    await db.execute(delete(ResetToken).where(ResetToken.user_id == user.id))

    # Create new token
    token = str(uuid.uuid4())
    expires = datetime.utcnow() + timedelta(hours=1)
    reset_token = ResetToken(
        token=token,
        user_id=user.id,
        expires_at=expires
    )
    db.add(reset_token)
    await db.commit()

    # Send email
    reset_link = f"https://your-frontend-url/reset-password?token={token}"
    # background_tasks.add_task(send_password_reset_email, email, reset_link)
    log.info(f"PASSWORD RESET LINK for {email}: {reset_link}") # Placeholder

    return {"message": "If an account with this email exists, a password reset link has been sent."}


@app.post("/auth/reset-password")
async def reset_password(
    token: str = Form(...),
    new_password: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Resets a user's password using a valid token.
    """
    q = select(ResetToken).where(ResetToken.token == token)
    r = await db.execute(q)
    token_data = r.scalar_one_or_none()
    
    if not token_data:
        raise HTTPException(status_code=400, detail="Invalid token")
        
    if token_data.expires_at < datetime.utcnow():
        raise HTTPException(status_code=400, detail="Token expired")
        
    q_user = select(User).where(User.id == token_data.user_id)
    r_user = await db.execute(q_user)
    user = r_user.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    user.hashed_password = hash_password(new_password)
    
    # Delete the used token
    await db.delete(token_data)
    
    await db.commit()
    
    return {"message": "Password reset successful"}


# =======================================
# USER ENDPOINTS (UNCHANGED)
# =======================================

@app.get("/users/me", response_model=None) # Define a Pydantic model for User
async def read_users_me(current_user: User = Depends(get_current_user)):
    """Fetches the current authenticated user's details."""
    # Note: Returning the DB model directly is bad practice.
    # Create a Pydantic schema for the user response.
    return {
        "id": current_user.id,
        "email": current_user.email,
        "is_verified": current_user.is_verified,
        "created_at": current_user.created_at,
        "country_code": current_user.country_code
    }


# =======================================
# STRIPE WEBHOOK (UNCHANGED)
# =======================================

@app.post("/stripe-webhook")
async def stripe_webhook(
    request: Request, 
    db: AsyncSession = Depends(get_db)
):
    """
    Handles incoming webhooks from Stripe.
    This is a placeholder and should be implemented in central_handler.
    """
    if not central_router:
        raise HTTPException(status_code=501, detail="Webhook handler not implemented")
        
    # This endpoint should ideally be handled by the central_router
    # to keep logic consolidated.
    log.warning("Received webhook at /stripe-webhook. Forwarding to central handler logic if available.")
    
    # Dummy implementation if central_handler doesn't expose it
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')

    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="Stripe webhook secret not configured")

    try:
        event = httpx.post( # This is a placeholder
            "http://localhost:8000/api/orders/webhook", 
            content=payload, 
            headers={"stripe-signature": sig_header}
        )
        event.raise_for_status()
        return event.json()
    except Exception as e:
        log.error(f"Webhook processing error: {e}")
        raise HTTPException(status_code=400, detail="Webhook processing error")
    

# =======================================
# FRONTEND SERVING (MODIFIED)
# =======================================

@app.get("/")
async def serve_frontend_root():
    """Serves the main index.html file."""
    try:
        with open("index.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        # Fallback if index.html is missing
        return {"message": "GlitchApe Backend", "status": "Frontend file not found"}


@app.get("/{path:path}")
async def serve_frontend_spa(path: str):
    """
    Catch-all route to serve the index.html for known Single Page Application (SPA) paths.
    This is CRITICAL for paths like /reset-password?token=... when clicked from an email.
    """
    # List of known frontend routes that should load the index.html
    # REMOVED "verify" as it's no longer a link-based route
    frontend_routes = ("reset-password", "auth-success", "auth-error")
    
    # Check if the path starts with any of the known frontend routes
    if any(path.startswith(route) for route in frontend_routes):
        try:
            with open("index.html", "r") as f:
                html_content = f.read()
            return HTMLResponse(content=html_content, status_code=200)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Index file not found for SPA route")
    
    # Handle known static files that might be missing (like favicon)
    if "." in path:
        raise HTTPException(status_code=404, detail="File Not Found")
        
    # Redirect unknown paths back to the root
    log.warning(f"Unknown path '{path}' requested. Redirecting to root.")
    return RedirectResponse(url="/", status_code=status.HTTP_307_TEMPORARY_REDIRECT)
