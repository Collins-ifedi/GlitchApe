# server.py
import os
import uuid
import logging
import sys  # <-- ADDED for robust path handling
from datetime import datetime, timedelta, timezone
from typing import Optional, List

import httpx  # <-- Already here, needed for Brevo
from fastapi import (
    FastAPI, Request, Form,
    Depends, HTTPException, status, BackgroundTasks, APIRouter  # <-- ADDED APIRouter
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# --- Integration of Central Brain (MODIFIED for Robust Import) ---
try:
    # Add script's directory to system path for Render environment
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
        
    from glitchape_central_handler import router as central_router
    log.info("Successfully imported 'router' as 'central_router' from glitchape_central_handler.py.")
except ImportError as e:
    log.error(f"CRITICAL: Failed to import 'router' from 'glitchape_central_handler.py'. Error: {e}")
    log.error(f"Current sys.path: {sys.path}")
    log.error(f"Current working directory: {os.getcwd()}")
    try:
        log.error(f"Files in '.': {os.listdir('.')}")
        if 'src' in os.listdir('.'):
             log.error(f"Files in './src': {os.listdir('src')}")
    except Exception as list_e:
        log.error(f"Could not list directory contents: {list_e}")
    central_router = None
except AttributeError as e:
    log.error(f"CRITICAL: 'glitchape_central_handler.py' was found, but 'router' was not defined within it. Error: {e}")
    central_router = None


# --- Import Core Components from Interface ---
try:
    from interface import (
        Base, engine, get_db, get_current_user,
        hash_password, verify_password, create_access_token,
        generate_verification_code,
        User, VerificationToken, ResetToken,
        ChatSession, ChatMessage, OrderRecord, OrderPayment,
        JWT_EXPIRY_MINUTES, AsyncSessionLocal # <-- ADDED AsyncSessionLocal for cleanup job
    )
except ImportError as e:
    log.critical(f"Failed to import from interface.py: {e}")
    raise

# --- Stripe Webhook Secret ---
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
if not STRIPE_WEBHOOK_SECRET:
    log.warning("STRIPE_WEBHOOK_SECRET env var not set. Webhooks will fail.")

# --- ADDED: Brevo (Sendinblue) Email Config ---
BREVO_API_KEY = os.getenv("BREVO_API_KEY")
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
if not BREVO_API_KEY or not EMAIL_SENDER:
    log.warning("BREVO_API_KEY or EMAIL_SENDER env vars not set. Email sending will fail.")

# --- App Initialization ---
app = FastAPI(
    title="GlitchApe Backend API",
    description="Main API server for GlitchApe, handling auth, AI, and orders.",
    version="1.0.0"
)

# --- NEW: Dedicated Router for Authentication and Users ---
# This router will be prefixed with /api to match index.html
auth_router = APIRouter(
    prefix="/auth",
    tags=["Auth & Users"]
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
    # scheduler.add_job(run_cleanup, "interval", hours=1)
    # scheduler.start()
    # log.info("Started periodic cleanup task.")


# --- Helper Functions (Email & Cleanup) ---

# --- MODIFIED: Replaced dummy function with Brevo API call ---
async def send_verification_email(email: str, code: str):
    """
    Sends the verification code email using the Brevo (Sendinblue) API.
    This is an async function run in the background.
    """
    if not BREVO_API_KEY or not EMAIL_SENDER:
        log.error(f"EMAIL TASK FAILED: BREVO_API_KEY or EMAIL_SENDER not set for {email}")
        return False

    api_url = "https://api.brevo.com/v3/smtp/email"
    headers = {
        "api-key": BREVO_API_KEY,
        "accept": "application/json",
        "content-type": "application/json",
    }
    data = {
        "sender": {"email": EMAIL_SENDER, "name": "GlitchApe"},
        "to": [{"email": email}],
        "subject": "Your GlitchApe Verification Code",
        "htmlContent": f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                .container {{ width: 90%; margin: auto; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                .code {{ font-size: 24px; font-weight: bold; color: #FF006E; letter-spacing: 2px; }}
                .footer {{ margin-top: 20px; font-size: 12px; color: #888; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Welcome to GlitchApe! 🐒</h2>
                <p>Thank you for registering. Please use the following code to verify your email address:</p>
                <p class="code">{code}</p>
                <p>This code will expire in 10 minutes.</p>
                <p class="footer">If you did not request this, please ignore this email.</p>
            </div>
        </body>
        </html>
        """,
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(api_url, headers=headers, json=data)
            
            if response.status_code == 201:
                log.info(f"EMAIL TASK: Successfully sent verification code to {email}")
                return True
            else:
                log.error(f"EMAIL TASK FAILED: Brevo API error for {email}. Status: {response.status_code}, Response: {response.text}")
                return False
                
    except Exception as e:
        log.error(f"EMAIL TASK FAILED: Exception during Brevo API call for {email}. Error: {e}")
        return False

async def cleanup_unverified_users(db: AsyncSession):
    """
    Deletes users who are not verified and were created more than 2 hours ago.
    This should be run periodically by a scheduler.
    """
    CLEANUP_THRESHOLD = timedelta(hours=2)
    cutoff_time = datetime.now(timezone.utc) - CLEANUP_THRESHOLD
    
    q_users_to_delete = delete(User).where(
        User.is_verified == False,
        User.created_at < cutoff_time
    ).returning(User.id)

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

async def run_cleanup():
    """Wrapper to create a session for the scheduled cleanup job."""
    log.info("Running periodic cleanup task...")
    async with AsyncSessionLocal() as session:
        await cleanup_unverified_users(session)


# ===================================================================
# AUTHENTICATION ENDPOINTS (MOVED TO auth_router)
# All paths are now relative (e.g., "/register")
# ===================================================================

@auth_router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(
    email: str = Form(...),
    password: str = Form(...),
    country_code: str = Form(None),
    db: AsyncSession = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Handles new user registration.
    Path: /api/auth/register
    """
    user_q = select(User).where(User.email == email)
    existing_user = (await db.execute(user_q)).scalar_one_or_none()

    if existing_user:
        if not existing_user.is_verified:
            log.info(f"Deleting stale unverified user for new registration: {email}")
            await db.delete(existing_user)
            await db.commit()
        else:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="User with this email already exists."
            )

    hashed_password = hash_password(password)
    new_user = User(
        email=email,
        hashed_password=hashed_password,
        is_verified=False,
        country_code=country_code
    )
    db.add(new_user)
    await db.flush()

    verification_code = generate_verification_code()
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(minutes=10)
    
    new_token = VerificationToken(
        code=verification_code,
        user_id=new_user.id,
        expires_at=expires_at,
        last_sent_at=now
    )
    db.add(new_token)
    await db.commit()

    # This will now call the async Brevo function in the background
    background_tasks.add_task(send_verification_email, email, verification_code)
    
    return {
        "message": "Registration successful. Please check your email for a 6-digit verification code.", 
        "email": email
    }


@auth_router.post("/login")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """
    Standard OAuth2 password flow login.
    Path: /api/auth/login
    MODIFIED: Returns user object as expected by index.html
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

    # FIX: Return user object as expected by index.html
    user_data = {
        "id": str(user.id),
        "email": user.email,
        "is_verified": user.is_verified,
        "created_at": user.created_at.isoformat(),
        "country_code": user.country_code
    }

    return {"access_token": access_token, "token_type": "bearer", "user": user_data}


@auth_router.post("/verify-code", status_code=status.HTTP_200_OK)
async def verify_code(
    email: str = Form(...),
    code: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Verifies a user's account using the 6-digit code.
    Path: /api/auth/verify-code
    """
    user_q = select(User).where(User.email == email)
    user = (await db.execute(user_q)).scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    
    if user.is_verified:
        return {"message": "Account already verified."}

    token_q = select(VerificationToken).where(
        VerificationToken.user_id == user.id,
        VerificationToken.code == code
    )
    token = (await db.execute(token_q)).scalar_one_or_none()

    if not token:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid verification code.")

    if token.expires_at < datetime.now(timezone.utc):
        await db.delete(token)
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Verification code has expired. Please request a new one."
        )

    user.is_verified = True
    await db.delete(token)
    await db.commit()

    return {"message": "Account successfully verified."}


@auth_router.post("/resend-code", status_code=status.HTTP_200_OK)
async def resend_code(
    email: str = Form(...),
    db: AsyncSession = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Resends a new verification code.
    Path: /api/auth/resend-code
    """
    user_q = select(User).where(User.email == email)
    user = (await db.execute(user_q)).scalar_one_or_none()

    if not user or user.is_verified:
        log.warning(f"Resend code request for non-existent or verified user: {email}")
        return {"message": "If an unverified account exists for this email, a new code has been sent."}

    token_q = select(VerificationToken).where(VerificationToken.user_id == user.id)
    token = (await db.execute(token_q)).scalar_one_or_none()
    
    now = datetime.now(timezone.utc)
    RESEND_DELAY_SECONDS = 60
    TOKEN_EXPIRY_MINUTES = 10

    if token:
        can_resend_at = token.last_sent_at + timedelta(seconds=RESEND_DELAY_SECONDS)
        if now < can_resend_at:
            wait_seconds = int((can_resend_at - now).total_seconds()) + 1
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Please wait {wait_seconds} seconds before requesting a new code."
            )

    new_code = generate_verification_code()
    new_expires_at = now + timedelta(minutes=TOKEN_EXPIRY_MINUTES)
    
    if token:
        token.code = new_code
        token.expires_at = new_expires_at
        token.last_sent_at = now
    else:
        token = VerificationToken(
            user_id=user.id,
            code=new_code,
            expires_at=new_expires_at,
            last_sent_at=now
        )
        db.add(token)

    await db.commit()
    # This will now call the async Brevo function in the background
    background_tasks.add_task(send_verification_email, email, new_code)
    return {"message": "A new verification code has been sent to your email."}


@auth_router.post("/forgot-password")
async def forgot_password(
    email: str = Form(...),
    db: AsyncSession = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Handles a forgot password request.
    Path: /api/auth/forgot-password
    """
    q = select(User).where(User.email == email)
    r = await db.execute(q)
    user = r.scalar_one_or_none()
    
    if not user:
        log.warning(f"Forgot password attempt for non-existent user: {email}")
        return {"message": "If an account with this email exists, a password reset link has been sent."}

    await db.execute(delete(ResetToken).where(ResetToken.user_id == user.id))

    token = str(uuid.uuid4())
    expires = datetime.now(timezone.utc) + timedelta(hours=1)
    reset_token = ResetToken(
        token=token,
        user_id=user.id,
        expires_at=expires
    )
    db.add(reset_token)
    await db.commit()

    reset_link = f"https://glitchape-by.onrender.com/reset-password?token={token}" # Use production URL
    # TODO: Create a new 'send_password_reset_email' async function using Brevo
    # background_tasks.add_task(send_password_reset_email, email, reset_link)
    log.info(f"PASSWORD RESET LINK for {email}: {reset_link}")

    return {"message": "If an account with this email exists, a password reset link has been sent."}


@auth_router.post("/reset-password")
async def reset_password(
    token: str = Form(...),
    new_password: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Resets a user's password using a valid token.
    Path: /api/auth/reset-password
    """
    q = select(ResetToken).where(ResetToken.token == token)
    r = await db.execute(q)
    token_data = r.scalar_one_or_none()
    
    if not token_data:
        raise HTTPException(status_code=400, detail="Invalid token")
        
    if token_data.expires_at < datetime.now(timezone.utc):
        await db.delete(token_data) # Clean up expired token
        await db.commit()
        raise HTTPException(status_code=400, detail="Token expired")
        
    q_user = select(User).where(User.id == token_data.user_id)
    r_user = await db.execute(q_user)
    user = r_user.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=4404, detail="User not found")
        
    user.hashed_password = hash_password(new_password)
    
    await db.delete(token_data)
    await db.commit()
    
    return {"message": "Password reset successful"}


@auth_router.get("/me", response_model=None)
async def read_users_me(current_user: User = Depends(get_current_user)):
    """
    Fetches the current authenticated user's details.
    Path: /api/auth/me
    """
    return {
        "id": current_user.id,
        "email": current_user.email,
        "is_verified": current_user.is_verified,
        "created_at": current_user.created_at,
        "country_code": current_user.country_code
    }

@auth_router.post("/delete", status_code=status.HTTP_200_OK)
async def delete_account(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Deletes the currently authenticated user's account.
    Path: /api/auth/delete
    """
    try:
        await db.delete(current_user)
        await db.commit()
        return {"message": "Account deleted successfully."}
    except Exception as e:
        await db.rollback()
        log.error(f"Failed to delete user {current_user.email}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete account.")


# =======================================
# ROUTER INCLUSION (MODIFIED)
# =======================================

# 1. Include the Central AI & Orders router
if central_router:
    app.include_router(central_router, prefix="/api")
else:
    log.error("Central router not included. AI/Order endpoints will be missing.")

# 2. Include the new Auth router
app.include_router(auth_router, prefix="/api") # All auth routes are now /api/auth/...


# =======================================
# FRONTEND SERVING (Unchanged)
# =======================================

@app.get("/")
async def serve_frontend_root():
    """Serves the main index.html file."""
    try:
        with open("index.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        return {"message": "GlitchApe Backend", "status": "Frontend file not found"}


@app.get("/{path:path}")
async def serve_frontend_spa(path: str):
    """
    Catch-all route to serve the index.html for known Single Page Application (SPA) paths.
    """
    frontend_routes = ("reset-password", "auth-success", "auth-error")
    
    if any(path.startswith(route) for route in frontend_routes):
        try:
            with open("index.html", "r") as f:
                html_content = f.read()
            return HTMLResponse(content=html_content, status_code=200)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Index file not found for SPA route")
    
    # Don't redirect API calls
    if path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API endpoint not found")

    # Handle known static files that might be missing (like favicon)
    if "." in path:
        raise HTTPException(status_code=404, detail="File Not Found")
        
    log.warning(f"Unknown path '{path}' requested. Redirecting to root.")
    return RedirectResponse(url="/", status_code=status.HTTP_307_TEMPORARY_REDIRECT)