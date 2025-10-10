# server.py
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from db import Base, engine
from settings import settings

# --- Routers ---
# Import all the routers from your application modules
from auth import router as auth_router
from designs import router as designs_router
from orders import router as orders_router
from referrals import router as referrals_router
from products import router as products_router

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Application Lifespan Events ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's lifespan.
    - On startup, it logs the application start.
    - On shutdown, it disposes of the database engine's connection pool.
    """
    # Startup logic
    logger.info("🚀 Starting up application...")
    # Note: The original 'create_all' logic was commented out, which is good practice.
    # Production environments should use a migration tool like Alembic.
    logger.info("✅ Application startup complete.")
    
    yield  # The application runs while the context manager is active

    # Shutdown logic
    logger.info("...Shutting down application")
    await engine.dispose()
    logger.info("✅ Database connection pool closed.")


# --- FastAPI Application Initialization ---

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Backend API for an AI-powered design studio with Printful, Stripe, and referrals.",
    version="1.0.0",
    lifespan=lifespan,  # Use the new lifespan context manager
    contact={
        "name": "GlitchApe Support",
        "url": "https://x.com/GlitchApeFun",
        "email": "collins@glitchape.fun",
    },
)


# --- Middleware ---

# Global Exception Handler
# This middleware catches any unhandled exception and returns a generic 500 error
# to prevent leaking sensitive stack trace information to the client.
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception for request {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred."},
    )

# CORS (Cross-Origin Resource Sharing) Middleware
# This allows the frontend application (running on a different domain)
# to communicate with this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- API Routers ---
# Include the routers from different modules to organize the API endpoints.

app.include_router(auth_router, prefix=settings.API_V1_PREFIX)
app.include_router(designs_router, prefix=settings.API_V1_PREFIX)
app.include_router(orders_router, prefix=settings.API_V1_PREFIX)
app.include_router(referrals_router, prefix=settings.API_V1_PREFIX)
app.include_router(products_router, prefix=settings.API_V1_PREFIX)


# --- Root Endpoint ---

@app.get("/", tags=["Root"])
def read_root():
    """
    A simple root endpoint to confirm the API is running.
    """
    return {"message": f"🚀 {settings.PROJECT_NAME} API is running!"}
