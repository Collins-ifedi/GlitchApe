# settings.py
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Project
    PROJECT_NAME: str = "GlitchApe Backend"
    API_V1_PREFIX: str = "/api"

    # Security
    JWT_SECRET: str = os.getenv("JWT_SECRET", "super-secret-key")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_MINUTES: int = 60 * 24  # 24h

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "sqlite+aiosqlite:///./dev.db"  # default local SQLite
    )

    # Stripe
    STRIPE_SECRET_KEY: str = os.getenv("STRIPE_SECRET_KEY", "")
    STRIPE_WEBHOOK_SECRET: str = os.getenv("STRIPE_WEBHOOK_SECRET", "")

    # Printful
    PRINTFUL_API_KEY: str = os.getenv("PRINTFUL_API_KEY", "")

    # AI keys
    LLAMA_API_KEY: str = os.getenv("LLAMA_API_KEY", "")
    LLAMA_API_URL: str = os.getenv("LLAMA_API_URL", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    BREVO_API_KEY: str = os.getenv("BREVO_API_KEY", "")
    ALLOWED_HOSTS: str = os.getenv("ALLOWED_HOSTS", "")
    JWT_KEY: str = os.getenv("JWT_KEY", "")

    # Frontend URL (CORS)
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:5173")

    class Config:
        env_file = ".env"
        case_sensitive = True


# ✅ Instantiate settings globally
settings = Settings()