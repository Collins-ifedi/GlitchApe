# db.py
import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Import the centralized settings object
from settings import settings

# --- Configuration & Setup ---
logger = logging.getLogger(__name__)
DATABASE_URL = settings.DATABASE_URL

# Production-ready check for the database driver.
# The `settings.py` default is SQLite, which is great for development.
# For production (e.g., on Render), DATABASE_URL will be a PostgreSQL URL.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
    logger.info("✅ Connecting to PostgreSQL database.")
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
    logger.info("✅ Connecting to PostgreSQL database.")
else:
    logger.info("✅ Using local SQLite database for development.")


# --- SQLAlchemy Engine & Session ---

# Create an asynchronous engine for database interaction.
engine = create_async_engine(
    DATABASE_URL,
    echo=False,  # Set to True only for debugging to see generated SQL queries.
    
    # --- Connection Pooling (Essential for Production) ---
    # `pool_size`: The number of connections to keep open in the pool.
    # `max_overflow`: The number of extra connections that can be opened beyond `pool_size`.
    # `pool_timeout`: Seconds to wait before giving up on getting a connection from the pool.
    # `pool_recycle`: Seconds after which a connection is automatically recycled.
    #   This is crucial for preventing connections from being terminated by the database
    #   or network infrastructure after a period of inactivity. 1800 seconds (30 minutes) is a common value.
    pool_size=10,
    max_overflow=5,
    pool_timeout=30,
    pool_recycle=1800
)

# Create a session factory for creating new async sessions.
# `expire_on_commit=False` prevents attributes from being expired after commit,
# which is useful when you need to access the object after the session is closed.
async_session_maker = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# Base class for declarative models. All models in `models.py` will inherit from this.
Base = declarative_base()


# --- FastAPI Dependency ---

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides a database session to each request.
    
    Uses an `async with` block to ensure the session is always
    closed correctly, even if an error occurs.
    """
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()