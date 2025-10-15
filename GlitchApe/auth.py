# auth.py
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from db import get_db
from models import User
from settings import settings

# ===================================================================
# Pydantic Schemas (Data Validation)
# ===================================================================

class UserCreate(BaseModel):
    """Schema for user registration request."""
    email: EmailStr
    password: str

class UserPublic(BaseModel):
    """Schema for safely exposing user data."""
    id: str
    email: EmailStr
    created_at: datetime

    class Config:
        orm_mode = True

class Token(BaseModel):
    """Schema for the authentication token response."""
    access_token: str
    token_type: str

class TokenData(BaseModel):
    """Schema for data inside the JWT."""
    user_id: Optional[int] = None


# ===================================================================
# Configuration
# ===================================================================

router = APIRouter(prefix="/auth", tags=["Auth"])

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# ===================================================================
# Utility Functions
# ===================================================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against a hashed one."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hashes a plain password."""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Creates a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.JWT_EXPIRY_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt

async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    """Fetches a user from the database by email."""
    result = await db.execute(select(User).where(User.email == email))
    return result.scalars().first()


# ===================================================================
# Current User Dependency
# ===================================================================

async def get_current_user(
    token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)
) -> User:
    """Dependency to get the current authenticated user from a token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        user_id = payload.get("user_id")
        if user_id is None:
            raise credentials_exception
        token_data = TokenData(user_id=user_id)
    except JWTError:
        raise credentials_exception

    user = await db.get(User, token_data.user_id)
    if user is None:
        raise credentials_exception
    return user


# ===================================================================
# API Endpoints
# ===================================================================

@router.post("/register", status_code=status.HTTP_201_CREATED, response_model=UserPublic)
async def register_user(
    user_in: UserCreate,
    ref: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Handles new user registration.
    - Hashes the password for security.
    - Assigns a referrer if a referral ID (`ref`) is provided.
    """
    # ❗️ Local import to prevent circular dependency with 'referrals.py'
    from referrals import assign_referrer

    existing_user = await get_user_by_email(db, user_in.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email already exists.",
        )

    hashed_password = get_password_hash(user_in.password)
    new_user = User(email=user_in.email, hashed_password=hashed_password)

    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    # Handle referral assignment if 'ref' query parameter is present
    if ref:
        # Note: This assumes 'assign_referrer' is also an async function.
        # You will need to update referrals.py to use `async def`.
        await assign_referrer(new_user, ref, db)

    return {
    "id": str(new_user.id),
    "email": new_user.email,
    "created_at": new_user.created_at
    }


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)
):
    """
    Handles user login and returns a JWT access token.
    Uses OAuth2PasswordRequestForm, expecting form-data (`username` and `password`).
    """
    user = await get_user_by_email(db, form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"user_id": str(user.id)})
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=UserPublic)
async def read_users_me(current_user: User = Depends(get_current_user)):
    """
    Fetches the profile of the currently authenticated user.
    """
    return current_user
