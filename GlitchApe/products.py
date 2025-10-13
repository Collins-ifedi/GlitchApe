# products.py
import httpx
import logging
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, ValidationError

from settings import settings

# --- Configuration & Setup ---
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/products", tags=["Products"])
PRINTFUL_API_URL = "https://api.printful.com/products"


# --- Pydantic Schemas for Data Validation ---

class ProductOut(BaseModel):
    """Defines the structure of a product returned by our API."""
    id: int
    name: str
    category: str
    base_price: float

    class Config:
        from_attributes = True

class _PrintfulPrice(BaseModel):
    """Internal model to parse the 'price' object from Printful's response."""
    min: float = 0.0

class _PrintfulProduct(BaseModel):
    """Internal model to validate a single product from Printful's API."""
    id: int
    name: str
    type: Optional[str] = "unknown"
    # MODIFIED: Made the 'price' field optional to handle cases where the API
    # response for a product does not include pricing information.
    price: Optional[_PrintfulPrice] = None

class _PrintfulApiResponse(BaseModel):
    """Internal model to validate the top-level structure of Printful's API response."""
    result: List[_PrintfulProduct]


# --- Fallback Data ---
# This data is used if the Printful API is unavailable, ensuring the frontend
# remains functional. In a larger system, this could be sourced from a cache like Redis.
FALLBACK_PRODUCTS: List[ProductOut] = [
    ProductOut(id=1, name="Classic T-Shirt", category="shirts", base_price=19.99),
    ProductOut(id=2, name="Glitch Hoodie", category="jackets", base_price=39.99),
    ProductOut(id=4, name="Cyber Polo", category="polos", base_price=29.99),
    ProductOut(id=3, name="Neon Kicks", category="sneakers", base_price=59.99),
]


# --- API Endpoint ---

@router.get("/", response_model=List[ProductOut], summary="List all available products")
async def list_products():
    """
    Fetches the list of available products from the Printful API.

    This endpoint is designed for resilience:
    - It makes an asynchronous call to Printful to avoid blocking.
    - It validates the incoming data structure using Pydantic models.
    - If the Printful API fails for any reason (e.g., network error, invalid data),
      it gracefully serves a hardcoded list of fallback products.
    """
    if not settings.PRINTFUL_API_KEY:
        logger.warning("PRINTFUL_API_KEY is not set. Serving fallback products.")
        return FALLBACK_PRODUCTS

    headers = {"Authorization": f"Bearer {settings.PRINTFUL_API_KEY}"}

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            logger.info("📡 Fetching products from Printful API...")
            response = await client.get(PRINTFUL_API_URL, headers=headers)
            response.raise_for_status()  # Raises an exception for 4xx or 5xx status codes

            # Validate the entire API response structure
            validated_response = _PrintfulApiResponse.parse_obj(response.json())
            
            # Transform the validated Printful data into our public ProductOut model
            products = [
                ProductOut(
                    id=item.id,
                    name=item.name,
                    category=item.type or "unknown",
                    base_price=item.price.min if item.price else 0.0
                ) for item in validated_response.result
            ]
            
            logger.info(f"✅ Successfully retrieved and validated {len(products)} products from Printful.")
            return products

    except (httpx.RequestError, httpx.HTTPStatusError, ValidationError) as e:
        logger.error(f"⚠️ Printful API call failed ({type(e).__name__}), serving fallback products. Error: {e}")
        return FALLBACK_PRODUCTS
    except Exception as e:
        logger.exception(f"An unexpected error occurred while fetching products: {e}")
        return FALLBACK_PRODUCTS