# designs.py
import asyncio
import logging
from typing import Dict, List, Optional, Any
import uuid  # MODIFIED: Added for generating unique filenames
import os # MODIFIED: Added to access environment variables for blob storage

from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile # MODIFIED: Added File and UploadFile
from pydantic import BaseModel, Field, HttpUrl
import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime
from vercel_blob import put as vercel_put # MODIFIED: Added Vercel Blob SDK for uploads

# Local application imports
from db import get_db
from models import Design, User
from auth import get_current_user
from settings import settings

# --- Module-level Configuration ---
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/designs", tags=["Designs"])

# Placement normalization helps standardize keys from Printful's API
PLACEMENT_ALIASES = {
    "front": "front",
    "back": "back",
    "left": "left",
    "right": "right",
    "sleeve_left": "left",
    "sleeve_right": "right",
    "outside_label": "collar",
    "label_outside": "collar",
    "collar": "collar",
}


# ===================================================================
# Pydantic Schemas for API Contracts
# ===================================================================

class GenerateDesignRequest(BaseModel):
    """Request body for generating a new AI design."""
    prompt: str = Field(..., min_length=10, max_length=500, example="A glowing cyberpunk skull on the front, with neon circuit patterns on the sleeves.")
    product_id: int = Field(..., gt=0, example=1)
    placements: Optional[List[str]] = Field(
        None,
        description="Optional list of specific placements to generate (e.g., ['front', 'back']). If omitted, all available placements are used.",
        example=["front", "sleeve_left"]
    )

class SaveDesignRequest(BaseModel):
    """Request body for saving a user-uploaded design."""
    product_id: int = Field(..., gt=0, example=1)
    design_url: HttpUrl = Field(..., example="https://storage.googleapis.com/user_designs/design123.png")

# MODIFIED: Added a new response model for the upload endpoint
class UploadResponse(BaseModel):
    """Response model for a successful file upload."""
    url: HttpUrl

class DesignResponse(BaseModel):
    """Standard response model for a design object."""
    id: str
    user_id: str
    product_id: int
    design_url: HttpUrl
    prompt: Optional[str]
    mockup_urls: Dict[str, HttpUrl]
    created_at: datetime

    class Config:
        from_attributes = True

class MockupTemplatesResponse(BaseModel):
    """Response model for available mockup templates for a product."""
    pass # Inferred from usage, the original `__root__` is deprecated. Will return a Dict.


# ===================================================================
# HTTP Client & External Service Utilities
# ===================================================================

async def _make_request(method: str, url: str, **kwargs) -> Any:
    """
    A robust, retry-enabled async HTTP request helper.

    Args:
        method: The HTTP method ('GET' or 'POST').
        url: The URL to request.
        **kwargs: Additional arguments for httpx.AsyncClient.request.

    Returns:
        The JSON response from the server.

    Raises:
        HTTPException: If the request fails after all retries.
    """
    last_exc = None
    # NOTE: Assuming these settings exist or will be added to settings.py
    timeout = getattr(settings, 'DESIGNS_HTTP_TIMEOUT', 30.0)
    max_retries = getattr(settings, 'DESIGNS_MAX_RETRIES', 2)

    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(max_retries + 1):
            try:
                response = await client.request(method, url, **kwargs)
                response.raise_for_status()
                return response.json()
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                last_exc = e
                logger.warning(
                    f"HTTP {method} to {url} failed (attempt {attempt+1}/{max_retries + 1}): {e}"
                )
                if attempt < max_retries:
                    # Exponential backoff: 0.5s, 1s, 2s
                    await asyncio.sleep(0.5 * (2 ** attempt))
            except Exception as e:
                last_exc = e
                logger.exception(f"An unexpected error occurred during HTTP request to {url}")
                break  # Don't retry on unexpected errors

    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=f"External service at {url} is unavailable after multiple retries: {last_exc}"
    )


async def get_printful_mockup_templates(product_id: int) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Fetches available mockup templates and their properties from Printful.
    """
    if not settings.PRINTFUL_API_KEY:
        logger.warning("PRINTFUL_API_KEY is not set. Cannot fetch mockup templates.")
        return {}

    url = f"https://api.printful.com/mockup-generator/templates/{product_id}"
    headers = {"Authorization": f"Bearer {settings.PRINTFUL_API_KEY}"}

    try:
        data = await _make_request("GET", url, headers=headers)
        result = data.get("result", [])
        
        placements = {}
        for item in result:
            raw_key = item.get("placement")
            if not raw_key:
                continue
            
            placement_key = PLACEMENT_ALIASES.get(raw_key, raw_key)
            if placement_key and item.get("image_url"):
                placements[placement_key] = {
                    "base": item.get("image_url"),
                    "mask": item.get("mask_url"),
                }
        return placements
    except HTTPException as e:
        logger.error(f"Failed to fetch Printful templates for product {product_id}: {e.detail}")
        return {}


async def generate_image_with_ai(base_image_url: str, mask_url: Optional[str], prompt: str) -> HttpUrl:
    """
    Generates an image by calling the configured AI image generation service (Gemini).
    """
    if not settings.GEMINI_API_KEY:
        logger.error("Gemini API is not configured.")
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="AI generation service is not configured.")
    
    # Assuming GEMINI_API_URL is the correct setting for the service endpoint
    gemini_url = getattr(settings, 'GEMINI_API_URL', 'https://api.gemini.com/v1/images/generations') # Example URL

    payload = {
        "prompt": prompt,
        "image_url": base_image_url,
        "mask_url": mask_url,
        "size": "1024x1024",
        "n": 1,
    }
    headers = {"Authorization": f"Bearer {settings.GEMINI_API_KEY}"}
    
    response_data = await _make_request("POST", gemini_url, json=payload, headers=headers)
    
    # Robustly parse the response to find the output URL
    if isinstance(response_data, dict):
        # Look for a list of images in common response structures
        for key in ("data", "outputs", "results", "images"):
            if isinstance(response_data.get(key), list) and response_data[key]:
                first_item = response_data[key][0]
                if isinstance(first_item, dict) and first_item.get("url"):
                    return first_item["url"]
    
    logger.error(f"Could not find a valid image URL in the AI API response: {response_data}")
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="AI service returned an unexpected response format.")


# ===================================================================
# API Endpoints
# ===================================================================

# MODIFIED: Added a new endpoint to handle file uploads
@router.post("/upload/", response_model=UploadResponse, status_code=status.HTTP_201_CREATED, summary="Upload a design image")
async def upload_design_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """
    Handles uploading a user's design image to cloud storage.

    This endpoint accepts a multipart/form-data request with an image file,
    uploads it to Vercel Blob storage, and returns the permanent, public URL.

    **Note**: Requires `BLOB_READ_WRITE_TOKEN` environment variable to be set.
    """
    try:
        # Generate a unique filename to prevent overwrites
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"designs/{current_user.id}/{uuid.uuid4()}{file_extension}"

        # Read the file content
        contents = await file.read()

        # Upload to Vercel Blob
        # The `vercel-blob` library automatically uses the BLOB_READ_WRITE_TOKEN env var.
        blob_result = vercel_put(
            pathname=unique_filename,
            body=contents,
            access='public' # Ensure the file is publicly accessible
        )
        
        return {"url": blob_result['url']}

    except Exception as e:
        logger.exception(f"File upload failed for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not upload the file. Please try again later."
        )


@router.post("/generate/", response_model=DesignResponse, status_code=status.HTTP_201_CREATED, summary="Generate a new design with AI")
async def generate_design(
    req: GenerateDesignRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Generates a multi-placement design for a specified product using an AI prompt.

    This endpoint performs the following steps:
    1.  Fetches available mockup templates for the given `product_id` from Printful.
    2.  Asynchronously calls an AI image generation service for each required placement.
    3.  Collects the generated image URLs.
    4.  Saves the new design and its associated mockups to the database.
    """
    templates = await get_printful_mockup_templates(req.product_id)
    if not templates:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No mockup templates found for product ID {req.product_id}.")

    # Determine which placements to generate
    target_placements = [p for p in (req.placements or list(templates.keys())) if p in templates]
    if not target_placements:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="None of the requested placements are valid for this product.")

    # Create generation tasks to run in parallel
    async def _generate_for_placement(placement: str):
        template = templates[placement]
        try:
            # Create a more specific prompt for each placement
            section_prompt = f"{req.prompt} (Design for {placement.replace('_', ' ')} area)"
            result_url = await generate_image_with_ai(template["base"], template["mask"], section_prompt)
            return placement, result_url
        except Exception as e:
            logger.error(f"Failed to generate image for placement '{placement}': {e}")
            return placement, None

    tasks = [_generate_for_placement(p) for p in target_placements]
    results = await asyncio.gather(*tasks)

    # Collect successful results
    edited_urls = {placement: url for placement, url in results if url}
    if not edited_urls:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="AI failed to generate an image for all requested placements.")

    # Save the design to the database
    primary_url = edited_urls.get("front") or next(iter(edited_urls.values()))
    new_design = Design(
        user_id=current_user.id,
        product_id=req.product_id,
        design_url=str(primary_url),
        prompt=req.prompt,
        mockup_urls=edited_urls
    )
    
    try:
        db.add(new_design)
        await db.commit()
        await db.refresh(new_design)
    except Exception as e:
        await db.rollback()
        logger.exception("Database error while saving a new design.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not save the generated design.")
    
    return new_design


@router.get("/", response_model=List[DesignResponse], summary="List user's designs")
async def list_user_designs(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieves a list of all designs created by the currently authenticated user,
    sorted by creation date in descending order.
    """
    query = (
        select(Design)
        .where(Design.user_id == current_user.id)
        .order_by(Design.created_at.desc())
    )
    result = await db.execute(query)
    designs = result.scalars().all()
    return designs


@router.post("/", response_model=DesignResponse, status_code=status.HTTP_201_CREATED, summary="Save a new design")
async def save_design(
    payload: SaveDesignRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Saves a user-provided design URL to their account. This is typically used
    for designs uploaded manually rather than generated by AI.
    """
    new_design = Design(
        user_id=current_user.id,
        product_id=payload.product_id,
        design_url=str(payload.design_url),
        prompt="User uploaded design",
        mockup_urls={"custom": str(payload.design_url)}
    )
    try:
        db.add(new_design)
        await db.commit()
        await db.refresh(new_design)
    except Exception as e:
        await db.rollback()
        logger.exception("Database error while saving a user-uploaded design.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not save the design.")
    
    return new_design


@router.get("/mockups/{product_id}", response_model=Dict[str, HttpUrl], summary="Get mockup templates for a product")
async def fetch_mockup_templates(
    product_id: int,
    current_user: User = Depends(get_current_user)
):
    """
    Retrieves the base mockup image URLs for all available placements of a given product.
    This is useful for the frontend to display product templates before AI generation.
    """
    templates = await get_printful_mockup_templates(product_id)
    if not templates:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No mockup templates found for this product.")

    # Return only the base image URLs
    base_urls = {placement: urls["base"] for placement, urls in templates.items() if urls.get("base")}
    if not base_urls:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No mockup preview URLs are available for this product.")
    
    return base_urls