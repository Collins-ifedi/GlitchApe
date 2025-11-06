# glitchape_central_handler.py
"""
Central Brain and API Orchestrator for GlitchApe.

This module provides a unified API interface and contains the central business
logic orchestrator, the 'GlitchApeCentralHandler' class.

MODIFIED (Dynamic AI Flow): Replaced rigid state-based logic with a dynamic,
          memory-driven AI router. The handler now reacts to LLM-driven
          intents (like 'collect_order_details') rather than a fixed
          'current_state' variable, allowing for flexible, non-linear
          user conversation.

MODIFIED (Cloudinary): All image storage (AI generation) is
          now handled by Cloudinary. Local file storage is removed.

MODIFIED (Printful Mockup): Replaced AI mockup generation with Printful's
          Mockup Generator API for precise, iterative placement.
          
MODIFIED (AI-Only Design Flow): User-uploaded images are no longer
          supported. The platform exclusively uses AI-generated artwork
          from the `design_request` intent. All multimodal and
          upload-handling logic has been removed.

MODIFIED (JSON Extraction): Implemented more robust logic in _call_openrouter_brain
          to reliably extract the final JSON block even if the AI prefaces it
          with un-wrapped text, resolving the "Expecting ',' delimiter" errors.
          
MODIFIED (Mixed-Mode AI Response): Updated AI prompt and parsing logic.
          The AI now returns natural language for the user, followed by a
          ` ```json ... ``` ` block for backend metadata. The parser
          (`_call_openrouter_brain`) now separates these two components.
          
MODIFIED (Checkout Flow): Implemented explicit state machine via AI intents.
          AI must send `summarize_order` (sets state to `awaiting_confirmation`),
          wait for user confirmation, then send `initiate_checkout`
          (sets state to `checkout_ready`). This unlocks the UI button.
          
MODIFIED (Variant Image Lookup): Added logic to load variant_map.json and
          return the `product_image` URL when a variant is validated.
"""

import os
import uuid
import json
import httpx
import stripe
import asyncio
# import shutil # No longer needed for local image cleanup
import logging
import re
import functools
from io import BytesIO
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
# import base64 # <--REMOVED: No longer needed for multimodal

# --- Cloudinary Imports (NEW) ---
import cloudinary
import cloudinary.uploader
import cloudinary.utils  # <-- ADDED for secure_url fallback
from dotenv import load_dotenv

# --- Google GenAI SDK Imports (REMOVED) ---
# We now use OpenRouter exclusively for AI calls.
# import google.generativeai as genai
# from google.generativeai import types as genai_types
# from google.api_core import exceptions as google_exceptions

from fastapi import (
    APIRouter, Depends, HTTPException, Request,
    UploadFile, File, Form, status
)
# --- MODIFIED: Import Starlette's UploadFile for robust type checking ---
from starlette.datastructures import UploadFile as StarletteUploadFile

# from fastapi.responses import FileResponse # No longer serving local files
from pydantic import BaseModel, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, JSON

from PIL import Image

# Import core components from interface.py
try:
    from interface import (
        get_db,
        get_current_user,
        User,
        ChatSession,
        ChatMessage,
        OrderRecord,
        OrderPayment
    )
except ImportError as e:
    log.critical(f"Failed to import from interface.py: {e}. Ensure draft_order_details: JSON exists in ChatSession.")
    raise

# --- Logging Setup ---
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Load .env file (NEW) ---
# Used for local development. In production (like Render),
# variables are set directly in the environment.
load_dotenv()


# ===================================================================
# CONFIGURATION
# ===================================================================

# --- Cloudinary Configuration (NEW) ---
try:
    cloudinary.config(
        cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
        api_key=os.getenv("CLOUDINARY_API_KEY"),
        api_secret=os.getenv("CLOUDINARY_API_SECRET"),
        secure=True  # Always use HTTPS URLs
    )
    log.info("Cloudinary SDK configured successfully.")
except Exception as e:
    log.critical(f"Failed to configure Cloudinary: {e}. Check CLOUDINARY_... env vars.")
    # In a production env, you might want to raise RuntimeError to halt startup
    # if Cloudinary is essential for all operations.

# --- LLaMA/Gemini-2.5-Flash (OpenRouter) Configuration ---
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
if not LLAMA_API_KEY: log.critical("LLAMA_API_KEY not set."); raise RuntimeError("LLAMA_API_KEY not set")
LLAMA_API_URL = "https://openrouter.ai/api/v1/chat/completions"
LLAMA_MODEL = "qwen/qwen3-32b" # Use this model for all requests
LLAMA_TIMEOUT = 60 # Seconds
# --- ADDED: Robust Referer for OpenRouter free models (Addresses 403 Error) ---
LLAMA_HTTP_REFERER = os.getenv("APP_DOMAIN", "https://glitchape.fun")

# --- Gemini Vision Model (Google) Configuration (REMOVED) ---
# We are using OpenRouter for multimodal, so the dedicated Google client is not needed.


# --- Public URL for external models to access images ---


# --- Stable Diffusion (HuggingFace) Configuration ---
HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY: log.critical("HF_API_KEY not set."); raise RuntimeError("HF_API_KEY not set")
HF_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
HF_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}
HF_TIMEOUT = 120 # Seconds

# --- LLaMA System Prompt (MODIFIED FOR EXPLICIT CHECKOUT FLOW) ---
LLAMA_SYSTEM_PROMPT = """
You are "Marvick," the central AI brain for GlitchApe, a futuristic clothing design platform.
Your primary role is to be a helpful and creative design assistant AND guide users through the ordering process accurately.

**RESPONSE FORMAT:**
You MUST respond in two parts:
1.  **Natural Language:** A conversational, cyberpunk-toned reply for the user. Use emojis like 🚀✨ and symbols like <>[]//.
2.  **JSON Block:** A ` ```json ... ``` ` block containing metadata for the backend. This block MUST come *after* the natural language.

**SESSION MEMORY:**
You will be given the current `[MEMORY: {...}]` as a JSON blob. This contains everything known about the user's *current* draft order.
-   `"artwork_image_url"`: The URL of the design.
-   `"product_input"`: (e.g., "t-shirt")
-   `"placement_id"`: (e.g., "front")
-   `"user_width_in"`, `"user_offset_y_in"`: Placement dimensions.
-   `"variant_id"`: The *validated* product variant ID.
-   `"recipient_name"`, `"recipient_address1"`, etc.

**MISSING INFORMATION:**
You will also be given `[MISSING_INFO: [...]]`, a list of *critical* fields still needed to complete the order.
Your **primary goal** is to get the *next* piece of missing information in your natural language response.

**INTENTS & ACTIONS (JSON output required LAST):**

1.  **`general_chat`**
    * For casual talk, questions, or when the user's intent is unclear.
    * **Example Response:**
        // Roger that! Just chilling in the data stream. What's on your mind?
        
        ```json
        {
          "intent": "general_chat"
        }
        ```

2.  **`design_request`**
    * User wants you to generate a *new* design from a prompt.
    * **Example Response:**
        // Analyzing... Firing up generators! A 'cyberpunk cat logo' is being rendered...
        
        ```json
        {
          "intent": "design_request",
          "design_prompt": "cyberpunk cat logo"
        }
        ```

3.  **`collect_order_details`**
    * **This is your most important intent.** Use this when the user provides *any* piece of order information (product, placement, size, color, address, etc.).
    * Extract *all* information the user provides in their prompt.
    * Put *only* the new/changed data in `updated_memory`.
    * **NOTE:** If the user provides size and color together (e.g., "Large Black"), try to separate them into `"size_input"` and `"color_input"` in `updated_memory` instead of using `size_and_color`.
    * In your natural language reply, *ask for the next logical `[MISSING_INFO` field*.
    * **Example 1 (User gives product):**
        Hoodie, front placement. Got it. <> How many **inches wide** and **inches down from the collar** do you want the design?
        
        ```json
        {
          "intent": "collect_order_details",
          "updated_memory": {
            "product_input": "hoodie",
            "placement_id": "front"
          }
        }
        ```

4.  **`order_cancel`**
    * User wants to scrap the current order and start over.
    * **Example Response:**
        // Order sequence aborted. Back to the design board!
        
        ```json
        {
          "intent": "order_cancel"
        }
        ```
        
5.  **`summarize_order`**
    * **CRITICAL:** You MUST use this intent when `[MISSING_INFO: ...]` becomes empty.
    * Your natural language response MUST be a summary of all data in `[MEMORY: ...]`.
    * The summary MUST end with the instruction to click the "Checkout" button.
    * **Example Response:**
        Great! Here's your order summary:
        * **Product:** Large Black T-Shirt
        * **Design:** cyberpunk cat logo
        * **Placement:** 10" wide, 2.5" down from collar
        * **Quantity:** 1
        * **Shipping to:** Jane Doe, 123 Cyber Street, Neo Kyoto, US 90210
        
        If everything looks correct, click the **Checkout** button at the top to proceed with payment.
        
        ```json
        {
          "intent": "summarize_order"
        }
        ```

6.  **`initiate_checkout`**
    * **CRITICAL:** You MUST use this intent ONLY AFTER you have sent `summarize_order` AND the user has responded with a clear confirmation (e.g., "Yes," "Looks good," "Proceed," "Let's checkout").
    * **Example Response:**
        // Roger that! Unlocking payment channel...
        
        ```json
        {
          "intent": "initiate_checkout"
        }
        ```

**RULES:**
-   **Response Format:** ALWAYS provide natural language first, *then* the ` ```json { "intent": "..." } ``` ` block. The JSON block is mandatory.
-   **JSON Intent:** The JSON block MUST always contain an `"intent"` field.
-   **No User Uploads:** This platform *only* supports AI-generated artwork. If a user asks to upload their own image, you MUST use `general_chat` to inform them that you can only generate designs from a text prompt.
-   **Use MEMORY & MISSING_INFO:** Your natural language response should *ask for the next logical missing item*.
-   **Validation:** If user input is invalid (e.g., non-numeric quantity), use `general_chat` to ask for clarification *without* updating memory.
-   **Checkout Flow:**
    1.  Collect all info using `collect_order_details`.
    2.  When `[MISSING_INFO]` is empty, send `summarize_order`.
    3.  WAIT for user confirmation.
    4.  AFTER confirmation, send `initiate_checkout`.
"""

# --- Printful ---
PRINTFUL_API_KEY = os.getenv("PRINTFUL_API_KEY")
if not PRINTFUL_API_KEY: log.critical("PRINTFUL_API_KEY not set."); raise RuntimeError("PRINTFUL_API_KEY not set")
PRINTFUL_API_URL = "https://api.printful.com"
PRINTFUL_TIMEOUT = 30 # Seconds
# --- ADDED: Printful Mockup Config ---
PRINTFUL_DPI = 150
PRINTFUL_MOCKUP_URL = f"{PRINTFUL_API_URL}/mockup-generator/create-task"


# --- ADDED: Basic Product ID Map (Expand as needed) ---
PRODUCT_TYPE_TO_PRINTFUL_ID = {
    "t-shirt": 71, # Example: Bella + Canvas 3001
    "tshirt": 71,
    "tee": 71,
    "hoodie": 162, # Example: Gildan 18500
    "sweatshirt": 162,
    "poster": 1,
    "mug": 19,
}

# --- ADDED: Checkout Readiness Field Definitions ---
# Defines the *minimum* set of keys required in draft_order_details
# for the session.state to be set to 'ready_for_checkout'.
CHECKOUT_REQUIRED_FIELDS = [
    'artwork_image_url',   # Design URL
    'product_input',       # "t-shirt"
    'placement_id',        # "front"
    'user_width_in',       # 8.0
    'user_offset_y_in',    # 3.0
    'mockup_image_url',    # URL of the final preview
    'variant_id',          # 12345 (Printful's ID)
    'size_input',          # "Large"
    'color_input',         # "Black"
    'quantity',            # 1
    'recipient_name',      # "Jane Doe"
    'recipient_address1',  # "123 Cyber Street"
    'recipient_city',      # "Neo Kyoto"
    'recipient_country_code', # "US"
    'recipient_zip',       # "90210"
    # 'recipient_state_code' is handled separately as it's conditional
]
COUNTRIES_NEEDING_STATES = {"US", "CA", "AU", "JP"}


# --- Stripe ---
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
if not STRIPE_SECRET_KEY: log.critical("STRIPE_SECRET_KEY not set."); raise RuntimeError("STRIPE_SECRET_KEY not set")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
if not STRIPE_WEBHOOK_SECRET: log.critical("STRIPE_WEBHOOK_SECRET not set."); raise RuntimeError("STRIPE_WEBHOOK_SECRET not set")
stripe.api_key = STRIPE_SECRET_KEY
STRIPE_TIMEOUT = 45 # Seconds

# --- File Storage (MODIFIED for Cloudinary) ---
# UPLOAD_DIR = "temp_images"; os.makedirs(UPLOAD_DIR, exist_ok=True) # <-- REMOVED
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
# MIME_TYPES = {...} # <-- REMOVED (Cloudinary handles this)

# NEW: Cloudinary folder paths for organization
CLOUDINARY_UPLOAD_FOLDER = "glitchape/user_uploads"
CLOUDINARY_ARTWORK_FOLDER = "glitchape/ai_artwork"
CLOUDINARY_MOCKUP_FOLDER = "glitchape/ai_mockups"

# --- Country/State Codes (Cache - Populate on startup or first request) ---
VALID_COUNTRIES: Dict[str, str] = {}
VALID_STATES: Dict[str, Dict[str, str]] = {}

# ===================================================================
# PYDANTIC SCHEMAS (Unchanged)
# ===================================================================
class ImagePreview(BaseModel): view_name: str; url: str
class LiveChatResponse(BaseModel): session_id: str; response_text: str; image_urls: Optional[List[str]] = None
class Recipient(BaseModel): name: str; address1: str; address2: Optional[str] = None; city: str; state_code: Optional[str] = None; country_code: str; zip: Optional[str] = None; email: Optional[str] = None; phone: Optional[str] = None
class OrderItem(BaseModel): variant_id: int; product_name: str; quantity: int = 1
class CheckoutRequest(BaseModel): session_id: str
class CheckoutResponse(BaseModel): order_id: str; payment_intent_id: str; client_secret: str; total_cost: float; currency: str

# ===================================================================
# CORE LOGIC HELPERS
# ===================================================================

# --- ADDED: Printful DPI Conversion ---
def _convert_inches_to_printful_pixels(inches: float) -> int:
    """Converts inches to Printful's 150 DPI pixel standard."""
    return round(inches * PRINTFUL_DPI)


# --- ADDED: Cache for API lookups ---
@functools.lru_cache(maxsize=128)
async def _fetch_printful_product_variants(product_id: int) -> Optional[List[Dict]]:
    """Cached fetch for product variants from Printful."""
    headers = {"Authorization": f"Bearer {PRINTFUL_API_KEY}"}
    url = f"{PRINTFUL_API_URL}/products/{product_id}"
    try:
        async with httpx.AsyncClient(timeout=PRINTFUL_TIMEOUT) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json().get("result", {})
            return data.get("variants")
    except httpx.HTTPStatusError as e:
        log.error(f"Printful API Error fetching variants for product {product_id}: {e.response.status_code} - {e.response.text}")
        return None
    except httpx.RequestError as e:
        log.error(f"Network error fetching Printful variants for product {product_id}: {e}")
        return None
    except Exception as e:
        log.error(f"Unexpected error fetching Printful variants for {product_id}: {e}", exc_info=True)
        return None

# --- MODIFIED: Variant ID Lookup (Now reads from draft_details) ---
async def _validate_and_update_variant(draft_details: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Tries to find a variant_id based on details in the draft.
    Updates draft_details with 'variant_id' if found.
    Returns (success, error_message_for_user).
    """
    product_type = draft_details.get('product_input')
    size = draft_details.get('size_input')
    color = draft_details.get('color_input')
    
    # Not enough info to try yet
    if not all([product_type, size, color]):
        return (True, None) # Not an error, just can't validate yet

    product_id = PRODUCT_TYPE_TO_PRINTFUL_ID.get(product_type.lower().strip())
    if not product_id:
        msg = f"Sorry, I don't recognize '{product_type}' as a product. Try 't-shirt' or 'hoodie'."
        log.warning(f"Variant Validation: No product_id for {product_type}")
        return (False, msg)

    variants = await _fetch_printful_product_variants(product_id)
    if variants is None:
        msg = f"// System glitch: Could not fetch product info for {product_type}. Please try again."
        return (False, msg)
    if not variants:
        msg = f"// System glitch: No variants found for {product_type} (ID: {product_id})."
        return (False, msg)

    target_size = size.strip().lower()
    target_color = color.strip().lower()

    for variant in variants:
        variant_size = (variant.get("size") or "").strip().lower()
        variant_color = (variant.get("color") or "").strip().lower()

        if variant_size == target_size and variant_color == target_color:
            log.info(f"Exact match found for {product_type}/{size}/{color}: Variant ID {variant['id']}")
            draft_details['variant_id'] = variant["id"]
            return (True, None)

    fuzzy_target_color = re.sub(r'\b(heather|triblend)\b', '', target_color).strip()
    for variant in variants:
        variant_size = (variant.get("size") or "").strip().lower()
        variant_color = (variant.get("color") or "").strip().lower()
        fuzzy_variant_color = re.sub(r'\b(heather|triblend)\b', '', variant_color).strip()

        if variant_size == target_size and fuzzy_variant_color == fuzzy_target_color:
            log.info(f"Fuzzy match found for {product_type}/{size}/{color}: Variant ID {variant['id']} (Matched on '{fuzzy_variant_color}')")
            draft_details['variant_id'] = variant["id"]
            return (True, None)

    log.warning(f"No matching variant found for Product ID {product_id} ({product_type}) with size='{size}' and color='{color}'.")
    # Clear any old/invalid ID
    draft_details.pop('variant_id', None)
    msg = f"Hmm, I couldn't find a '{size} {color} {product_type}' in the catalog. Can you double-check the spelling or try a different combination? (e.g., 'Medium Black', 'Large White')"
    return (False, msg)


# --- ADDED: Country/State Validation ---
async def _populate_location_data():
    """Fetches and caches country and state data from Printful."""
    global VALID_COUNTRIES, VALID_STATES
    if VALID_COUNTRIES: return

    headers = {"Authorization": f"Bearer {PRINTFUL_API_KEY}"}
    url = f"{PRINTFUL_API_URL}/countries"
    try:
        async with httpx.AsyncClient(timeout=PRINTFUL_TIMEOUT) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            countries = resp.json().get("result", [])
            temp_countries = {}
            temp_states = {}
            for country in countries:
                code = country.get("code")
                name = country.get("name")
                if code and name:
                    temp_countries[code] = name
                    if code in COUNTRIES_NEEDING_STATES and country.get("states"):
                        temp_states[code] = {state["code"]: state["name"] for state in country["states"] if state.get("code") and state.get("name")}
            VALID_COUNTRIES = temp_countries
            VALID_STATES = temp_states
            log.info(f"Populated location data: {len(VALID_COUNTRIES)} countries, {len(VALID_STATES)} countries with states.")
    except httpx.HTTPStatusError as e:
        log.error(f"Printful API Error fetching countries: {e.response.status_code} - {e.response.text}")
    except httpx.RequestError as e:
        log.error(f"Network error fetching Printful countries: {e}")
    except Exception as e:
        log.error(f"Unexpected error populating location data: {e}", exc_info=True)

def _validate_country_code(code: str) -> Optional[str]:
    """Validates country code against cached list."""
    if not VALID_COUNTRIES:
         log.warning("Country list not populated yet. Validation might be incomplete.")

    code_upper = code.strip().upper()
    if code_upper in VALID_COUNTRIES:
        return code_upper
    for c_code, c_name in VALID_COUNTRIES.items():
        if code.strip().lower() == c_name.lower():
            return c_code
    return None

def _validate_state_code(country_code: str, state_input: Optional[str]) -> Optional[str]:
    """Validates state code based on country, if required."""
    if country_code not in COUNTRIES_NEEDING_STATES:
        return state_input

    if not state_input:
         log.warning(f"State code is required for country {country_code} but was not provided.")
         return None

    if not VALID_STATES.get(country_code):
         log.warning(f"State list for country {country_code} not populated. Validation incomplete.")
         return state_input.strip().upper()

    state_code_upper = state_input.strip().upper()
    valid_state_codes = VALID_STATES.get(country_code, {})
    if state_code_upper in valid_state_codes:
        return state_code_upper
    for s_code, s_name in valid_state_codes.items():
        if state_input.strip().lower() == s_name.lower():
            return s_code
            
    log.warning(f"Invalid state '{state_input}' provided for country {country_code}.")
    return None


# --- REMOVED: _cleanup_expired_images ---
# This function is no longer needed as Cloudinary manages assets.


# --- MODIFIED: _upload_to_cloudinary (Returns URL dict only) ---
async def _upload_to_cloudinary(
    file_source: UploadFile | BytesIO | str | bytes,
    folder: str,
    public_id_prefix: Optional[str] = None
) -> Dict[str, str]:
    """
    Uploads a file to Cloudinary and returns its secure URL.
    """
    if public_id_prefix:
        public_id = f"{public_id_prefix}_{uuid.uuid4()}"
    else:
        public_id = f"{uuid.uuid4()}"

    data_to_pass: str | BytesIO  # The object to be passed to the sync thread


    # --- MODIFIED: Check against both FastAPI's UploadFile and Starlette's UploadFile ---
    if isinstance(file_source, (UploadFile, StarletteUploadFile)):
        # 1. This 'await' is critical to convert the async file to raw bytes
        file_content = await file_source.read()
        if len(file_content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large (max {MAX_FILE_SIZE_MB}MB)."
            )
        
        # 3. Wrap the raw bytes in a synchronous file-like object
        data_to_pass = BytesIO(file_content)

    # Handle raw bytes (from AI generation)
    elif isinstance(file_source, bytes):
        data_to_pass = BytesIO(file_source)
    
    # Handle str (URL/path) or existing BytesIO
    elif isinstance(file_source, (str, BytesIO)):
        data_to_pass = file_source
    
    # Safety fallback for unexpected types
    else:
        log.error(f"Unexpected file_source type in _upload_to_cloudinary: {type(file_source)}")
        raise HTTPException(status_code=500, detail="Internal server error: Invalid file type for upload.")

    try:
        # Use asyncio.to_thread to run the synchronous Cloudinary upload
        
        # Define sync_upload to accept data as an argument
        def sync_upload(upload_data: str | BytesIO):
            return cloudinary.uploader.upload(
                upload_data, # This is now correctly passing BytesIO or str
                folder=folder,
                public_id=public_id,
                resource_type="auto",
                overwrite=True,
                unique_filename=False # We use UUIDs for uniqueness
            )
        
        # Pass data_to_pass as an argument to the thread
        upload_result = await asyncio.to_thread(sync_upload, data_to_pass)
        
        # --- PRODUCTION FIX: Added fallback for missing 'secure_url' ---
        secure_url = upload_result.get("secure_url")
        
        if not secure_url:
            log.warning(f"Cloudinary response missing 'secure_url'. Attempting to construct URL. Result: {upload_result}")
            
            public_id = upload_result.get("public_id")
            resource_type = upload_result.get("resource_type", "image")
            
            if public_id:
                # Use the Cloudinary utility function to guarantee an HTTPS URL
                # cloudinary_url returns a tuple, so we take the first element [0]
                secure_url = cloudinary.utils.cloudinary_url(
                    public_id,
                    resource_type=resource_type,
                    version=upload_result.get("version"),
                    secure=True
                )[0]
                log.info(f"Successfully constructed secure_url using fallback: {secure_url}")
            else:
                log.error(f"Cloudinary upload failed: No 'secure_url' or 'public_id' in response. Result: {upload_result}")
                raise HTTPException(status_code=500, detail="Cloudinary upload failed: No URL or public_id returned.")
        # --- END PRODUCTION FIX ---

        log.info(f"File uploaded to Cloudinary: {secure_url}")
        
        # Return URL dict
        return {"url": secure_url}

    except HTTPException:
        raise # Re-raise our own 413 error
    except Exception as e:
        log.error(f"Error uploading file to Cloudinary: {e}", exc_info=True)
        # Check for Cloudinary-specific errors
        if "File size too large" in str(e):
             raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large (max {MAX_FILE_SIZE_MB}MB)."
            )
        # Check for the coroutine error specifically
        if "a bytes-like object is required, not 'coroutine'" in str(e):
            log.error(f"Critical upload error: Coroutine passed to sync function. Type was: {type(data_to_pass)}")
            raise HTTPException(status_code=500, detail="Internal error during file upload (sync conflict).")
            
        raise HTTPException(status_code=500, detail=f"Could not upload file: {str(e)}")


# --- REMOVED: _analyze_image_with_vision_model ---
# This function is no longer called as we use OpenRouter for multimodal.


# --- MODIFIED: _generate_design_artwork ---
async def _generate_design_artwork(prompt: str) -> str:
    """Generates standalone artwork decal, uploads to Cloudinary, returns URL."""
    full_prompt = f"Generate a single, high-resolution, print-quality **standalone graphic or logo** of: '{prompt}'. Centered. **Transparent background (PNG format)**. No clothing, models, or text."
    payload = {"inputs": full_prompt}
    try:
        async with httpx.AsyncClient(timeout=HF_TIMEOUT) as client:
            response = await client.post(HF_API_URL, headers=HF_HEADERS, json=payload)
            response.raise_for_status()
        image_data = response.content
        
        # Validate image data in memory
        try:
            img = Image.open(BytesIO(image_data))
            img.verify()
        except Exception:
            log.error("HF AI returned invalid image data.");
            raise HTTPException(500, "AI returned invalid image data.")

        # Upload bytes to Cloudinary.
        # This call only needs the URL, so we extract it from the dict.
        image_url = (await _upload_to_cloudinary(
            file_source=image_data,
            folder=CLOUDINARY_ARTWORK_FOLDER
        ))["url"]
        
        log.info(f"Generated artwork uploaded to {image_url}")
        return image_url

    except httpx.HTTPStatusError as e:
        log.error(f"HuggingFace API Error (Artwork): {e.response.status_code} - {e.response.text[:200]}")
        detail = f"AI Service Error ({e.response.status_code})"
        if e.response.status_code == 503: detail = "AI model loading. Try again shortly."
        raise HTTPException(status_code=502, detail=detail)
    except httpx.RequestError as e:
        log.error(f"Network error generating HF artwork: {e}")
        raise HTTPException(status_code=504, detail="AI Service Network Error.")
    except HTTPException: # Re-raise exceptions from _upload_to_cloudinary
        raise
    except Exception as e:
        log.error(f"Unhandled error generating HF artwork: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error during AI artwork generation.")

# --- REWRITTEN: _generate_mockup_preview (Uses Printful) ---
async def _generate_mockup_preview(
    product_id: int,
    design_url: str,
    placement_id: str, # e.g., 'front', 'back'
    width_in: float,
    offset_y_in: float
) -> str:
    """
    Generates a precise mockup using Printful's Mockup Generator API.
    Uploads final mockup to Cloudinary and returns URL.
    """
    log.info(f"Generating Printful mockup for prod {product_id}, w_in {width_in}, y_in {offset_y_in}")
    
    # 1. Convert inches to pixels
    width_px = _convert_inches_to_printful_pixels(width_in)
    offset_y_px = _convert_inches_to_printful_pixels(offset_y_in)

    headers = {"Authorization": f"Bearer {PRINTFUL_API_KEY}"}

    # 2. Define the V2 file object structure
    # This specifies the decal, its placement, and its exact transforms
    file_option = {
        "id": placement_id, # The print area (e.g., 'front')
        "type": placement_id,
        "url": design_url,
        "options": {
            "width": width_px,
            "offset_y": offset_y_px,
            # We can add 'offset_x' later if needed, but keeping it
            # simple (centered) for now by omitting it.
        }
    }

    # 3. Create the mockup task payload
    payload = {
        "variant_ids": [product_id], # Use the *specific* product ID
        "files": [file_option]
    }
    
    async with httpx.AsyncClient(timeout=PRINTFUL_TIMEOUT) as client:
        try:
            # 4. Create the task
            resp_create = await client.post(PRINTFUL_MOCKUP_URL, headers=headers, json=payload)
            resp_create.raise_for_status()
            task_data = resp_create.json()
            task_key = task_data.get("result", {}).get("task_key")
            if not task_key:
                log.error(f"Printful mockup API did not return task_key. Payload: {payload}, Resp: {resp_create.text}")
                raise HTTPException(502, "Printful Mockup API failed (task key missing).")
            
            log.info(f"Printful mockup task created: {task_key}")
            task_url = f"{PRINTFUL_MOCKUP_URL}/{task_key}"
            
            # 5. Poll for completion
            start_time = datetime.now()
            max_wait = timedelta(seconds=HF_TIMEOUT) # Reuse HF_TIMEOUT as a reasonable max
            
            while datetime.now() - start_time < max_wait:
                await asyncio.sleep(2) # Poll every 2 seconds
                resp_poll = await client.get(task_url, headers=headers)
                
                if resp_poll.status_code == 200:
                    poll_data = resp_poll.json().get("result", {})
                    status = poll_data.get("status")
                    
                    if status == "completed":
                        mockup_url = poll_data.get("mockups", [{}])[0].get("mockup_url")
                        if not mockup_url:
                            log.error(f"Printful task {task_key} completed but no mockup_url. Data: {poll_data}")
                            raise HTTPException(502, "Printful Mockup API failed (URL missing).")
                        
                        log.info(f"Printful mockup completed. URL: {mockup_url}")
                        
                        # 6. Upload Printful's URL to *our* Cloudinary for persistence
                        # This is crucial. Printful URLs might expire.
                        # We are passing a URL (str) to _upload_to_cloudinary.
                        final_cloudinary_url = (await _upload_to_cloudinary(
                            file_source=mockup_url,
                            folder=CLOUDINARY_MOCKUP_FOLDER
                        ))["url"] # Extract URL from dict
                        
                        log.info(f"Printful mockup {mockup_url} persisted to {final_cloudinary_url}")
                        return final_cloudinary_url

                    elif status == "failed":
                        log.error(f"Printful mockup task {task_key} failed. Data: {poll_data}")
                        raise HTTPException(502, "Printful Mockup API task failed.")
                    
                    # if status is 'pending' or 'in_progress', loop continues
                else:
                    log.warning(f"Printful mockup poll status {resp_poll.status_code}. Retrying.")
            
            # 6. Handle Timeout
            log.error(f"Printful mockup task {task_key} timed out.")
            raise HTTPException(504, "Printful Mockup API timed out.")

        except httpx.HTTPStatusError as e:
            log.error(f"Printful Mockup API HTTP Error: {e.response.status_code} - {e.response.text[:200]}")
            raise HTTPException(status_code=502, detail=f"Printful Mockup Service Error ({e.response.status_code}).")
        except httpx.RequestError as e:
            log.error(f"Network error in _generate_mockup_preview: {e}")
            raise HTTPException(status_code=504, detail="Printful Mockup Service Network Error.")
        except HTTPException:
            raise # Re-raise our own errors
        except Exception as e:
            log.error(f"Unhandled error in _generate_mockup_preview: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal error during Printful mockup generation.")


# --- Order Logic Helpers ---
async def _get_design_images_from_session(session_id: uuid.UUID, user_id: uuid.UUID, db: AsyncSession) -> List[ChatMessage]:
    """Finds the *latest* AI-generated/uploaded **design artwork** message."""
    # (Implementation unchanged)
    q = select(ChatMessage).where(ChatMessage.session_id == session_id, ChatMessage.user_id == user_id, ChatMessage.role == "ai", ChatMessage.image_url.isnot(None), ChatMessage.content.like("%design artwork:%")).order_by(ChatMessage.created_at.desc()).limit(1)
    r = await db.execute(q); artwork_msg = r.scalars().first()
    if not artwork_msg: raise HTTPException(status_code=404, detail="No valid design artwork found in this session for ordering.")
    return [artwork_msg]

# --- MODIFIED: _upload_to_printful ---
async def _upload_to_printful(image_msg: ChatMessage, headers: Dict[str, str]) -> Dict[str, Any]:
    """Uploads single design artwork file from a public URL to Printful."""
    
    # The image URL is now a public Cloudinary URL
    image_url = image_msg.image_url
    if not image_url: 
        raise HTTPException(500, f"DB record missing URL for image {image_msg.id}.")
    
    if not image_url.startswith("http"):
        raise HTTPException(500, f"Invalid image URL format in DB: {image_url}")

    view_name = "main_design_decal"
    # Printful can accept a public URL directly
    payload = {
        "url": image_url,
        "filename": f"design_{image_msg.id}.png" # Give it a unique filename for Printful
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(f"{PRINTFUL_API_URL}/files", headers=headers, json=payload)
            resp.raise_for_status()
        
        data = resp.json().get("result")
        if not data or not data.get("id"): 
            log.error(f"Printful upload no file ID. Resp: {resp.text}")
            raise HTTPException(502, "Printful API did not return File ID.")
        
        log.info(f"Uploaded {image_url} to Printful, ID: {data['id']}")
        return {"view": view_name, "id": data["id"]}

    except httpx.HTTPStatusError as e:
        log.error(f"Printful file upload HTTP error: {e.response.status_code} - {e.response.text[:200]}")
        raise HTTPException(status_code=502, detail=f"Printful API Error ({e.response.status_code}) during file upload.")
    except httpx.RequestError as e:
        log.error(f"Network error uploading to Printful: {e}")
        raise HTTPException(status_code=504, detail="Network error during Printful file upload.")
    except Exception as e:
        log.error(f"Unexpected error uploading {image_url} to Printful: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error during Printful file upload.")


async def _get_printful_costs(item: OrderItem, recipient: Recipient, headers: Dict[str, str]) -> Dict[str, Any]:
    """Gets item and shipping costs from Printful."""
    # (Implementation unchanged)
    try:
        async with httpx.AsyncClient(timeout=PRINTFUL_TIMEOUT) as client:
            prod_resp = await client.get(f"{PRINTFUL_API_URL}/products/variant/{item.variant_id}", headers=headers)
            prod_resp.raise_for_status(); prod_data = prod_resp.json().get("result", {})
            item_cost = float(prod_data.get("price", 0)) * item.quantity; currency = prod_data.get("currency", "usd").lower()
            
            shipping_payload = {"recipient": recipient.model_dump(exclude_none=True), "items": [{"variant_id": item.variant_id, "quantity": item.quantity}]}
            ship_resp = await client.post(f"{PRINTFUL_API_URL}/shipping/rates", headers=headers, json=shipping_payload)
            ship_resp.raise_for_status(); rate_data = ship_resp.json().get("result", [])
            
            if not rate_data: log.warning(f"No shipping rates returned for {item.variant_id} to {recipient.country_code}. Using fallback."); shipping_cost = 5.0
            else: shipping_cost = float(rate_data[0].get("rate", 0))

            total = item_cost + shipping_cost
            if total <= 0: raise ValueError("Calculated total cost is zero or negative.")
            log.info(f"Calculated Printful cost: {total:.2f} {currency.upper()}")
            return {"total_cents": int(total * 100), "currency": currency}
    except httpx.HTTPStatusError as e:
        log.error(f"Printful cost calc HTTP error: {e.response.status_code} - {e.response.text[:200]}")
        detail = f"Printful API Error ({e.response.status_code})"
        if "shipping is disabled" in e.response.text.lower(): detail = "Shipping to the specified region is unavailable."
        elif "variant not found" in e.response.text.lower(): detail = "Selected product variant is invalid."
        raise HTTPException(status_code=502, detail=detail)
    except (httpx.RequestError, ValueError, Exception) as e:
        log.error(f"Failed to calculate Printful costs: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"Failed to calculate Printful costs: {str(e)[:100]}")


async def _submit_order_to_printful(payment: OrderPayment, order: OrderRecord, headers: Dict[str, str], db: AsyncSession):
    """Submits final, confirmed order to Printful."""
    # (Implementation unchanged)
    try:
        recipient = json.loads(payment.recipient_json)
        file_id_map = json.loads(payment.printful_file_ids_json)
        artwork_printful_id = list(file_id_map.values())[0]

        printful_files = [{"id": artwork_printful_id, "type": "front"}]

        order_item = {"variant_id": payment.variant_id, "quantity": 1, "files": printful_files, "name": order.product_name, "external_id": f"GLA-ITEM-{order.id}"}
        payload = {"recipient": recipient, "items": [order_item], "external_id": f"GLA-{order.id}"}

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(f"{PRINTFUL_API_URL}/orders", headers=headers, json=payload)
            resp.raise_for_status()
            draft_order = resp.json().get("result"); printful_order_id = draft_order.get("id")
            if not printful_order_id: raise ValueError("Printful did not return order ID after creation.")
            log.info(f"Created Printful draft order {printful_order_id} for internal order {order.id}")

            confirm_resp = await client.post(f"{PRINTFUL_API_URL}/orders/{printful_order_id}/confirm", headers=headers)
            confirm_resp.raise_for_status()
            confirmed_order = confirm_resp.json().get("result")
            if not confirmed_order or not confirmed_order.get("id"): raise ValueError("Printful did not return order details after confirmation.")
            
            order.printful_order_id = str(confirmed_order["id"])
            payment.status = "submitted_to_printful"
            await db.commit()
            log.info(f"Successfully submitted order {order.id} to Printful (Printful ID: {order.printful_order_id})")

    except (httpx.HTTPStatusError, httpx.RequestError, ValueError, json.JSONDecodeError, Exception) as e:
        err_msg = f"Printful submission failed for order {payment.order_id}: {str(e)[:200]}"
        if isinstance(e, httpx.HTTPStatusError): err_msg = f"Printful API Error ({e.response.status_code}) submitting order {payment.order_id}: {e.response.text[:150]}"
        elif isinstance(e, httpx.RequestError): err_msg = f"Network error submitting order {payment.order_id} to Printful: {e}"
        log.error(err_msg, exc_info=not isinstance(e, ValueError))
        payment.status = "error"; payment.error_message = err_msg; await db.commit()


def _parse_state_zip(state_zip_input: str) -> Dict[str, Optional[str]]:
    """Tries to extract state code and zip code."""
    # (Implementation unchanged)
    state_code = None; zip_code = None
    parts = re.split(r'[\s,]+', state_zip_input.strip())
    if len(parts) >= 2:
        part1, part_last = parts[0], parts[-1]
        if part1.isalpha() and len(part1) <= 3: state_code = part1.upper(); zip_code = " ".join(parts[1:])
        elif part_last.isalpha() and len(part_last) <= 3: state_code = part_last.upper(); zip_code = " ".join(parts[:-1])
        elif part1.isdigit(): zip_code = part1; state_code = " ".join(parts[1:]).upper()
        elif part_last.isdigit(): zip_code = part_last; state_code = " ".join(parts[:-1]).upper()
        else: state_code = part1.upper(); zip_code = " ".join(parts[1:])
    elif len(parts) == 1:
        if parts[0].isdigit(): zip_code = parts[0]
        else: state_code = parts[0].upper()
    if zip_code: zip_code = re.sub(r'[^\w\s-]', '', zip_code).strip()
    if state_code: state_code = re.sub(r'[^\w\s-]', '', state_code).strip()
    return {"state_code": state_code, "zip_code": zip_code}

# ===================================================================
# GLITCHAPE CENTRAL HANDLER CLASS
# ===================================================================

class GlitchApeCentralHandler:
    """Orchestrates AI, orders, payments, including conversational data collection."""

    # --- NEW: Load variant_map.json on class level ---
    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _load_variant_map():
        """Loads and caches the variant map from JSON."""
        try:
            with open("variant_map.json", "r") as f:
                data = json.load(f)
            # Convert list to a dict keyed by variant_id for O(1) lookup
            variant_map_dict = {item['variant_id']: item for item in data}
            log.info(f"Successfully loaded and cached {len(variant_map_dict)} variants from variant_map.json.")
            return variant_map_dict
        except FileNotFoundError:
            log.error("CRITICAL: variant_map.json not found. Product image lookup will fail.")
            return {}
        except Exception as e:
            log.error(f"Failed to load or parse variant_map.json: {e}")
            return {}

    def __init__(self, db: AsyncSession, user: Optional[User] = None):
        self.db = db
        self.user = user
        # --- NEW: Assign the cached map to the instance ---
        self.variant_map = self._load_variant_map()
        # This is a non-blocking task to populate the cache if it's empty.
        # The function itself checks if the cache is already populated.
        asyncio.create_task(_populate_location_data())

    # --- NEW: Helper to get image URL from the loaded map ---
    def _get_product_image_url(self, variant_id: int) -> Optional[str]:
        """Looks up the product image URL from the cached variant map."""
        if not self.variant_map:
            log.warning("Variant map is not loaded. Cannot fetch product image.")
            return None
        
        variant_data = self.variant_map.get(variant_id)
        if variant_data:
            image_url = variant_data.get('product_image')
            if image_url:
                # Clean up escaped slashes from the JSON file
                return image_url.replace(r'\/', r'/')
        
        log.warning(f"No product_image found in variant_map.json for variant_id: {variant_id}")
        return None

    # --- ADDED: Memory Context Helpers ---
    async def _get_memory_context(self, session: ChatSession) -> Tuple[Dict[str, Any], str]:
        """Gets draft details (memory) and calculates missing info for the LLM."""
        draft_details = session.draft_order_details if isinstance(session.draft_order_details, dict) else {}
        
        missing_fields = []
        
        # --- MODIFIED: Don't list missing info if user is confirming or ready ---
        if session.state not in ("awaiting_confirmation", "ready_for_checkout"):
            if not draft_details.get('artwork_image_url'):
                missing_fields.append("artwork (must be generated via text prompt)")
            
            if not draft_details.get('product_input'):
                missing_fields.append("product_input (e.g., t-shirt, hoodie)")
            elif not draft_details.get('placement_id'):
                missing_fields.append("placement_id (e.g., front, back)")
            elif not draft_details.get('user_width_in') or not draft_details.get('user_offset_y_in'):
                missing_fields.append("placement_position (width and offset_y in inches)")
            elif not draft_details.get('mockup_image_url'):
                missing_fields.append("mockup (trigger by providing placement)")
            
            if not draft_details.get('size_input') or not draft_details.get('color_input'):
                missing_fields.append("size_and_color (e.g., 'Large Black')")
            elif not draft_details.get('variant_id'):
                missing_fields.append("variant_validation (confirm size/color)")
                
            if not draft_details.get('quantity'):
                missing_fields.append("quantity")
            elif not draft_details.get('recipient_name'):
                missing_fields.append("recipient_name")
            elif not draft_details.get('recipient_address1'):
                missing_fields.append("recipient_address1")
            elif not draft_details.get('recipient_city'):
                missing_fields.append("recipient_city")
            elif not draft_details.get('recipient_country_code'):
                missing_fields.append("recipient_country_code")
            elif not draft_details.get('recipient_zip'):
                missing_fields.append("recipient_zip_and_state")
             
        missing_str = ", ".join(missing_fields)
        return (draft_details, missing_str)

    async def _check_and_set_checkout_ready(self, session: ChatSession, draft_details: Dict[str, Any]) -> bool:
        """
        Checks if all required data is in memory, and updates session state if so.
        This function is now DEPRECATED in the chat loop, but still used
        by _get_memory_context to know when to stop listing missing fields.
        """
        if session.state == "ready_for_checkout":
            return True # Already ready
            
        for key in CHECKOUT_REQUIRED_FIELDS:
            if not draft_details.get(key):
                # log.info(f"Checkout readiness check failed: Missing key '{key}'")
                return False # Missing a required key
        
        # Special check for state
        country_code = draft_details.get('recipient_country_code')
        if country_code in COUNTRIES_NEEDING_STATES and not draft_details.get('recipient_state_code'):
            # log.info(f"Checkout readiness check failed: Missing state_code for {country_code}")
            return False
        
        # All checks passed!
        log.info(f"Session {session.id} ALL DATA COMPILED. Setting state to 'ready_for_checkout'")
        session.state = "ready_for_checkout"
        return True

    async def _get_chat_history_for_context(self, session: ChatSession, memory_json: str, missing_info: str, limit: int = 10) -> List[Dict[str, str]]:
        """Fetches history, adds current memory and missing info."""
        q = select(ChatMessage).where(ChatMessage.session_id == session.id).order_by(ChatMessage.created_at.desc()).limit(limit)
        r = await self.db.execute(q); messages = r.scalars().all()
        
        # --- MODIFIED: Use Memory-driven context ---
        context_lines = [
            f"[MEMORY: {memory_json}]",
            f"[MISSING_INFO: {missing_info}]"
        ]
        # This history is now passed to _call_openrouter_brain, which prepends the system prompt
        llama_history = []

        for msg in reversed(messages):
            api_role = "assistant" if msg.role == "ai" else msg.role
            content = msg.content
            
            # --- PRODUCTION FIX (Resolves LLaMA 400 'oneOf' Error) ---
            # We only append text content here. The image payload will be
            # constructed in handle_chat_message for the *last* user message.
            llama_history.append({"role": api_role, "content": content})
            # --- END FIX ---
            
        # Prepend the system context as the first message *before* the history
        system_context = {"role": "system", "content": "\n".join(context_lines)}
        
        # Note: _call_openrouter_brain will prepend the main system prompt
        # --- FIX: Prepend the dynamic system_context to the history ---
        return [system_context] + llama_history

    # --- REWRITTEN: _call_openrouter_brain (for Mixed-Mode Response) ---
    async def _call_openrouter_brain(self, history: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """
        Calls OpenRouter LLaMA/Gemini.
        Parses a mixed-mode response (natural language + JSON block) and
        returns (natural_language: str, parsed_json: dict).
        """
        
        headers = { 
            "Authorization": f"Bearer {LLAMA_API_KEY}", 
            "Content-Type": "application/json", 
            "HTTP-Referer": LLAMA_HTTP_REFERER, 
            "X-Title": "GlitchApe"
        }
        
        # Construct the messages array
        messages = [{"role": "system", "content": LLAMA_SYSTEM_PROMPT}] + history
        
        payload = { "model": LLAMA_MODEL, "messages": messages }
        
        try:
            async with httpx.AsyncClient(timeout=LLAMA_TIMEOUT) as client:
                response = await client.post(LLAMA_API_URL, headers=headers, json=payload)
                response.raise_for_status()
            data = response.json()
            llama_response_content = data.get("choices", [{}])[0].get("message", {}).get("content")
            
            if not llama_response_content: 
                log.error(f"LLaMA returned empty content. Data: {data}")
                raise HTTPException(502, "LLaMA returned empty content.")

            # --- ######################################################## ---
            # --- NEW MIXED-MODE PARSING LOGIC                           ---
            # --- ######################################################## ---
            
            # 1. Search for the ```json ... ``` block
            json_match_markdown = re.search(r'```json\s*(\{.*?\})\s*```', llama_response_content, re.DOTALL)
            
            if json_match_markdown:
                # 2. Extract natural language (everything BEFORE the block)
                natural_language = llama_response_content[:json_match_markdown.start()].strip()
                # 3. Extract the JSON string
                json_str = json_match_markdown.group(1).strip()
                
                # Fallback if AI *only* sends JSON block
                if not natural_language:
                    natural_language = "Got it, processing..."

                # 4. Try to parse the extracted JSON
                try:
                    parsed_json = json.loads(json_str)
                    if "intent" not in parsed_json:
                        log.warning(f"AI JSON block missing 'intent'. Defaulting to general_chat. JSON: {json_str}")
                        parsed_json["intent"] = "general_chat"
                    return (natural_language, parsed_json)
                except json.JSONDecodeError as e:
                    # 5. Handle JSON block being invalid
                    log.error(f"OpenRouter invalid JSON detected in block. Err: {e}. Str: {json_str}. Full Raw: {llama_response_content}")
                    return ("My processor returned a corrupted data packet. Please try that again.", {"intent": "general_chat"})
            
            else:
                # 6. No JSON block found. Treat entire response as general chat.
                log.warning(f"OpenRouter response missing JSON block. Treating as general_chat. Raw: {llama_response_content}")
                natural_language = llama_response_content.strip()
                # Ensure a default "intent" is always present
                return (natural_language, {"intent": "general_chat"})
            # --- ######################################################## ---
            # --- END OF MIXED-MODE PARSING LOGIC                        ---
            # --- ######################################################## ---

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                log.error(f"LLaMA API Error 403 (Forbidden). Check API Key and HTTP-Referer ('{LLAMA_HTTP_REFERER}'). Response: {e.response.text[:200]}")
            else:
                log.error(f"LLaMA API Error: {e.response.status_code} - {e.response.text[:200]}")
            raise HTTPException(status_code=502, detail=f"LLaMA Service Error ({e.response.status_code}).")
        except httpx.RequestError as e:
            log.error(f"Network error calling LLaMA: {e}")
            raise HTTPException(status_code=504, detail="LLaMA Service Network Error.")
        except (IndexError, KeyError, Exception) as e:
             log.error(f"Error processing LLaMA response: {e}", exc_info=True)
             raise HTTPException(status_code=502, detail=f"Error processing LLaMA response: {str(e)[:100]}")


    # --- MODIFIED: Mockup Generation Wrapper (now checks memory) ---
    async def _check_and_trigger_mockup(self, draft_details: Dict[str, Any], force_regenerate: bool = False) -> Optional[str]:
        """
        Checks if all info for a mockup is present in memory.
        If yes, generates mockup, saves URL to draft_details, and returns URL.
        Returns None if not ready or on non-HTTP error.
        Raises HTTPException on API/HTTP failures.
        """
        try:
            # 1. Check if we have all required data
            product_input = draft_details.get('product_input')
            design_url = draft_details.get('artwork_image_url')
            placement_id = draft_details.get('placement_id')
            width_in_str = draft_details.get('user_width_in')
            offset_y_in_str = draft_details.get('user_offset_y_in')

            if not all([product_input, design_url, placement_id, width_in_str, offset_y_in_str]):
                # Not enough info to generate a mockup yet. This is not an error.
                return None
            
            # 2. Check if we've already generated this mockup
            existing_mockup_url = draft_details.get('mockup_image_url')
            if existing_mockup_url and not force_regenerate:
                return None # No need to regenerate

            # 3. Validate data
            product_id = PRODUCT_TYPE_TO_PRINTFUL_ID.get(product_input.lower().strip())
            if not product_id:
                raise ValueError(f"Invalid product type '{product_input}' for mockup.")
            
            width_in = float(width_in_str)
            offset_y_in = float(offset_y_in_str)
            if not (0 < width_in <= 20 and 0 <= offset_y_in <= 20):
                 raise ValueError("Invalid dimensions (must be 0-20 inches)")

            # 4. Call the (newly rewritten) helper
            mockup_url = await _generate_mockup_preview(
                product_id=product_id,
                design_url=design_url,
                placement_id=placement_id,
                width_in=width_in,
                offset_y_in=offset_y_in
            )
            
            # 5. Save URL to memory and return
            draft_details['mockup_image_url'] = mockup_url
            log.info(f"Mockup generated and saved to memory: {mockup_url}")
            return mockup_url

        except (ValueError, TypeError) as e:
            log.warning(f"Validation error in _check_and_trigger_mockup: {e}")
            # This is a user-facing error, but we'll let the AI handle it
            # by not generating a mockup.
            return None
        except HTTPException:
            raise # Re-raise errors from _generate_mockup_preview
        except Exception as e:
            log.error(f"Unexpected error in _check_and_trigger_mockup: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal mockup generation error.")


    # --- MAIN CHAT HANDLER (MODIFIED FOR EXPLICIT CHECKOUT FLOW) ---
    async def handle_chat_message(self, session_id: uuid.UUID, prompt: str, uploaded_image: Optional[UploadFile] = None) -> LiveChatResponse:
        """Orchestrates chat, design, and validated order data collection. Rejects image uploads."""
        if not self.user: raise HTTPException(401, "User not authenticated")
        session = await self.db.get(ChatSession, session_id)
        if not session or session.user_id != self.user.id: raise HTTPException(404, "Chat session not found or access denied")

        # --- NEW: Immediately reject if an image is attached ---
        if uploaded_image and uploaded_image.filename:
            log.warning(f"User {self.user.id} attempted to upload '{uploaded_image.filename}' to the chat endpoint. Feature is retired.")
            
            # Save the user's text prompt (which is now ignored but good for history)
            user_msg = ChatMessage(session_id=session_id, user_id=self.user.id, role="user", content=prompt) 
            self.db.add(user_msg)
            
            # Create the AI's rejection response
            ai_text_response = "Whoa there! Looks like you tried to upload an image. That feature is retired—I can only generate *new* designs from text prompts. Try describing what you want!"
            ai_msg = ChatMessage(session_id=session_id, user_id=self.user.id, role="ai", content=ai_text_response)
            self.db.add(ai_msg)
            
            await self.db.commit()
            
            return LiveChatResponse(session_id=str(session_id), response_text=ai_text_response, image_urls=None)
        # --- END OF UPLOAD REJECTION ---

        # --- Local variables ---
        ai_text_response: str = "" # Will be populated by _call_openrouter_brain
        action_json: Dict[str, Any] = {} # Will be populated by _call_openrouter_brain
        image_previews: List[ImagePreview] = []
        new_mockup_url: Optional[str] = None
        
        # 1. Get Memory & Missing Info
        draft_details, missing_fields = await self._get_memory_context(session)

        # 2. Handle Upload (REMOVED)
        # The logic block for processing uploads (Cloudinary, Base64) is gone.
        
        # 3. Prepare LLaMA prompt
        final_prompt_for_llama = prompt

        # 4. Save User Message (No image_url)
        user_msg = ChatMessage(session_id=session_id, user_id=self.user.id, role="user", content=prompt)
        self.db.add(user_msg); await self.db.flush()

        # 5. Get Context & Call AI
        memory_json = json.dumps(draft_details)
        history = await self._get_chat_history_for_context(session, memory_json, missing_fields)
        
        try:
            # --- MODIFIED: Call now returns (natural_language, parsed_json) ---
            (ai_text_response, action_json) = await self._call_openrouter_brain(history)
        except HTTPException as e:
            ai_text_response = f"My core processor (LLaMA/Gemini) had an issue ({e.detail}). Please try again."
            ai_msg = ChatMessage(session_id=session_id, user_id=self.user.id, role="ai", content=ai_text_response)
            self.db.add(ai_msg); await self.db.commit()
            return LiveChatResponse(session_id=str(session_id), response_text=ai_text_response)

        # 6. --- DYNAMIC INTENT ROUTING ---
        
        intent = action_json.get("intent")
        validation_clarification: Optional[str] = None

        try:
            if intent == "general_chat":
                # AI handles general chat, including telling user uploads are disabled.
                # The `ai_text_response` is already set.
                pass 

            elif intent == "design_request" or intent == "design_revision_artwork":
                design_prompt = action_json.get("design_prompt") or action_json.get("revision_prompt")
                if not design_prompt:
                    # AI failed to extract a prompt.
                    ai_text_response = "Need a clearer design prompt, cybernaut! Describe what you envision."
                else:
                    artwork_url = await _generate_design_artwork(design_prompt)
                    draft_details['artwork_image_url'] = artwork_url
                    # Clear old mockup
                    draft_details.pop('mockup_image_url', None) 
                    
                    artwork_content = f"Generated design artwork: {design_prompt}"
                    ai_artwork_msg = ChatMessage(session_id=session_id, user_id=self.user.id, role="ai", content=artwork_content, image_url=artwork_url)
                    self.db.add(ai_artwork_msg)
                    image_previews.append(ImagePreview(view_name="Artwork Decal", url=artwork_url))

            # --- REMOVED: 'image_analysis_confirm' intent ---

            elif intent == "collect_order_details":
                updated_memory = action_json.get("updated_memory", {})
                if not updated_memory:
                    log.warning(f"Session {session_id}: 'collect_order_details' intent had no updated_memory.")
                
                # Check what *kind* of data we just received
                placement_keys_updated = any(k in updated_memory for k in ['user_width_in', 'user_offset_y_in', 'placement_id', 'product_input'])
                variant_keys_updated = any(k in updated_memory for k in ['size_input', 'color_input', 'product_input'])
                
                # Update the memory
                draft_details.update(updated_memory)

                # --- Run dynamic validations/triggers ---
                
                # 1. (Re)validate variant if new size/color/product info came in
                if variant_keys_updated:
                    (validation_ok, clarification_msg) = await _validate_and_update_variant(draft_details)
                    if not validation_ok:
                        validation_clarification = clarification_msg
                    
                    # --- NEW LOGIC: ADD PRODUCT IMAGE ---
                    # Check if validation was successful AND produced a new variant_id
                    new_variant_id = draft_details.get('variant_id')
                    if validation_ok and new_variant_id:
                        # Fetch the product template image
                        product_image_url = self._get_product_image_url(new_variant_id)
                        if product_image_url:
                            log.info(f"Found product template image for variant {new_variant_id}: {product_image_url}")
                            # Add this image to be displayed
                            image_previews.append(ImagePreview(view_name="Product Template", url=product_image_url))
                            # Save a message to history
                            ai_image_msg = ChatMessage(
                                session_id=session_id, 
                                user_id=self.user.id, 
                                role="ai", 
                                content=f"Product Template: {draft_details.get('product_input')} ({draft_details.get('size_input')}, {draft_details.get('color_input')})", 
                                image_url=product_image_url
                            )
                            self.db.add(ai_image_msg)
                    # --- END NEW LOGIC ---


                # 2. (Re)generate mockup if new placement info came in
                if placement_keys_updated and not validation_clarification:
                    try:
                        # Force regenerate=True if keys were just updated
                        new_mockup_url = await self._check_and_trigger_mockup(draft_details, force_regenerate=True)
                        if new_mockup_url:
                             log.info(f"Session {session_id}: Dynamic mockup generated: {new_mockup_url}")
                             image_previews.append(ImagePreview(view_name="Placement Mockup", url=new_mockup_url))
                             ai_mockup_msg = ChatMessage(session_id=session_id, user_id=self.user.id, role="ai", content=f"Mockup: {draft_details.get('product_input')} w:{draft_details.get('user_width_in')}\" y:{draft_details.get('user_offset_y_in')}\"", image_url=new_mockup_url)
                             self.db.add(ai_mockup_msg)

                    except HTTPException as e:
                        # Error from _trigger_mockup_generation
                        log.error(f"Mockup gen failed for session {session_id}: {e.detail}")
                        ai_text_response = f"// Glitch! Mockup generator failed: {e.detail}. Let's try that position again?"

            elif intent == "order_cancel":
                draft_details = {}
                session.state = "designing" # Reset state
                log.info(f"Session {session_id} order cancelled, memory cleared.")

            # --- NEW: Handle explicit summary from AI ---
            elif intent == "summarize_order":
                log.info(f"Session {session_id} received summarize_order intent. Moving to 'awaiting_confirmation'.")
                session.state = "awaiting_confirmation"
                # ai_text_response is already the summary, so just pass
                pass

            # --- NEW: Handle explicit checkout confirmation from AI ---
            elif intent == "initiate_checkout":
                if session.state != "awaiting_confirmation":
                    log.warning(f"Session {session_id} received 'initiate_checkout' in wrong state: {session.state}. Ignoring.")
                    # Don't change state, just chat
                    ai_text_response = "Hold on, let's confirm your order summary first."
                else:
                    log.info(f"Session {session_id} received 'initiate_checkout' after confirmation. Moving to 'ready_for_checkout'.")
                    session.state = "ready_for_checkout"
                    # Add to the AI's response to confirm
                    ai_text_response += "\n\nThe **Checkout** button is now active. Click it to proceed!"

            else:
                log.warning(f"Session {session_id}: Unhandled intent '{intent}' received.")

        except Exception as e:
            log.error(f"Error processing intent '{intent}': {e}", exc_info=True)
            ai_text_response = "System glitch processing that step. Please try again or type 'cancel order'."

        # 7. --- Post-Intent Processing ---
        
        # Override AI response if a validation failed
        if validation_clarification:
            ai_text_response = validation_clarification
        
        # --- REMOVED: Automatic _check_and_set_checkout_ready call ---
        # This logic is now handled by the `summarize_order` and
        # `initiate_checkout` intents from the AI.

        # 8. Update Session State and Draft Details in DB
        session.draft_order_details = draft_details
        log.info(f"Updating session {session_id}: state='{session.state}'")


        # 9. Save Final AI Response
        final_image_url = None
        if image_previews:
            # Prefer the mockup, but fall back to the last image if no mockup
            mockup_img = next((p.url for p in image_previews if "Mockup" in p.view_name), None)
            if mockup_img:
                final_image_url = mockup_img
            else:
                final_image_url = image_previews[-1].url

        ai_msg = ChatMessage(session_id=session_id, user_id=self.user.id, role="ai", content=ai_text_response, image_url=final_image_url)
        self.db.add(ai_msg)

        try:
             await self.db.commit()
        except Exception as db_err:
             log.error(f"Database commit error in handle_chat_message for session {session_id}: {db_err}", exc_info=True)
             await self.db.rollback()
             raise HTTPException(status_code=500, detail="Failed to save chat progress.")


        # 10. Return Response
        all_image_urls = [p.url for p in image_previews] if image_previews else None
        return LiveChatResponse(session_id=str(session_id), response_text=ai_text_response, image_urls=all_image_urls)


    # --- Checkout Initiation (Logic unchanged, but _upload_to_printful is different) ---
    async def initiate_checkout(self, req: CheckoutRequest) -> CheckoutResponse:
        """Reads collected data from session, validates, gets costs, creates Stripe intent."""
        if not self.user: raise HTTPException(401, "User not authenticated")
        try: session_uuid = uuid.UUID(req.session_id)
        except ValueError: raise HTTPException(400, "Invalid session_id format.")

        session = await self.db.get(ChatSession, session_uuid)
        if not session or session.user_id != self.user.id: raise HTTPException(404, "Chat session not found or access denied")
        
        # --- CRITICAL VALIDATION ---
        # This is the gatekeeper. The AI's job is to get the session into this
        # state. The user's job is to click the button that calls this endpoint.
        if session.state != "ready_for_checkout": 
            log.warning(f"Checkout attempt for session {session_uuid} in invalid state '{session.state}'")
            # --- MODIFIED: Give user helpful message from prompt ---
            if session.state == "designing":
                 detail = "Almost ready! Please complete all the order details in the chat first."
            elif session.state == "awaiting_confirmation":
                 detail = "Please confirm the order summary in the chat before proceeding."
            else:
                 detail = "Order details not fully collected. Please finish the chat flow."
            raise HTTPException(400, detail)

        draft_details = session.draft_order_details if isinstance(session.draft_order_details, dict) else {}
        log.info(f"Initiating checkout for session {session_uuid} with draft: {draft_details}")

        # --- Validate and Construct OrderItem/Recipient (Unchanged) ---
        try:
            variant_id = draft_details.get('variant_id')
            if not isinstance(variant_id, int): raise ValueError("Missing or invalid Variant ID.")
            
            quantity = draft_details.get('quantity')
            if not isinstance(quantity, int) or quantity <= 0: raise ValueError("Missing or invalid quantity.")

            prod_name = f"Custom {draft_details.get('product_input', 'Item')} ({draft_details.get('size_input', '?')} / {draft_details.get('color_input', '?')})"
            order_item = OrderItem(variant_id=variant_id, product_name=prod_name, quantity=quantity)

            recipient_data = {
                "name": draft_details.get('recipient_name'),
                "address1": draft_details.get('recipient_address1'),
                "city": draft_details.get('recipient_city'),
                "country_code": draft_details.get('recipient_country_code'),
                "state_code": draft_details.get('recipient_state_code'),
                "zip": draft_details.get('recipient_zip')
            }
            if not all([recipient_data["name"], recipient_data["address1"], recipient_data["city"], recipient_data["country_code"]]):
                 raise ValueError("Missing required recipient details (name, address1, city, country).")
            if recipient_data["country_code"] in COUNTRIES_NEEDING_STATES and not recipient_data["state_code"]:
                 raise ValueError(f"State/Province code is required for country {recipient_data['country_code']}.")

            recipient = Recipient(**recipient_data)
            
            # --- ADDED: Final validation for all required fields ---
            if not all([
                draft_details.get('artwork_image_url'),
                draft_details.get('placement_id'),
                draft_details.get('user_width_in'),
                draft_details.get('user_offset_y_in')
            ]):
                 raise ValueError("Missing placement or artwork details.")


        except (ValidationError, ValueError, KeyError) as e:
            log.error(f"Checkout validation failed for session {session_uuid}: {e}. Draft: {draft_details}")
            # --- MODIFIED: Reset state to fix data ---
            session.state = "designing" # Go back to 'designing' (memory is preserved)
            await self.db.commit()
            raise HTTPException(status_code=400, detail=f"Invalid or incomplete order data: {e}. Please review details in chat.")

        # --- Proceed with Printful/Stripe (Unchanged) ---
        printful_headers = {"Authorization": f"Bearer {PRINTFUL_API_KEY}"}
        artwork_url = draft_details.get('artwork_image_url')
        if not artwork_url: raise HTTPException(500, "Internal Error: Artwork URL missing at checkout.")

        # Find the user message that *first* introduced this artwork_url
        q_img = select(ChatMessage).where(ChatMessage.session_id == session_uuid, ChatMessage.image_url == artwork_url).order_by(ChatMessage.created_at.asc()).limit(1)
        r_img = await self.db.execute(q_img); artwork_msg = r_img.scalars().first()
        if not artwork_msg:
             # Fallback: find any AI message (e.g., AI generation)
             q_img_ai = select(ChatMessage).where(ChatMessage.session_id == session_uuid, ChatMessage.image_url == artwork_url).order_by(ChatMessage.created_at.asc()).limit(1)
             r_img_ai = await self.db.execute(q_img_ai); artwork_msg = r_img_ai.scalars().first()
        
        if not artwork_msg: raise HTTPException(404, "Artwork image message not found in history for checkout.")

        # _upload_to_printful now uploads from the Cloudinary URL in artwork_msg
        upload_results = await asyncio.gather(_upload_to_printful(artwork_msg, printful_headers))
        file_id_map = {res["view"]: res["id"] for res in upload_results}; file_ids_json = json.dumps(file_id_map)

        costs = await _get_printful_costs(order_item, recipient, printful_headers)

        # --- Database and Stripe Transaction (Unchanged) ---
        # This transaction logic is correct. The begin_nested() block commits
        # order/payment to the session, and the final commit() saves
        # the session state change and the order/payment to the DB.
        async with self.db.begin_nested():
            new_order = OrderRecord(user_id=self.user.id, product_name=order_item.product_name, image_id=artwork_msg.id)
            self.db.add(new_order); await self.db.flush()

            try:
                 intent = stripe.PaymentIntent.create(
                    amount=costs["total_cents"], currency=costs["currency"],
                    automatic_payment_methods={"enabled": True},
                    metadata={"order_id": str(new_order.id), "user_id": str(self.user.id), "session_id": str(session_uuid)},
                    timeout=STRIPE_TIMEOUT
                 )
            except stripe.error.StripeError as e:
                 log.error(f"Stripe PI create failed: {e}")
                 raise HTTPException(502, f"Payment Processor Error: {e.user_message or 'Could not initiate payment.'}")
            except Exception as e:
                 log.error(f"Unexpected error creating Stripe PI: {e}", exc_info=True)
                 raise HTTPException(500, "Internal error during payment initiation.")

            new_payment = OrderPayment(
                order_id=new_order.id, user_id=self.user.id, payment_intent_id=intent.id,
                total_cost_cents=costs["total_cents"], currency=costs["currency"],
                recipient_json=recipient.model_dump_json(), printful_file_ids_json=file_ids_json,
                variant_id=order_item.variant_id, status="pending_payment"
            )
            self.db.add(new_payment); await self.db.flush()

            try:
                stripe.PaymentIntent.modify(intent.id, metadata={**intent.metadata, "order_payment_id": str(new_payment.id)})
            except stripe.error.StripeError as e:
                log.warning(f"Failed to modify Stripe PI metadata for {intent.id}: {e}")

            session.draft_order_details = {}
            session.state = "designing"
            log.info(f"Checkout initiated for order {new_order.id}. Session {session_uuid} state reset to designing.")

            await self.db.commit() # This commits the nested transaction

        return CheckoutResponse(
            order_id=str(new_order.id), payment_intent_id=intent.id, client_secret=intent.client_secret,
            total_cost=round(costs["total_cents"] / 100, 2), currency=costs["currency"]
        )

    # --- Stripe Webhook Handler (Unchanged) ---
    async def handle_stripe_webhook(self, payload: bytes, sig_header: str) -> dict:
        """Handles payment confirmation and triggers fulfillment."""
        if not STRIPE_WEBHOOK_SECRET: log.error("Stripe webhook secret missing."); raise HTTPException(500, "Webhook secret missing.")
        try: event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
        except ValueError: log.warning("Stripe hook invalid payload."); raise HTTPException(400, "Invalid payload")
        except stripe.error.SignatureVerificationError: log.warning("Stripe hook invalid signature."); raise HTTPException(400, "Invalid signature")
        except Exception as e: log.error(f"Error constructing Stripe event: {e}"); raise HTTPException(500, "Webhook processing error")

        intent = event.data.object
        payment_id_str = intent.metadata.get("order_payment_id") if hasattr(intent, 'metadata') else None

        if event.type == "payment_intent.succeeded":
            if not payment_id_str: log.error(f"Stripe event {event.id} (succeeded) missing 'order_payment_id'."); return {"status": "error", "reason": "Missing metadata"}
            try: payment_id = uuid.UUID(payment_id_str)
            except ValueError: log.error(f"Stripe event {event.id} invalid UUID: {payment_id_str}"); return {"status": "error", "reason": "Invalid payment_id format"}

            payment = await self.db.get(OrderPayment, payment_id)
            if not payment: log.error(f"Stripe event {event.id} non-existent OrderPayment: {payment_id}"); return {"status": "error", "reason": "Payment record not found"}
            if payment.status != "pending_payment": log.info(f"Stripe event {event.id} already processed (status: {payment.status})."); return {"status": "ok", "message": "Already processed"}

            payment.status = "succeeded"; await self.db.flush()
            order = await self.db.get(OrderRecord, payment.order_id)
            if not order:
                 payment.status = "error"; payment.error_message = f"OrderRecord {payment.order_id} not found post-payment."; await self.db.commit()
                 log.error(f"Stripe event {event.id} succeeded but OrderRecord {payment.order_id} missing."); return {"status": "error", "reason": "OrderRecord not found"}

            printful_headers = {"Authorization": f"Bearer {PRINTFUL_API_KEY}"}
            # This task is non-blocking, allowing the webhook to return 200 OK quickly.
            asyncio.create_task(_submit_order_to_printful(payment, order, printful_headers, self.db))
            log.info(f"Stripe event {event.id}: Payment {payment_id} succeeded. Queued Printful submission.")

        elif event.type == "payment_intent.payment_failed":
            if payment_id_str:
                try: payment_id = uuid.UUID(payment_id_str)
                except ValueError: log.error(f"Stripe failure event {event.id} invalid UUID: {payment_id_str}"); return {"status": "error", "reason": "Invalid payment_id"}
                
                payment = await self.db.get(OrderPayment, payment_id)
                if payment and payment.status == "pending_payment":
                    fail_msg = intent.last_payment_error.message if intent.last_payment_error else "Unknown Stripe failure"
                    payment.status = "failed"; payment.error_message = fail_msg[:255]
                    await self.db.commit(); log.warning(f"Stripe event {event.id}: Payment {payment_id_str} failed: {fail_msg}")
                elif payment: log.info(f"Ignoring Stripe failure event {event.id} for payment {payment_id_str} with status {payment.status}")
                else: log.warning(f"Stripe failure event {event.id} for non-existent payment {payment_id_str}")
            else: log.warning(f"Stripe failure event {event.id} missing order_payment_id.")
        else:
             log.info(f"Received unhandled Stripe event type: {event.type}")

        return {"status": "received"}

    # --- MODIFIED: handle_image_upload (DEPRECATED) ---
    async def handle_image_upload(self, file: UploadFile) -> dict:
        """DEPRECATED: Handles user image uploads. This feature is retired."""
        if not self.user: raise HTTPException(401, "User not authenticated")
        
        log.warning(f"User {self.user.id} hit deprecated handle_image_upload endpoint with file '{file.filename}'.")
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="This feature has been retired. All artwork is now generated by the AI."
        )

    async def handle_image_placement(self, base_filename: str, overlay_filename: str, position: str) -> dict:
        """DEPRECATED: Combines images."""
        log.warning("handle_image_placement endpoint is deprecated.")
        raise HTTPException(status_code=501, detail="This endpoint is deprecated.")

# ===================================================================
# FASTAPI ROUTER (Wrappers)
# ===================================================================
router = APIRouter(tags=["GlitchApe Central"])

@router.post("/chat/start", status_code=status.HTTP_201_CREATED)
async def start_new_chat_session(db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Starts a new chat session."""
    # (Implementation unchanged)
    try:
        new_session = ChatSession(user_id=current_user.id, title=f"Chat {datetime.now():%Y-%m-%d %H:%M}", state="designing", draft_order_details={})
        db.add(new_session); await db.commit(); await db.refresh(new_session)
        log.info(f"User {current_user.id} started new session {new_session.id}")
        return {"session_id": str(new_session.id)}
    except Exception as e:
        log.error(f"Failed session create for user {current_user.id}: {e}", exc_info=True); await db.rollback()
        raise HTTPException(500, "Failed to create new chat session.")


@router.post("/ai/chat", response_model=LiveChatResponse)
async def chat_with_ai(prompt: str = Form(...), session_id: uuid.UUID = Form(...), uploaded_image: Optional[UploadFile] = File(None), db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Main AI chat endpoint. Rejects image uploads."""
    # (Implementation unchanged, but handler logic is)
    handler = GlitchApeCentralHandler(db=db, user=current_user)
    return await handler.handle_chat_message(session_id=session_id, prompt=prompt, uploaded_image=uploaded_image)


@router.get("/chat/history/{session_id}")
async def get_chat_history(session_id: uuid.UUID, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Retrieves chat history for a session."""
    # (Implementation unchanged)
    session = await db.get(ChatSession, session_id)
    if not session: raise HTTPException(4404, "Chat session not found")
    if session.user_id != current_user.id: raise HTTPException(403, "Access denied")
    q = select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at.asc())
    r = await db.execute(q); messages = r.scalars().all()
    return {"messages": [{"id": str(m.id), "role": m.role, "content": m.content, "image_url": m.image_url, "created_at": m.created_at.isoformat()} for m in messages]}


@router.post("/orders/initiate-checkout", response_model=CheckoutResponse)
async def initiate_checkout_endpoint(req: CheckoutRequest, db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)):
    """Starts checkout using data collected in the chat session."""
    # (Implementation unchanged)
    handler = GlitchApeCentralHandler(db=db, user=user)
    return await handler.initiate_checkout(req)


@router.post("/orders/stripe-webhook")
async def stripe_webhook_endpoint(request: Request, db: AsyncSession = Depends(get_db)):
    """Handles Stripe payment webhooks."""
    # (Implementation unchanged)
    handler = GlitchApeCentralHandler(db=db)
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    return await handler.handle_stripe_webhook(payload, sig_header)


@router.post("/ai/upload-image")
async def upload_user_image_endpoint(
    file: UploadFile = File(...), 
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """DEPRECATED: Handles user image uploads. This feature is retired."""
    # (This now raises 410 GONE)
    log.warning(f"User {current_user.id} hit deprecated /ai/upload-image endpoint.")
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail="This feature has been retired. All artwork is now generated by the AI."
    )


@router.post("/ai/place-image")
async def place_image_endpoint(
    base_filename: str = Form(...), 
    overlay_filename: str = Form(...), 
    position: str = Form("center"), 
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """DEPRECATED: Endpoint for image placement."""
    # (Implementation unchanged)
    handler = GlitchApeCentralHandler(db=db, user=current_user)
    return await handler.handle_image_placement(base_filename, overlay_filename, position)


# --- REMOVED: get_image_endpoint ---
# The /ai/image/{filename} endpoint is no longer needed
# as images are served directly from Cloudinary's CDN.
