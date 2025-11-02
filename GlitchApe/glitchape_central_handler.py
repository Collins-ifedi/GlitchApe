# glitchape_central_handler.py
"""
Central Brain and API Orchestrator for GlitchApe.

This module provides a unified API interface and contains the central business
logic orchestrator, the 'GlitchApeCentralHandler' class.

MODIFIED: Implements conversational order data collection, variant ID lookup,
          input validation, and enhanced error handling.

MODIFIED (Cloudinary): All image storage (uploads, AI generation) is
          now handled by Cloudinary. Local file storage is removed.

MODIFIED (Printful Mockup): Replaced AI mockup generation with Printful's
          Mockup Generator API for precise, iterative placement.
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
from typing import List, Dict, Any, Optional, Tuple

# --- Cloudinary Imports (NEW) ---
import cloudinary
import cloudinary.uploader
import cloudinary.utils  # <-- ADDED for secure_url fallback
from dotenv import load_dotenv

# --- Google GenAI SDK Imports (FIXED) ---
import google.generativeai as genai
from google.generativeai import types as genai_types
from google.api_core import exceptions as google_exceptions

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

# --- LLaMA 4 (Text Model) Configuration ---
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
if not LLAMA_API_KEY: log.critical("LLAMA_API_KEY not set."); raise RuntimeError("LLAMA_API_KEY not set")
LLAMA_API_URL = "https://openrouter.ai/api/v1/chat/completions"
LLAMA_MODEL = "meta-llama/llama-4-maverick:free"
LLAMA_TIMEOUT = 60 # Seconds

# --- Gemini Vision Model (Google) Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY: log.warning("GEMINI_API_KEY not set. Vision features disabled.")
VISION_MODEL = "gemini-2.5-flash"
GEMINI_TIMEOUT = 90 # Seconds

# Global Gemini Client
gemini_client: Optional[genai.GenerativeModel] = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_client = genai.GenerativeModel(VISION_MODEL)
        log.info(f"Gemini Client initialized for model {VISION_MODEL}.")
    except Exception as e:
        log.error(f"Failed to initialize Gemini Client: {e}")
        gemini_client = None
else:
    log.warning("Gemini Client not initialized: GEMINI_API_KEY is missing.")


# --- Public URL for external models to access images ---


# --- Stable Diffusion (HuggingFace) Configuration ---
HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY: log.critical("HF_API_KEY not set."); raise RuntimeError("HF_API_KEY not set")
HF_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
HF_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}
HF_TIMEOUT = 120 # Seconds

# --- LLaMA System Prompt (Enhanced for Validation & Clarity) ---
LLAMA_SYSTEM_PROMPT = """
You are "Marvick," the central AI brain for GlitchApe, a futuristic clothing design platform.
Your primary role is to be a helpful and creative design assistant AND guide users through the ordering process accurately.
Respond conversationally, with a cyberpunk, enthusiastic tone. Use emojis like 🚀✨ and symbols like <>[]//.

**SESSION STATE:** Current state is provided (e.g., "[STATE: collecting_variant]", "[STATE: collecting_placement]", "[STATE: collecting_placement_position]"). Use this to guide the conversation.

**IMAGE ANALYSIS:** If prompt starts with "[IMAGE ANALYSIS: ...]", use that analysis to understand the uploaded design.

**YOUR TASKS:**
1.  **Chat:** Brainstorm, answer questions.
2.  **Intent Detection:** Analyze prompt, history, and state.
3.  **Orchestration:** Decide next action (tool use or next question).

**INTENTS & ACTIONS (JSON output required LAST):**

1.  **`general_chat`** (Any State): Casual talk.
    * Action: `{"intent": "general_chat", "response_text": "Your conversational reply."}`

2.  **`design_request`** (State: designing): User wants new design (AI or uploaded). System generates artwork.
    * Action: `{"intent": "design_request", "clothing_type": "t-shirt", "design_prompt": "cyberpunk cat logo", "response_text": "// Analyzing... Firing up generators!"}`
    * Valid `clothing_type`: "t-shirt", "hoodie", "jacket", "pants". Default: "t-shirt".

3.  **`image_analysis`** (State: designing): User uploaded image. Confirm analysis and ask next step.
    * Action: `{"intent": "image_analysis", "response_text": "Got the visual data! Looks like [Summary]. Ready to place this, or want to chat more?", "analysis_prompt": "Uploaded image analysis provided."}`

4.  **`design_revision`** (State: designing): User wants changes to the *last* design artwork (not placement).
    * Action: `{"intent": "design_revision", "revision_prompt": "make it neon green", "response_text": "Recalibrating pixels... Change incoming!"}`

5.  **`order_request`** (State: designing -> collecting_placement): User wants to order. Start collection.
    * Action: `{"intent": "order_request", "response_text": "Excellent! 🚀 Let's lock in the specs. What **product type** (e.g., t-shirt, hoodie) and **placement** (e.g., front, back) do you want? (e.g., 'front of a t-shirt', 'back of a hoodie')"}`

6.  **`collect_placement_type`** (State: collecting_placement -> collecting_placement_position): User provided product/placement. Extract & ask position.
    * Action: `{"intent": "collect_placement_type", "extracted_product": "t-shirt", "extracted_placement": "front", "response_text": "<> Product/Placement logged! Now, let's position it. How many **inches wide** should the design be, and how many **inches down from the collar**? (e.g., '8 inches wide, 3 inches down')"}`
    * **CRITICAL:** Extract product and placement. Default placement: 'front'.

7.  **`collect_placement_position`** (State: collecting_placement_position -> collecting_variant): User provided width/offset. Extract, trigger mockup, ask for variant.
    * Action: `{"intent": "collect_placement_position", "extracted_width_in": "8", "extracted_offset_y_in": "3", "response_text": "Position locked! Generating preview... OK! ✨ Check the new mockup. Now, what **size** and **color**? (e.g., 'Medium Black', 'Large White')"}`
    * **CRITICAL:** Extract width and offset_y in *inches*. Default offset_y to 0 if not given.

8.  **`design_revision_placement`** (Any ordering state -> same state): User wants placement changes. Extract & trigger mockup.
    * Action: `{"intent": "design_revision_placement", "revision_width_in": "10", "revision_offset_y_in": "2.5", "response_text": "Recalibrating placement... New mockup incoming!"}`
    * **CRITICAL:** Use this if user says "make it bigger" or "move it down".

9.  **`collect_variant`** (State: collecting_variant -> collecting_quantity): User provided product/size/color. Extract & ask quantity.
    * Action: `{"intent": "collect_variant", "extracted_product": "t-shirt", "extracted_size": "Medium", "extracted_color": "Black", "response_text": "<> Product/Size/Color logged! How many of these do you need? (Just the number, like '1')"}`
    * **CRITICAL:** Extract product, size, *and* color separately if possible.

10. **`collect_quantity`** (State: collecting_quantity -> collecting_recipient_name): User provided quantity. Extract & ask name.
    * Action: `{"intent": "collect_quantity", "extracted_quantity": "1", "response_text": "Quantity set. Who's the target recipient? Need their **full name**."}`

11. **`collect_recipient_name`** (State: collecting_recipient_name -> collecting_address_line1): User provided name. Extract & ask Address Line 1.
    * Action: `{"intent": "collect_recipient_name", "extracted_name": "Jane Doe", "response_text": "Name acquired. Main **street address** (Address Line 1)?"}`

12. **`collect_address_line1`** (State: collecting_address_line1 -> collecting_city): User provided Address Line 1. Extract & ask City.
    * Action: `{"intent": "collect_address_line1", "extracted_address1": "123 Cyber Street", "response_text": "Address logged. And the **city**?"}`

13. **`collect_city`** (State: collecting_city -> collecting_country): User provided City. Extract & ask Country.
     * Action: `{"intent": "collect_city", "extracted_city": "Neo Kyoto", "response_text": "City set. Which **country**? (e.g., 'USA', 'Canada', 'Japan')"}`

14. **`collect_country`** (State: collecting_country -> collecting_state_zip): User provided Country. Extract & ask State/Province + ZIP/Postal code.
     * Action: `{"intent": "collect_country", "extracted_country": "USA", "response_text": "Country locked. Almost done! Need **State/Province** and **ZIP/Postal Code**. (e.g., 'CA 90210', 'Ontario L4B 4R3', or just ZIP if no state like '10115' for Germany)"}`

15. **`collect_state_zip`** (State: collecting_state_zip -> ready_for_checkout): User provided State/ZIP. Extract & confirm. Instruct to checkout.
     * Action: `{"intent": "collect_state_zip", "extracted_state_zip": "CA 90210", "response_text": "All data compiled! ✨ Design, specs, destination locked. Hit **'Checkout'** to bridge to payment!"}`

16. **`order_cancel`** (Any ordering state -> designing): User cancels order. Reset state.
    * Action: `{"intent": "order_cancel", "response_text": "// Order sequence aborted. Back to the design board!"}`

17. **`order_modify`** (Any ordering state -> specific collecting state): User wants to change something. Ask *what* specifically.
    * Action: `{"intent": "order_modify", "response_text": "Need a parameter adjustment? Specify what to change (placement, product/size/color, quantity, name, address, etc.)?"}`

**RULES:**
-   **JSON object LAST.** Mandatory `response_text`.
-   **Use State:** Ask the *next* question based on `[STATE: ...]`.
-   **Extract Input:** Use `extracted_...` fields for `collect_...` intents. Be precise. Extract `product`, `size`, `color` separately in `collect_variant`.
-   **Validation:** If user input seems invalid for the current step (e.g., non-numeric quantity, unclear address), use `general_chat` to ask for clarification *without changing the state*. Example: `{"intent": "general_chat", "response_text": "Hold up - that quantity 'lots' isn't computing. Just the number please?"}`
-   Be creative, cyberpunk, helpful! Stay on task during order collection.
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
COUNTRIES_NEEDING_STATES = {"US", "CA", "AU", "JP"}
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

# --- ADDED: Variant ID Lookup Implementation ---
async def _find_printful_variant_id(product_type: str, size: str, color: str) -> Optional[int]:
    """Finds Printful variant ID based on product type, size, and color."""
    product_id = PRODUCT_TYPE_TO_PRINTFUL_ID.get(product_type.lower().strip())
    if not product_id:
        log.warning(f"Product type '{product_type}' not mapped to a Printful Product ID.")
        return None

    variants = await _fetch_printful_product_variants(product_id)
    if variants is None:
        return None

    if not variants:
        log.warning(f"No variants found for Printful Product ID {product_id} ({product_type}).")
        return None

    target_size = size.strip().lower()
    target_color = color.strip().lower()

    for variant in variants:
        variant_size = (variant.get("size") or "").strip().lower()
        variant_color = (variant.get("color") or "").strip().lower()

        if variant_size == target_size and variant_color == target_color:
            log.info(f"Exact match found for {product_type}/{size}/{color}: Variant ID {variant['id']}")
            return variant["id"]

    fuzzy_target_color = re.sub(r'\b(heather|triblend)\b', '', target_color).strip()
    for variant in variants:
        variant_size = (variant.get("size") or "").strip().lower()
        variant_color = (variant.get("color") or "").strip().lower()
        fuzzy_variant_color = re.sub(r'\b(heather|triblend)\b', '', variant_color).strip()

        if variant_size == target_size and fuzzy_variant_color == fuzzy_target_color:
            log.info(f"Fuzzy match found for {product_type}/{size}/{color}: Variant ID {variant['id']} (Matched on '{fuzzy_variant_color}')")
            return variant["id"]

    log.warning(f"No matching variant found for Product ID {product_id} ({product_type}) with size='{size}' and color='{color}'.")
    return None

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


# --- MODIFIED: Replaced _save_temp_image with Cloudinary uploader ---
async def _upload_to_cloudinary(
    file_source: UploadFile | BytesIO | str | bytes,
    folder: str,
    public_id_prefix: Optional[str] = None
) -> str:
    """
    Uploads a file (from UploadFile, bytes, or path) to Cloudinary
    and returns the secure URL.
    
    Runs the synchronous Cloudinary upload in a separate thread, explicitly
    passing a synchronous BytesIO wrapper to avoid 'coroutine' TypeErrors.
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
        
        # 2. Wrap the raw bytes in a synchronous file-like object
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
        return secure_url

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


# --- MODIFIED: _analyze_image_with_vision_model ---
async def _analyze_image_with_vision_model(image_url: str, user_prompt: str) -> str:
    """Uses Gemini SDK to describe image content from a public URL."""
    global gemini_client
    if not gemini_client: return "ERROR: Gemini Client not initialized."

    log.info(f"Calling Gemini ({VISION_MODEL}) via SDK for analysis on {image_url}...")
    
    try:
        # Gemini 1.5 Flash SDK can accept public URLs, but it's often
        # more reliable to send the bytes.
        async with httpx.AsyncClient() as client:
            try:
                head_resp = await client.head(image_url, follow_redirects=True, timeout=10)
                head_resp.raise_for_status()
                mime_type = head_resp.headers.get("content-type")
                if not mime_type or not mime_type.startswith("image/"):
                    raise ValueError(f"Invalid MIME type from URL: {mime_type}")
                
                get_resp = await client.get(image_url, follow_redirects=True, timeout=30)
                get_resp.raise_for_status()
                image_bytes = get_resp.content

            except httpx.RequestError as e:
                log.error(f"Failed to fetch image from URL {image_url} for Gemini: {e}")
                return f"ERROR: Could not fetch image from URL."

        # Create the image part for the Gemini API
        image_part = {"mime_type": mime_type, "data": image_bytes}

        vision_analysis_prompt = f"Analyze clothing design in image. Describe content, style, primary colors objectively for mockup generation. User request: '{user_prompt}'"

        def sync_generate():
            return gemini_client.generate_content(
                [image_part, vision_analysis_prompt],
                generation_config=genai_types.GenerationConfig(max_output_tokens=1024, temperature=0.3),
                request_options={'timeout': GEMINI_TIMEOUT}
            )
        response = await asyncio.to_thread(sync_generate)

        analysis_text = response.text
        if not analysis_text: log.warning(f"Gemini returned empty analysis for {image_url}."); return "Vision model returned empty analysis."
        log.info(f"Gemini analysis successful for {image_url}.")
        return analysis_text

    except google_exceptions.GoogleAPIError as e:
        log.error(f"Gemini API Error: {e}", exc_info=True)
        return f"ERROR: Gemini API Error - {e.message[:100]}"
    except ValueError as e:
         log.error(f"Gemini Value Error (likely processing): {e}", exc_info=True)
         return f"ERROR: Gemini Processing Error - {str(e)[:100]}"
    except Exception as e:
        log.error(f"Unexpected error in Gemini SDK call: {e}", exc_info=True)
        return f"ERROR: Unexpected Gemini SDK Error - {str(e)[:100]}"


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

        # Upload bytes to Cloudinary
        image_url = await _upload_to_cloudinary(
            file_source=image_data,
            folder=CLOUDINARY_ARTWORK_FOLDER
        )
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
    log.info(f"Generating Printful mockup for prod {product_id}, w_in: {width_in}, y_in: {offset_y_in}")
    
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
                        # We are passing a URL (str) to _upload_to_cloudinary, which
                        # the modified helper function supports.
                        final_cloudinary_url = await _upload_to_cloudinary(
                            file_source=mockup_url,
                            folder=CLOUDINARY_MOCKUP_FOLDER
                        )
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
    def __init__(self, db: AsyncSession, user: Optional[User] = None):
        self.db = db
        self.user = user
        # This is a non-blocking task to populate the cache if it's empty.
        # The function itself checks if the cache is already populated.
        asyncio.create_task(_populate_location_data())

    async def _get_chat_history_for_context(self, session: ChatSession, limit: int = 10) -> List[Dict[str, str]]:
        """Fetches history, adds current state."""
        # (Implementation unchanged)
        q = select(ChatMessage).where(ChatMessage.session_id == session.id).order_by(ChatMessage.created_at.desc()).limit(limit)
        r = await self.db.execute(q); messages = r.scalars().all()
        llama_history = [{"role": "system", "content": f"[STATE: {session.state or 'designing'}]"}]
        for msg in reversed(messages):
            api_role = "assistant" if msg.role == "ai" else msg.role
            content = msg.content
            if content.startswith("[IMAGE ANALYSIS:") and len(content) > 300: content = "[Image analysis provided] User request: " + content.split("User request:", 1)[-1].strip()
            
            # --- PRODUCTION FIX (Resolves LLaMA 400 'oneOf' Error) ---
            # The LLaMA API schema requires the role to be one of ["user", "assistant", "system"].
            # The previous code hard-coded `"role": "api_role"`, which is invalid.
            # This fix uses the `api_role` variable (which is correctly set to "user" or "assistant")
            # as the key's value, satisfying the schema.
            llama_history.append({"role": api_role, "content": content})
            # --- END FIX ---
            
        return llama_history

    async def _call_llama_brain(self, prompt: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Calls OpenRouter LLaMA, handles JSON extraction and errors."""
        # (Implementation unchanged)
        headers = { "Authorization": f"Bearer {LLAMA_API_KEY}", "Content-Type": "application/json", "HTTP-Referer": "https://glitchape.fun", "X-Title": "GlitchApe"}
        messages = [ {"role": "system", "content": LLAMA_SYSTEM_PROMPT}, *history, {"role": "user", "content": prompt} ]
        payload = { "model": LLAMA_MODEL, "messages": messages }
        try:
            async with httpx.AsyncClient(timeout=LLAMA_TIMEOUT) as client:
                response = await client.post(LLAMA_API_URL, headers=headers, json=payload)
                response.raise_for_status()
            data = response.json()
            llama_response_content = data.get("choices", [{}])[0].get("message", {}).get("content")
            if not llama_response_content: log.error(f"LLaMA empty content. Data: {data}"); raise HTTPException(502, "LLaMA returned empty content.")
            clean_content = llama_response_content.strip(); start = clean_content.find('{'); end = clean_content.rfind('}')
            if start != -1 and end != -1 and end > start: json_str = clean_content[start:end+1]
            else: log.error(f"LLaMA failed to include JSON object. Raw: {llama_response_content}"); json_str = clean_content
            try: return json.loads(json_str)
            except json.JSONDecodeError as e: log.error(f"LLaMA invalid JSON. Err: {e}. Str: {json_str}. Orig: {llama_response_content}"); raise HTTPException(502, f"LLaMA returned malformed JSON: {e}")
        except httpx.HTTPStatusError as e:
            log.error(f"LLaMA API Error: {e.response.status_code} - {e.response.text[:200]}")
            raise HTTPException(status_code=502, detail=f"LLaMA Service Error ({e.response.status_code}).")
        except httpx.RequestError as e:
            log.error(f"Network error calling LLaMA: {e}")
            raise HTTPException(status_code=504, detail="LLaMA Service Network Error.")
        except (IndexError, KeyError, Exception) as e:
             log.error(f"Error processing LLaMA response: {e}", exc_info=True)
             raise HTTPException(status_code=502, detail=f"Error processing LLaMA response: {str(e)[:100]}")


    # --- ADDED: Mockup Generation Wrapper ---
    async def _trigger_mockup_generation(self, draft_details: Dict[str, Any]) -> str:
        """
        Helper to validate draft_details and call the Printful mockup generator.
        Returns the new mockup's Cloudinary URL.
        """
        try:
            # 1. Extract data from draft_details
            product_input = draft_details.get('product_input')
            design_url = draft_details.get('artwork_image_url')
            placement_id = draft_details.get('placement_id')
            width_in = float(draft_details.get('user_width_in', 8)) # Default 8"
            offset_y_in = float(draft_details.get('user_offset_y_in', 3)) # Default 3"

            if not all([product_input, design_url, placement_id]):
                log.error(f"Mockup trigger failed: Missing data. Need product, design_url, placement. Got: {draft_details}")
                raise ValueError("Missing data for mockup generation.")
            
            product_id = PRODUCT_TYPE_TO_PRINTFUL_ID.get(product_input.lower().strip())
            if not product_id:
                log.error(f"Mockup trigger failed: No product_id for '{product_input}'")
                raise ValueError(f"Invalid product type '{product_input}' for mockup.")

            # 2. Call the (newly rewritten) helper
            mockup_url = await _generate_mockup_preview(
                product_id=product_id,
                design_url=design_url,
                placement_id=placement_id,
                width_in=width_in,
                offset_y_in=offset_y_in
            )
            return mockup_url

        except (ValueError, TypeError) as e:
            log.warning(f"Validation error in _trigger_mockup_generation: {e}")
            raise HTTPException(status_code=400, detail=f"Mockup generation error: {e}")
        except HTTPException:
            raise # Re-raise errors from _generate_mockup_preview
        except Exception as e:
            log.error(f"Unexpected error in _trigger_mockup_generation: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal mockup generation error.")


    # --- MAIN CHAT HANDLER (MODIFIED for Cloudinary) ---
    async def handle_chat_message(self, session_id: uuid.UUID, prompt: str, uploaded_image: Optional[UploadFile] = None) -> LiveChatResponse:
        """Orchestrates chat, design, vision, and validated order data collection."""
        if not self.user: raise HTTPException(401, "User not authenticated")
        session = await self.db.get(ChatSession, session_id)
        if not session or session.user_id != self.user.id: raise HTTPException(404, "Chat session not found or access denied")

        current_state = session.state or "designing"
        draft_details = session.draft_order_details if isinstance(session.draft_order_details, dict) else {}

        # --- Local variables ---
        # local_image_path: Optional[str] = None # <-- REMOVED
        image_analysis_text: Optional[str] = None
        analysis_error_message: Optional[str] = None
        db_image_url: Optional[str] = None # This will now be a Cloudinary URL
        ai_text_response = "Processing your request..."
        next_state = current_state
        image_previews: List[ImagePreview] = []
        ask_user_to_clarify = False
        clarification_request = ""

        # 1. Handle Upload & Vision (Only in designing state)
        if current_state == "designing" and uploaded_image and uploaded_image.filename:
            try:
                # --- MODIFIED: Upload to Cloudinary ---
                log.info(f"User {self.user.id} uploading '{uploaded_image.filename}'. Uploading to Cloudinary...")
                db_image_url = await _upload_to_cloudinary(
                    file_source=uploaded_image,
                    folder=CLOUDINARY_UPLOAD_FOLDER,
                    public_id_prefix=f"user_{self.user.id}_{session_id}"
                )
                log.info(f"User {self.user.id} uploaded to {db_image_url}. Analyzing...")
                # --- END MODIFICATION ---

                # --- MODIFIED: Pass URL to vision model ---
                image_analysis_text = await _analyze_image_with_vision_model(db_image_url, prompt)
                
                if image_analysis_text.startswith("ERROR:"):
                    analysis_error_message = image_analysis_text; image_analysis_text = None
                    log.warning(f"Vision analysis failed: {analysis_error_message}")
                else: log.info("Vision analysis successful.")
            except HTTPException as e: raise e # Propagate HTTP errors from validation/upload
            except Exception as e: log.error(f"Upload/Vision error: {e}", exc_info=True); analysis_error_message = "Image processing setup error."

        # 2. Prepare LLaMA prompt
        final_prompt_for_llama = prompt
        if image_analysis_text: final_prompt_for_llama = f"[IMAGE ANALYSIS: {image_analysis_text}] User request: {prompt}"
        elif analysis_error_message: final_prompt_for_llama = f"[IMAGE ANALYSIS FAILED: {analysis_error_message}] User request: {prompt}"
        elif db_image_url: final_prompt_for_llama = f"[IMAGE UPLOADED: {db_image_url}] User request: {prompt}" # URL is now Cloudinary

        # 3. Save User Message
        user_msg = ChatMessage(session_id=session_id, user_id=self.user.id, role="user", content=prompt, image_url=db_image_url if current_state == "designing" else None)
        self.db.add(user_msg); await self.db.flush()

        # 4. Get Context & Call LLaMA
        history = await self._get_chat_history_for_context(session)
        try: action_json = await self._call_llama_brain(final_prompt_for_llama, history)
        except HTTPException as e:
            ai_text_response = f"My core processor (LLaMA) had an issue ({e.detail}). Please try again."
            ai_msg = ChatMessage(session_id=session_id, user_id=self.user.id, role="ai", content=ai_text_response)
            self.db.add(ai_msg); await self.db.commit()
            return LiveChatResponse(session_id=str(session_id), response_text=ai_text_response)

        # 5. Process LLaMA's Response & Update State/Draft with Validation
        intent = action_json.get("intent")
        ai_text_response = action_json.get("response_text", "Got it, processing...")

        try:
            # --- MODIFIED: Placement Revision Intent (handles all ordering states) ---
            if intent == "design_revision_placement" and current_state.startswith("collecting_"):
                try:
                    # This intent can override the AI's main text response
                    ai_text_response = action_json.get("response_text", "Recalibrating placement...")
                    
                    # Update draft details
                    rev_width = action_json.get("revision_width_in")
                    rev_offset = action_json.get("revision_offset_y_in")
                    
                    if not rev_width and not rev_offset:
                         raise ValueError("No revision data provided.")

                    if rev_width: draft_details['user_width_in'] = float(rev_width)
                    if rev_offset: draft_details['user_offset_y_in'] = float(rev_offset)
                    
                    log.info(f"Session {session_id}: Revising placement. w:{rev_width}, y:{rev_offset}")

                    # --- Trigger Printful Mockup ---
                    mockup_url = await self._trigger_mockup_generation(draft_details)
                    image_previews.append(ImagePreview(view_name="Revised Mockup", url=mockup_url))
                    
                    # Save mockup to chat
                    ai_mockup_msg = ChatMessage(session_id=session_id, user_id=self.user.id, role="ai", content=f"Revised Mockup: w:{draft_details['user_width_in']}\" y:{draft_details['user_offset_y_in']}\"", image_url=mockup_url)
                    self.db.add(ai_mockup_msg)
                    
                    # IMPORTANT: Stay in the *current* state
                    next_state = current_state
                    log.info(f"Session {session_id}: Revision mockup: {mockup_url}. State remains {current_state}")

                except (ValueError, TypeError):
                    ai_text_response = "Couldn't parse that revision. Try 'make it 10 inches wide' or 'move it 4 inches down'."
                except HTTPException as e:
                    ai_text_response = f"// Mockup revision failed: {e.detail}. Please try again."

            # --- Designing State ---
            elif current_state == "designing":
                if intent == "design_request" or (intent == "image_analysis" and db_image_url):
                    clothing_type = action_json.get("clothing_type", "t-shirt")
                    design_prompt = action_json.get("design_prompt")
                    artwork_url = db_image_url; artwork_content = None; mockup_prompt_source = None

                    if db_image_url: # User uploaded (URL is from Cloudinary)
                        artwork_content = f"User-uploaded design artwork: {prompt}"
                        mockup_prompt_source = image_analysis_text or "custom logo or design"
                    elif design_prompt: # AI generates
                        # _generate_design_artwork now uploads to Cloudinary and returns URL
                        artwork_url = await _generate_design_artwork(design_prompt)
                        artwork_content = f"Generated design artwork: {design_prompt}"
                        mockup_prompt_source = design_prompt
                    else:
                        ai_text_response = "Need a clearer design prompt, cybernaut! Describe what you envision."; intent = "general_chat"

                    if artwork_url and mockup_prompt_source:
                        ai_artwork_msg = ChatMessage(session_id=session_id, user_id=self.user.id, role="ai", content=artwork_content, image_url=artwork_url)
                        self.db.add(ai_artwork_msg); draft_details['artwork_image_url'] = artwork_url

                        # --- MODIFICATION: No mockup generated here anymore ---
                        # Mockup is generated after placement is collected.
                        
                        ai_text_response = action_json.get("response_text") or (f"Design set! Using your upload. Artwork link below. Ready to order, or refine?" if db_image_url else f"Design complete! Artwork link below. Order or revise?")
                        image_previews.append(ImagePreview(view_name="Artwork Decal", url=artwork_url))
                    elif intent != "general_chat":
                         ai_text_response = "Couldn't generate the design visuals. Please try describing it differently."


                elif intent == "order_request":
                    if not draft_details.get('artwork_image_url'):
                         ai_text_response = "We need a design first! Describe what you want or upload an image."
                         intent = "general_chat"
                    else:
                         # --- MODIFIED: Transition to placement collection ---
                         next_state = "collecting_placement"
                         draft_details = {'artwork_image_url': draft_details.get('artwork_image_url')}
                         log.info(f"Session {session_id} state -> {next_state}")

            # --- ADDED: Placement Collection States ---
            elif current_state == "collecting_placement":
                if intent == "collect_placement_type":
                    prod_type = action_json.get("extracted_product", "").strip()
                    placement = action_json.get("extracted_placement", "front").strip()

                    if prod_type and PRODUCT_TYPE_TO_PRINTFUL_ID.get(prod_type.lower()):
                        draft_details['product_input'] = prod_type
                        draft_details['placement_id'] = placement.lower() # e.g., 'front', 'back'
                        next_state = "collecting_placement_position"
                        log.info(f"Session {session_id}: Collected placement {prod_type}/{placement}. State -> {next_state}")
                    else:
                        ask_user_to_clarify = True
                        clarification_request = f"Sorry, I don't recognize '{prod_type}' as a product. Try 't-shirt' or 'hoodie'."
                elif intent == "order_cancel": next_state = "designing"; draft_details = {}

            elif current_state == "collecting_placement_position":
                if intent == "collect_placement_position":
                    try:
                        width_in = float(action_json.get("extracted_width_in", "8").strip())
                        offset_y_in = float(action_json.get("extracted_offset_y_in", "3").strip())
                        
                        if not (0 < width_in <= 20 and 0 <= offset_y_in <= 20):
                             raise ValueError("Invalid dimensions")

                        draft_details['user_width_in'] = width_in
                        draft_details['user_offset_y_in'] = offset_y_in
                        
                        # --- Trigger Printful Mockup ---
                        mockup_url = await self._trigger_mockup_generation(draft_details)
                        image_previews.append(ImagePreview(view_name="Placement Mockup", url=mockup_url))
                        
                        # Save mockup to chat
                        ai_mockup_msg = ChatMessage(session_id=session_id, user_id=self.user.id, role="ai", content=f"Mockup: {draft_details['product_input']} w:{width_in}\" y:{offset_y_in}\"", image_url=mockup_url)
                        self.db.add(ai_mockup_msg)
                        
                        next_state = "collecting_variant"
                        log.info(f"Session {session_id}: Collected position {width_in}\"w, {offset_y_in}\"y. Mockup: {mockup_url}. State -> {next_state}")

                    except (ValueError, TypeError):
                        ask_user_to_clarify = True
                        clarification_request = "Please provide the width and offset as numbers (in inches). (e.g., '8 wide, 3 down')"
                    except HTTPException as e:
                        # Error from _trigger_mockup_generation
                        log.error(f"Mockup gen failed for session {session_id}: {e.detail}")
                        ai_text_response = f"// Glitch! Mockup generator failed: {e.detail}. Let's try that position again?"
                        # Stay in the same state
                        next_state = current_state
                elif intent == "order_cancel": next_state = "designing"; draft_details = {}

            # --- ORDER DATA COLLECTION STATES (Unchanged) ---
            elif current_state == "collecting_variant":
                if intent == "collect_variant":
                    # --- MODIFICATION: Product type is already in draft_details ---
                    prod_type = draft_details.get('product_input')
                    size = action_json.get("extracted_size", "").strip()
                    color = action_json.get("extracted_color", "").strip()
                    
                    # LLaMA might extract product_type again, which is fine, but
                    # we should prioritize the one from the placement step.
                    if not prod_type:
                        prod_type = action_json.get("extracted_product", "").strip()

                    if prod_type and size and color:
                        variant_id = await _find_printful_variant_id(prod_type, size, color)
                        if variant_id:
                            draft_details['product_input'] = prod_type # Ensure it's set
                            draft_details['size_input'] = size
                            draft_details['color_input'] = color
                            draft_details['variant_id'] = variant_id
                            log.info(f"Session {session_id}: Found Variant ID {variant_id} for {prod_type}/{size}/{color}")
                            next_state = "collecting_quantity"
                        else:
                            ask_user_to_clarify = True
                            clarification_request = f"Hmm, I couldn't find a '{size} {color} {prod_type}' in the catalog. Can you double-check the spelling or try a different combination? (e.g., 'Medium Black', 'Large White')"
                    else:
                        ask_user_to_clarify = True
                        clarification_request = "Didn't quite catch the size AND color. Please provide both (e.g., 'Large White')."
                elif intent == "order_cancel": next_state = "designing"; draft_details = {}

            elif current_state == "collecting_quantity":
                if intent == "collect_quantity":
                    extracted = action_json.get("extracted_quantity", "").strip()
                    try:
                        quantity = int(extracted)
                        if quantity > 0 and quantity <= 1000:
                             draft_details['quantity'] = quantity
                             log.info(f"Session {session_id}: Collected quantity '{quantity}'")
                             next_state = "collecting_recipient_name"
                        else: raise ValueError("Quantity out of range")
                    except ValueError:
                        ask_user_to_clarify = True
                        clarification_request = "Invalid quantity. Please enter a number between 1 and 1000."
                elif intent == "order_cancel": next_state = "designing"; draft_details = {}

            elif current_state == "collecting_recipient_name":
                if intent == "collect_recipient_name":
                    extracted = action_json.get("extracted_name", "").strip()
                    if extracted: draft_details['recipient_name'] = extracted; next_state = "collecting_address_line1"; log.info(f"Session {session_id}: Collected name '{extracted}'")
                    else: ask_user_to_clarify = True; clarification_request = "Please provide the recipient's full name."
                elif intent == "order_cancel": next_state = "designing"; draft_details = {}

            elif current_state == "collecting_address_line1":
                if intent == "collect_address_line1":
                    extracted = action_json.get("extracted_address1", "").strip()
                    if extracted: draft_details['recipient_address1'] = extracted; next_state = "collecting_city"; log.info(f"Session {session_id}: Collected address1")
                    else: ask_user_to_clarify = True; clarification_request = "The main street address is needed."
                elif intent == "order_cancel": next_state = "designing"; draft_details = {}

            elif current_state == "collecting_city":
                 if intent == "collect_city":
                    extracted = action_json.get("extracted_city", "").strip()
                    if extracted: draft_details['recipient_city'] = extracted; next_state = "collecting_country"; log.info(f"Session {session_id}: Collected city")
                    else: ask_user_to_clarify = True; clarification_request = "Which city should this go to?"
                 elif intent == "order_cancel": next_state = "designing"; draft_details = {}

            elif current_state == "collecting_country":
                 if intent == "collect_country":
                    extracted = action_json.get("extracted_country", "").strip()
                    validated_code = _validate_country_code(extracted) if extracted else None
                    if validated_code: draft_details['recipient_country_code'] = validated_code; next_state = "collecting_state_zip"; log.info(f"Session {session_id}: Collected country '{validated_code}'")
                    else: ask_user_to_clarify = True; clarification_request = f"Invalid country '{extracted}'. Please provide a valid country name or 2-letter code."
                 elif intent == "order_cancel": next_state = "designing"; draft_details = {}

            elif current_state == "collecting_state_zip":
                if intent == "collect_state_zip":
                    extracted = action_json.get("extracted_state_zip", "").strip()
                    country_code = draft_details.get('recipient_country_code')
                    if not country_code:
                         log.error(f"Session {session_id}: Country code missing in state_zip collection state.")
                         ask_user_to_clarify = True; clarification_request = "Something went wrong, I need the country again first."; next_state = "collecting_country"
                    elif extracted:
                         parsed = _parse_state_zip(extracted)
                         state_input = parsed.get("state_code")
                         zip_input = parsed.get("zip_code")
                         
                         validated_state = _validate_state_code(country_code, state_input)

                         if country_code in COUNTRIES_NEEDING_STATES and not validated_state and state_input:
                             ask_user_to_clarify = True
                             clarification_request = f"Invalid state '{state_input}' for {country_code}. Please provide a valid state/province name or code."
                         elif country_code in COUNTRIES_NEEDING_STATES and not validated_state and not state_input and zip_input:
                              ask_user_to_clarify = True
                              clarification_request = f"State/province is required for {country_code}. Please provide both state and ZIP (e.g., 'CA 90210')."
                         else:
                             draft_details['recipient_state_code'] = validated_state
                             draft_details['recipient_zip'] = zip_input
                             next_state = "ready_for_checkout"
                             log.info(f"Session {session_id}: Collected state/zip. State: {validated_state}, ZIP: {zip_input}")
                             log.info(f"Session {session_id} state -> {next_state}. Final Draft: {draft_details}")
                    else:
                        ask_user_to_clarify = True; clarification_request = "Please provide the State/Province and ZIP/Postal Code."
                elif intent == "order_cancel": next_state = "designing"; draft_details = {}
            
            # --- Generic order modification/cancel/clarification (Unchanged) ---
            elif intent == "order_cancel":
                 next_state = "designing"; draft_details = {}
                 log.info(f"Session {session_id} order cancelled, state -> {next_state}")
            elif intent == "order_modify":
                 pass # The AI response ("Specify what to change") is sufficient
            elif current_state.startswith("collecting_") or current_state == "ready_for_checkout":
                 # User said something other than a revision, cancel, or modify
                 ask_user_to_clarify = True

            # --- Handle Clarification (Unchanged) ---
            if ask_user_to_clarify:
                 ai_text_response = clarification_request or f"Sorry, I need the {current_state.split('_')[-1]} info. Can you provide that again?"
                 next_state = current_state
                 log.info(f"Session {session_id}: Asking for clarification in state '{current_state}'.")


        except Exception as e:
            log.error(f"Error processing intent '{intent}' in state '{current_state}': {e}", exc_info=True)
            ai_text_response = "System glitch processing that step. Please try again or type 'cancel order'."
            next_state = current_state

        # 6. Update Session State and Draft Details in DB (Unchanged)
        if next_state != session.state or draft_details != session.draft_order_details:
             session.state = next_state
             session.draft_order_details = draft_details
             log.info(f"Updating session {session_id}: state='{next_state}'")

        # 7. Save Final AI Response (Unchanged)
        ai_msg = ChatMessage(session_id=session_id, user_id=self.user.id, role="ai", content=ai_text_response, image_url=image_previews[-1].url if image_previews else None)
        self.db.add(ai_msg)

        try:
             await self.db.commit()
        except Exception as db_err:
             log.error(f"Database commit error in handle_chat_message for session {session_id}: {db_err}", exc_info=True)
             await self.db.rollback()
             raise HTTPException(status_code=500, detail="Failed to save chat progress.")


        # 8. Return Response (Unchanged)
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
        if session.state != "ready_for_checkout": 
            log.warning(f"Checkout attempt for session {session_uuid} in invalid state '{session.state}'")
            raise HTTPException(400, "Order details not fully collected. Please finish the chat flow.")

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
            session.state = "collecting_placement" # Go back to a reasonable collection state
            await self.db.commit()
            raise HTTPException(status_code=400, detail=f"Invalid or incomplete order data: {e}. Please review details in chat.")

        # --- Proceed with Printful/Stripe (Unchanged) ---
        printful_headers = {"Authorization": f"Bearer {PRINTFUL_API_KEY}"}
        artwork_url = draft_details.get('artwork_image_url')
        if not artwork_url: raise HTTPException(500, "Internal Error: Artwork URL missing at checkout.")

        q_img = select(ChatMessage).where(ChatMessage.session_id == session_uuid, ChatMessage.image_url == artwork_url).limit(1)
        r_img = await self.db.execute(q_img); artwork_msg = r_img.scalars().first()
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

    # --- MODIFIED: handle_image_upload ---
    async def handle_image_upload(self, file: UploadFile) -> dict:
        """Handles user image uploads (for reference/design) by uploading to Cloudinary."""
        if not self.user: raise HTTPException(401, "User not authenticated")
        
        log.info(f"User {self.user.id} uploading file '{file.filename}' via /ai/upload-image endpoint.")
        
        # Upload to Cloudinary using the helper
        image_url = await _upload_to_cloudinary(
            file_source=file,
            folder=CLOUDINARY_UPLOAD_FOLDER,
            public_id_prefix=f"user_{self.user.id}"
        )
        
        return {"status": "success", "filename": file.filename, "file_url": image_url}

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
    """Main AI chat endpoint handling conversation, design, and order collection."""
    # (Removed _cleanup_expired_images task as it's no longer needed)
    handler = GlitchApeCentralHandler(db=db, user=current_user)
    return await handler.handle_chat_message(session_id=session_id, prompt=prompt, uploaded_image=uploaded_image)


@router.get("/chat/history/{session_id}")
async def get_chat_history(session_id: uuid.UUID, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Retrieves chat history for a session."""
    # (Implementation unchanged)
    session = await db.get(ChatSession, session_id)
    if not session: raise HTTPException(404, "Chat session not found")
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
    """Handles user image uploads (for reference/design)."""
    # (This now uses the modified handler)
    handler = GlitchApeCentralHandler(db=db, user=current_user)
    
    # --- FIX: This 'await' is critical to call the async handler correctly ---
    return await handler.handle_image_upload(file)


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