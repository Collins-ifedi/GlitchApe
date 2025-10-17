# ai.py
"""
AI interaction and image generation handler for GlitchApe.
Handles:
- Conversational chat & multi-view image generation (Gemini API)
- Generation of front, back, and sleeve views for clothing.
- Conversational flow for design feedback and ordering.
- Image uploads & PIL-based placement.
- Temporary storage & cleanup (auto-delete after 48h)
"""

import os
import uuid
import shutil
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from PIL import Image

from google import genai  # Gemini API
from jose import jwt

# Import DB and auth components from server.py
from server import get_db, User, get_current_user, ChatSession, ChatMessage

# ===================================================================
# CONFIGURATION
# ===================================================================

AI_API_KEY = os.getenv("AI_API_KEY")
if not AI_API_KEY:
    raise RuntimeError("AI_API_KEY environment variable not set")

# Temporary storage for uploaded/generated images
UPLOAD_DIR = "temp_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize router and Gemini client
# Note: server.py mounts this router at "/api", so paths will be "/api/ai/..."
router = APIRouter(prefix="/ai", tags=["AI"])
client = genai.Client(api_key=AI_API_KEY)


# ===================================================================
# SCHEMAS (Updated for advanced chat)
# ===================================================================

class AIRequest(BaseModel):
    """Request for the main /chat endpoint."""
    prompt: str
    session_id: str  # This is a UUID string from the ChatSession model


class ImagePreview(BaseModel):
    """Represents a single generated image view."""
    view_name: str  # e.g., "front", "back", "left_sleeve"
    url: str        # Publicly accessible URL, e.g., /api/ai/image/uuid.png


class AIResponse(BaseModel):
    """Response from the main /chat endpoint."""
    session_id: str
    ai_message: str
    images: List[ImagePreview] = []
    # State helps the frontend show contextual buttons (e.g., "Order", "Redesign")
    conversation_state: str = "awaiting_feedback"  # States: "designing", "awaiting_feedback", "finalized"


# ===================================================================
# UTILITIES
# ===================================================================

async def cleanup_expired_images():
    """Deletes images from UPLOAD_DIR older than 48 hours."""
    now = datetime.utcnow()
    for filename in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.isfile(file_path):
            try:
                file_time = datetime.utcfromtimestamp(os.path.getmtime(file_path))
                if now - file_time > timedelta(hours=48):  # Updated to 48 hours
                    os.remove(file_path)
            except OSError:
                # Ignore errors (e.g., file in use)
                pass


async def save_temp_image(file: UploadFile) -> str:
    """
    Saves an uploaded image temporarily to UPLOAD_DIR.
    Returns the full local file path.
    """
    try:
        ext = file.filename.split(".")[-1]
        if ext not in ["png", "jpg", "jpeg", "webp"]:
            ext = "png"  # Default to png
        file_id = f"{uuid.uuid4()}.{ext}"
        file_path = os.path.join(UPLOAD_DIR, file_id)
        
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        return file_path
    finally:
        file.file.close()


async def generate_design_view(prompt: str, view: str) -> str:
    """
    Generates a single, backgroundless 2D image view for a design.
    Returns the local file path.
    """
    # This prompt is engineered to request 2D, backgroundless assets
    full_prompt = (
        f"Generate a single, realistic, 2D product image of the {view} view "
        f"of a {prompt}. The image must have a transparent background. "
        f"Only show the clothing item, no models, mannequins, or shadows."
    )

    try:
        # Using the same method as the original file.
        # This API likely routes to an Imagen model.
        response = client.models.generate_image(
            model="gemini-1.5-pro",  # Using model from original file
            prompt=full_prompt,
        )

        if not response.images:
            raise HTTPException(status_code=500, detail=f"AI failed to generate image for {view}")

        image_data = response.images[0].bytes
        file_id = f"{uuid.uuid4()}.png"
        file_path = os.path.join(UPLOAD_DIR, file_id)
        with open(file_path, "wb") as f:
            f.write(image_data)

        return file_path
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI image generation error: {e}")


def get_required_views(prompt: str) -> List[str]:
    """Determines which clothing views to generate based on the prompt."""
    prompt_lower = prompt.lower()
    if "hoodie" in prompt_lower or "shirt" in prompt_lower or "t-shirt" in prompt_lower or "jacket" in prompt_lower:
        return ["front", "back", "left_sleeve", "right_sleeve"]
    if "trousers" in prompt_lower or "pants" in prompt_lower or "jeans" in prompt_lower:
        return ["front", "back"]
    # Default for unknown items
    return ["main_design"]


def place_uploaded_image(base_path: str, overlay_path: str, position: str) -> str:
    """
    Places an uploaded image (overlay) on top of a generated image (base).
    position: 'center', 'top', 'bottom', 'left', 'right'
    Returns the local path to the new combined image.
    """
    try:
        base = Image.open(base_path).convert("RGBA")
        overlay = Image.open(overlay_path).convert("RGBA")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image files: {e}")

    # Resize overlay to a reasonable size, e.g., 25% of base width
    w, h = base.size
    overlay_width = w // 4
    overlay_height = int(overlay.height * (overlay_width / overlay.width))
    overlay = overlay.resize((overlay_width, overlay_height), Image.LANCZOS)

    # Calculate position
    positions = {
        "center": ((w - overlay.width) // 2, (h - overlay.height) // 2),
        "top": ((w - overlay.width) // 2, h // 10),  # 10% from top
        "bottom": ((w - overlay.width) // 2, h - overlay.height - (h // 10)),
        "left": (w // 10, (h - overlay.height) // 2),
        "right": (w - overlay.width - (w // 10), (h - overlay.height) // 2),
    }
    pos = positions.get(position, positions["center"])

    # Paste overlay onto base
    base.paste(overlay, pos, overlay)
    
    result_id = f"{uuid.uuid4()}.png"
    result_path = os.path.join(UPLOAD_DIR, result_id)
    base.save(result_path, "PNG")

    return result_path


# ===================================================================
# ENDPOINTS
# ===================================================================

@router.post("/chat", response_model=AIResponse)
async def chat_with_ai(
    request: AIRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Main AI chat endpoint.
    Handles conversational design, multi-view image generation, and order flow.
    """
    # Run cleanup task in the background
    await cleanup_expired_images()

    try:
        session_uuid = uuid.UUID(request.session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session_id format. Must be a UUID.")

    # 1. Verify session ownership
    session = await db.get(ChatSession, session_uuid)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Chat session not found or access denied")

    # 2. Save user's message to DB
    user_msg = ChatMessage(
        session_id=session_uuid,
        user_id=current_user.id,
        role="user",
        content=request.prompt
    )
    db.add(user_msg)

    # 3. Process request and determine AI action
    ai_text_response = ""
    image_previews: List[ImagePreview] = []
    new_state = "awaiting_feedback"
    
    prompt_lower = request.prompt.lower()

    # Intent: Place order
    if "order" in prompt_lower or "place order" in prompt_lower:
        ai_text_response = "Great! I'm preparing your designs for the order. This feature will be connected to our ordering system soon!"
        new_state = "finalized"
        # In a real implementation, this would gather the latest design URLs
        # and pass them to an ordering service (e.g., order.py)
    
    # Intent: Redesign
    elif "redesign" in prompt_lower or "change" in prompt_lower:
        ai_text_response = "Okay, what part would you like to change, and what should it look like?"
        new_state = "designing"

    # Intent: Generate design
    else:
        views_to_gen = get_required_views(prompt_lower)
        ai_text_response = f"I'm generating the {', '.join(views_to_gen)} views for your design. Here's the preview:"
        
        # Generate all image views in parallel
        tasks = [generate_design_view(request.prompt, view) for view in views_to_gen]
        
        try:
            generated_paths = await asyncio.gather(*tasks)
            
            # Create public URLs and DB entries for each generated image
            for i, local_path in enumerate(generated_paths):
                filename = os.path.basename(local_path)
                # Note: Full path is /api/ai/image/{filename}
                public_url = f"/api/ai/image/{filename}"
                view_name = views_to_gen[i]
                
                image_previews.append(ImagePreview(
                    view_name=view_name,
                    url=public_url
                ))
                
                # Save each image as a separate AI message in the DB
                ai_img_msg = ChatMessage(
                    session_id=session_uuid,
                    user_id=current_user.id,
                    role="ai",
                    content=f"Generated image: {view_name}",
                    image_url=public_url
                )
                db.add(ai_img_msg)

            # Add the follow-up question
            ai_text_response += "\n\nIs it OK to place an order, or would you like to redesign a section?"
            new_state = "awaiting_feedback"

        except Exception as e:
            ai_text_response = f"I tried to generate the images, but an error occurred: {e}"
            new_state = "designing"

    # 4. Save the main AI text response to DB
    ai_text_msg = ChatMessage(
        session_id=session_uuid,
        user_id=current_user.id,
        role="ai",
        content=ai_text_response
    )
    db.add(ai_text_msg)

    # 5. Commit all DB changes
    await db.commit()

    # 6. Return the consolidated response to the frontend
    return AIResponse(
        session_id=request.session_id,
        ai_message=ai_text_response,
        images=image_previews,
        conversation_state=new_state
    )


@router.post("/upload-image")
async def upload_user_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """
    Handles user-uploaded images (e.g., a logo for placement).
    Saves to temp storage and returns the URL.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
        
    path = await save_temp_image(file)
    filename = os.path.basename(path)
    return {
        "status": "success",
        "filename": filename,  # e.g., "uuid.png"
        "file_url": f"/api/ai/image/{filename}" # Full path
    }


@router.post("/place-image")
async def place_image_on_design(
    base_filename: str = Form(...),
    overlay_filename: str = Form(...),
    position: str = Form("center"),
    current_user: User = Depends(get_current_user),
):
    """
    Combines a user-uploaded image (overlay) with a generated outfit image (base).
    """
    base_path = os.path.join(UPLOAD_DIR, base_filename)
    overlay_path = os.path.join(UPLOAD_DIR, overlay_filename)

    if not os.path.exists(base_path) or not os.path.exists(overlay_path):
        raise HTTPException(status_code=404, detail="One or both image files not found in temp storage.")

    try:
        result_path = place_uploaded_image(base_path, overlay_path, position)
        filename = os.path.basename(result_path)
        return {
            "status": "success",
            "filename": filename,
            "image_url": f"/api/ai/image/{filename}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to place image: {e}")


@router.get("/image/{filename}")
async def get_image(filename: str):
    """Serves generated or uploaded images stored in the temp folder."""
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
        
    # Prevent path traversal
    if ".." in filename or filename.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid filename")
        
    return FileResponse(file_path)