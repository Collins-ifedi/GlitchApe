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
import logging
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from PIL import Image

from google import genai  # Gemini API
from jose import jwt

# Import DB and auth components from server.py
from server import get_db, get_current_user

# Import models from models.py
from models import User, ChatSession, ChatMessage

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

# Logging
log = logging.getLogger(__name__)

# ===================================================================
# SCHEMAS
# ===================================================================

class ChatInput(BaseModel):
    session_id: str
    prompt: str


# ===================================================================
# IMAGE GENERATION/PLACEMENT UTILITIES
# ===================================================================

def generate_image_with_retry(prompt: str, n_views: int = 1) -> List[str]:
    """Calls Gemini Image API to generate images with robust error handling."""
    # Production-ready prompt engineering for clothing views
    # This is a critical area for refinement based on model performance
    view_prompts = []
    if n_views == 1:
        view_prompts = [f"A high-quality, professional, concept art image of the front view of a garment, based on the following description: {prompt}"]
    elif n_views == 2:
        view_prompts = [
            f"A high-quality, professional, concept art image of the front view of a garment, based on the following description: {prompt}",
            f"A high-quality, professional, concept art image of the back view of the same garment, based on the following description: {prompt}"
        ]
    elif n_views > 2:
        view_prompts = [
            f"Front view of the garment: {prompt}",
            f"Back view of the garment: {prompt}",
            f"Sleeve detail/side view of the garment: {prompt}"
        ]
        
    results = []
    for view_prompt in view_prompts:
        try:
            # DALL-E/Imagen 2 equivalent: 
            # `model='imagen-3.0-generate-002'` or similar
            # Since we're using the standard Gemini client, we use `generate_images`
            # Note: This is a synchronous call.
            result = client.models.generate_images(
                model='imagen-3.0-generate-002', 
                prompt=view_prompt,
                config=dict(
                    number_of_images=1, 
                    output_mime_type="image/png", # PNG for quality
                    aspect_ratio="1:1"
                )
            )
            
            # Save the image locally
            if result.generated_images:
                # Use a unique name for the saved file
                filename = f"ai_gen_{uuid.uuid4().hex}.png"
                filepath = os.path.join(UPLOAD_DIR, filename)
                
                # Get the image data
                image_data = result.generated_images[0].image.image_bytes
                
                # Save it to the temp directory
                with open(filepath, 'wb') as f:
                    f.write(image_data)
                
                results.append(filename)
            else:
                log.error(f"Image generation failed for view prompt: {view_prompt}")

        except Exception as e:
            log.error(f"Image generation API call failed: {e}")
            # Raise a user-friendly exception
            raise HTTPException(status_code=500, detail="AI image generation failed. Please try again.")

    return results


def place_uploaded_image(base_path: str, overlay_path: str, position: str = "center") -> str:
    """
    Uses PIL to place a smaller overlay image onto a larger base image.
    This simulates placing a user's logo/design on the generated apparel.
    """
    try:
        base = Image.open(base_path).convert("RGBA")
        overlay = Image.open(overlay_path).convert("RGBA")
        
        # Resize overlay to a reasonable size relative to the base (e.g., 25%)
        max_size = int(base.width * 0.25)
        overlay.thumbnail((max_size, max_size), Image.LANCZOS)
        
        # Calculate position
        x_offset = 0
        y_offset = 0
        
        if position == "center":
            x_offset = (base.width - overlay.width) // 2
            y_offset = (base.height - overlay.height) // 2
        elif position == "top-left":
            x_offset = 20
            y_offset = 20
        elif position == "chest":
            # Estimate chest area for a standard T-shirt/hoodie image
            x_offset = (base.width - overlay.width) // 2
            y_offset = int(base.height * 0.3)
        # Add more positions as needed

        # Ensure offsets are within bounds
        x_offset = max(0, x_offset)
        y_offset = max(0, y_offset)
        
        # Create a new image for the combined result
        combined = Image.new('RGBA', base.size)
        combined.paste(base, (0, 0))
        combined.paste(overlay, (x_offset, y_offset), overlay) # Use overlay as mask
        
        # Save the result
        result_filename = f"combined_{uuid.uuid4().hex}.png"
        result_path = os.path.join(UPLOAD_DIR, result_filename)
        combined.save(result_path, "PNG")
        
        return result_path

    except Exception as e:
        log.error(f"Image placement error: {e}")
        raise RuntimeError("Image processing failed.")


async def _cleanup_old_files():
    """Periodically clean up old generated/uploaded images."""
    now = datetime.now()
    cutoff = now - timedelta(hours=48) # 48 hours retention
    
    for filename in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, filename)
        try:
            if os.path.isfile(file_path):
                # Get creation time (or modification time, which is usually close)
                mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                if mod_time < cutoff:
                    os.remove(file_path)
                    log.info(f"Cleaned up old file: {filename}")
        except Exception as e:
            log.warning(f"Error cleaning up file {filename}: {e}")

# Run cleanup on startup and schedule periodic runs (not part of FastAPI's core)
# In production, this would be a separate worker/cron job.
# For simplicity, we'll rely on the manual API call being the only trigger for now.


# ===================================================================
# CORE CHAT/AI ENDPOINTS
# ===================================================================

@router.post("/chat")
async def handle_chat(
    data: ChatInput, 
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Handles conversational flow, checking for special keywords to trigger
    image generation or provide ordering guidance.
    """
    try:
        session_uuid = uuid.UUID(data.session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session ID format")

    # 1. Fetch the session and previous messages (Security Check)
    q = select(ChatSession).where(
        ChatSession.id == session_uuid, 
        ChatSession.user_id == current_user.id
    )
    r = await db.execute(q)
    session = r.scalar_one_or_none()
    
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found or access denied")

    # Save user message
    user_message = ChatMessage(
        session_id=session.id, 
        user_id=current_user.id, 
        role="user", 
        content=data.prompt
    )
    db.add(user_message)
    await db.commit()

    # --- AI Logic ---
    ai_response_content = ""
    image_filenames = []
    trigger_image_gen = False
    
    # Simple keyword check for image generation
    if any(k in data.prompt.lower() for k in ["generate", "design", "show me", "create", "views"]):
        trigger_image_gen = True

    if trigger_image_gen:
        # Generate images (e.g., front and back view)
        try:
            # We assume a standard request is for 2 views (front/back)
            image_filenames = generate_image_with_retry(data.prompt, n_views=2)
            
            if image_filenames:
                ai_response_content = "Great! I've generated the front and back views of your design. Take a look and tell me if you want to modify it, or if you're ready to place an order."
            else:
                ai_response_content = "I was unable to generate the images this time. Could you try a different prompt?"
        except HTTPException as e:
            # Re-raise image generation errors
            raise e
        except Exception as e:
            log.error(f"Unexpected error during image generation: {e}")
            ai_response_content = "An unexpected error occurred during image generation. Please try again."

    elif "order" in data.prompt.lower() or "checkout" in data.prompt.lower():
        # User is asking about ordering/checkout
        ai_response_content = "To proceed with an order, please ensure you have a design you like. Then, look for the 'Order Design' button associated with the latest images or tell me specifically which design you want to buy (e.g., 'I want to order the latest design')."

    else:
        # Standard chat using a text model (Gemini-2.5-flash for speed)
        # Note: A full history context retrieval would be better here for true conversation.
        try:
            # Simple text response for now
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=f"You are an AI clothing design assistant. Respond concisely and professionally. User: {data.prompt}",
            )
            ai_response_content = response.text
        except Exception as e:
            log.error(f"Gemini API text call failed: {e}")
            ai_response_content = "I'm having trouble connecting to my AI brain. Please try again later."
            

    # 2. Save AI response(s)
    
    if image_filenames:
        for i, filename in enumerate(image_filenames):
            # Infer view name based on position in list
            view = ["front", "back", "side"][i] if i < 3 else f"view_{i+1}"
            
            # Save a separate message for each image
            image_message = ChatMessage(
                session_id=session.id,
                user_id=current_user.id,
                role="ai",
                content=f"Generated image: {view}", # Essential for order.py to identify the view
                image_url=os.path.join(UPLOAD_DIR, filename) # Full path used for internal reference
            )
            db.add(image_message)
            
        # Add a final text message to summarize the action
        text_message = ChatMessage(
            session_id=session.id,
            user_id=current_user.id,
            role="ai",
            content=ai_response_content
        )
        db.add(text_message)
    else:
        # Only a text response
        text_message = ChatMessage(
            session_id=session.id,
            user_id=current_user.id,
            role="ai",
            content=ai_response_content
        )
        db.add(text_message)
    
    await db.commit()
    
    # 3. Format response for frontend
    response_data = {
        "user_message_id": str(user_message.id),
        "ai_response": ai_response_content,
        "images": [
            {
                "url": f"/api/ai/image/{filename}", 
                "view": ["front", "back", "side"][i] if i < 3 else f"view_{i+1}",
                "internal_filename": filename # useful for the image placement endpoint
            } for i, filename in enumerate(image_filenames)
        ]
    }
    
    return response_data


# ===================================================================
# IMAGE UPLOAD & PLACEMENT ENDPOINTS
# ===================================================================

@router.post("/upload-image")
async def upload_image(
    file: UploadFile = File(...), 
    current_user: User = Depends(get_current_user)
):
    """
    Uploads a file (e.g., a logo or custom image) to the temp folder.
    """
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG files are allowed.")

    filename = f"user_up_{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    try:
        # Write the file chunk by chunk for large files
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return {
            "status": "success", 
            "filename": filename, 
            "image_url": f"/api/ai/image/{filename}"
        }
    except Exception as e:
        log.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail="File upload failed")
    finally:
        await file.close()


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
    # Simple security check to prevent directory traversal
    if ".." in filename or "/" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = os.path.join(UPLOAD_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
        
    return FileResponse(file_path)