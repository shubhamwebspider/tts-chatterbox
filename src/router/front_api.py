from fastapi import APIRouter,Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse


router = APIRouter(prefix="/voice_clone_front", tags=["Voice Clone Front"])

# --- Mount templates ---
templates = Jinja2Templates(directory="src/templates")

@router.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return templates.TemplateResponse("front.html", {"request": request})