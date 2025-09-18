from fastapi import APIRouter,Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse


router = APIRouter(prefix="/voice_clone_front", tags=["Voice Clone Front"])

# --- Mount templates ---
templates = Jinja2Templates(directory="src/templates")

@router.get("/", response_class=HTMLResponse)
async def serve_ui_root(request: Request):
    # Redirect root to the generate page
    return RedirectResponse(url="/voice_clone_front/generate")

@router.get("/generate", response_class=HTMLResponse)
async def serve_generate(request: Request):
    return templates.TemplateResponse("tts_generate.html", {"request": request})

@router.get("/save", response_class=HTMLResponse)
async def serve_save(request: Request):
    return templates.TemplateResponse("voice_save.html", {"request": request})