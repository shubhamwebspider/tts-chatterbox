from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from src.chatterbox.tts import ChatterboxTTS
from src.chatterbox.vc import ChatterboxVC
from src.router import api,front_api
import torch
from contextlib import asynccontextmanager
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VOICE_PROFILE_PATH = BASE_DIR / "voice_profiles"

@asynccontextmanager
async def lifespan_manager(app: FastAPI):
    print("Application starting up...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    app.state.tts_model = ChatterboxTTS.from_pretrained(device)
    app.state.vc_model = ChatterboxVC.from_pretrained(device)
    
    # Load existing voice profiles from disk
    app.state.voice_profiles = {}
    if VOICE_PROFILE_PATH.exists():
        for user_dir in VOICE_PROFILE_PATH.iterdir():
            if user_dir.is_dir():
                profile_id = user_dir.name
                audio_path = user_dir / "active_prompt.wav"  # Check for .wav first
                if not audio_path.exists():
                    # Check for other extensions
                    for ext in [".mp3", ".flac", ".aac"]:
                        audio_path = user_dir / f"active_prompt{ext}"
                        if audio_path.exists():
                            break
                
                cond_path = user_dir / "conditionals.pt"
                
                if audio_path.exists():
                    profile_data = {"audio_path": str(audio_path)}
                    if cond_path.exists():
                        profile_data["cond_path"] = str(cond_path)
                    app.state.voice_profiles[profile_id] = profile_data
                    print(f"Loaded voice profile: {profile_id}")
    
    print(f"TTS model loaded on device: {device}")
    print("Current working directory:", os.getcwd())
    print("VOICE_PROFILE_PATH:", VOICE_PROFILE_PATH.resolve())
    print("Voice Profile Path Exists:", VOICE_PROFILE_PATH.exists())
    print(f"Loaded {len(app.state.voice_profiles)} voice profiles")
    yield
    print("Application shutting down...")

app = FastAPI(lifespan=lifespan_manager)

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include router
app.include_router(api.router)
app.include_router(front_api.router)

@app.get("/")
async def root():
    return {"Server": "Running"}

@app.get("/voice_profiles")
async def list_voice_profiles(request: Request):
    profiles = getattr(request.app.state, "voice_profiles", {})
    return {
        "count": len(profiles),
        "profiles": list(profiles.keys())
    }