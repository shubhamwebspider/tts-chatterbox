from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from src.chatterbox.tts import ChatterboxTTS
from src.chatterbox.vc import ChatterboxVC
from src.router import api,front_api
import torch
from contextlib import asynccontextmanager
import os
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VOICE_PROFILE_PATH = BASE_DIR / "voice_profiles"

@asynccontextmanager
async def lifespan_manager(app: FastAPI):
    print("Application starting up...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    app.state.tts_model = ChatterboxTTS.from_pretrained(device)
    app.state.vc_model = ChatterboxVC.from_pretrained(device)

    # Do not cache voice profiles in memory; always use local directory
    print(f"TTS model loaded on device: {device}")
    print("Current working directory:", os.getcwd())
    print("VOICE_PROFILE_PATH:", VOICE_PROFILE_PATH.resolve())
    print("Voice Profile Path Exists:", VOICE_PROFILE_PATH.exists())
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
    profiles = []
    if VOICE_PROFILE_PATH.exists():
        for user_dir in VOICE_PROFILE_PATH.iterdir():
            if user_dir.is_dir():
                profiles.append(user_dir.name)
    return {
        "count": len(profiles),
        "profiles": profiles
    }

@app.get("/voice_profiles/{profile_name}/audio")
async def get_profile_audio(profile_name: str, request: Request):
    user_dir = VOICE_PROFILE_PATH / profile_name
    if not user_dir.exists() or not user_dir.is_dir():
        return {"error": "Profile not found"}, 404

    # Look for supported audio files in order
    audio_path = user_dir / "active_prompt.wav"
    if not audio_path.exists():
        for ext in [".mp3", ".flac", ".aac"]:
            candidate = user_dir / f"active_prompt{ext}"
            if candidate.exists():
                audio_path = candidate
                break

    if not audio_path.exists():
        return {"error": "Audio not found"}, 404

    # Serve with a reasonable default type; browser can often auto-detect
    media_type = "audio/wav" if audio_path.suffix.lower() == ".wav" else "application/octet-stream"
    return FileResponse(str(audio_path), media_type=media_type)

@app.delete("/voice_profiles/{profile_name}")
async def delete_profile(profile_name: str, request: Request):
    profile_dir = VOICE_PROFILE_PATH / profile_name
    if not profile_dir.exists() or not profile_dir.is_dir():
        return {"error": "Profile not found"}, 404
    shutil.rmtree(profile_dir)
    return {"message": "Profile deleted"}