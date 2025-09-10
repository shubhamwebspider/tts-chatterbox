from fastapi import APIRouter, HTTPException, Request, File, UploadFile, Form, Query
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from pydantic import BaseModel, Field
import os
import io
import shutil
import soundfile as sf
from pathlib import Path
import asyncio
import torch

router = APIRouter(prefix="/voice_clone", tags=["Voice Clone"])
VOICE_PROFILE_PATH = Path("voice_profiles")
VOICE_PROFILE_PATH.mkdir(parents=True, exist_ok=True)

# Concurrency control
MAX_CONCURRENT_TTS = 5
MAX_CONCURRENT_VC = 5
tts_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TTS)
vc_semaphore = asyncio.Semaphore(MAX_CONCURRENT_VC)

# Text input class
class TextInput(BaseModel):
    text: str = Field(..., description="The text to be converted to speech.")
    profile_name: str = Field("harry/sophie", description="User/profile name to select the voice.")
    # profile_id: str = Field(..., description="User/profile ID to select the voice.")
    exaggeration: float = Field(0.5, ge=0.25, le=2.0, description="How expressive should the speech be? (0.5 is neutral)")
    cfg_weight: float = Field(0.5, ge=0.0, le=1.0, description="Clarity vs. Pace. Higher values can be clearer but slower.")
    temperature: float = Field(0.8, ge=0.05, le=5.0, description="Creativity / Variability")
    seed_num: int = Field(0, description="Seed. 0 for random. Same number gives the same result.")
    min_p: float = Field(0.05, ge=0.0, le=1.0, description="Confidence. Advanced: Makes the model more confident. Default is good.")
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Randomness. Advanced: Sets randomness to the voice. Default is good.")
    repetition_penalty: float = Field(1.2, ge=1.0, le=2.0, description="Repetition Penalty. Advanced: Checks repetition. Default is good.")

# Run tts_model.generate in a thread pool with concurrency guard
async def tts_generate_stream(tts_model, text_input: TextInput, cond_path: str):
    """
    Async generator that yields WAV chunks as they're generated.
    """
    async with tts_semaphore:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[bytes] = asyncio.Queue()

        def _worker():
            buffer = io.BytesIO()
            with torch.inference_mode():
                try:
                    for audio_chunk, _ in tts_model.generate_stream(
                        text=text_input.text,
                        cond_path=cond_path,
                        chunk_size=25,
                        exaggeration=0.5,
                        temperature=0.8,
                        cfg_weight=0.5,
                    ):
                        audio_np = audio_chunk.squeeze().cpu().numpy()
                        buffer.write((audio_np * 32767).astype("int16").tobytes())

                        # push data to async queue
                        buffer.seek(0)
                        data = buffer.read()
                        buffer.seek(0)
                        buffer.truncate(0)
                        asyncio.run_coroutine_threadsafe(queue.put(data), loop)
                except Exception as e:
                    print(f"Error during streaming: {e}")

            # Final flush
            buffer.seek(0)
            remaining = buffer.read()
            if remaining:
                asyncio.run_coroutine_threadsafe(queue.put(remaining), loop)

            # Signal end of stream
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        # Run worker in a background thread
        loop.run_in_executor(None, _worker)

        # Yield from the async queue
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk


# Run tts_model.generate in a thread pool with concurrency guard
async def tts_generate(tts_model, text_input: TextInput, cond_path: str):
    """Run tts_model.generate with cached conditionals"""
    loop = asyncio.get_event_loop()
    async with tts_semaphore:
        def _blocking():
            with torch.inference_mode():
                wav = tts_model.generate(
                    text_input.text,
                    audio_prompt_path=None,  # Don't recompute - use cached
                    cond_path=cond_path,     # Use cached conditionals
                    exaggeration=text_input.exaggeration,
                    temperature=text_input.temperature,
                    cfg_weight=text_input.cfg_weight,
                    min_p=text_input.min_p,
                    top_p=text_input.top_p,
                    repetition_penalty=text_input.repetition_penalty,
                )
                buf = io.BytesIO()
                sf.write(buf, wav.squeeze(0).cpu().numpy(), tts_model.sr, format='WAV')
                buf.seek(0)
                return buf
        return await loop.run_in_executor(None, _blocking)
    
# Run vc_model in a thread pool with concurrency guard
async def run_vc_generate(vc_model, input_file: UploadFile, cond_path: str):
    """Run vc_model in a thread pool with concurrency guard."""
    loop = asyncio.get_event_loop()
    async with vc_semaphore:
        def _blocking():
            with torch.inference_mode():
                wav = vc_model.generate(input_file.file, cond_path=cond_path)
                buf = io.BytesIO()
                sf.write(buf, wav.squeeze(0).cpu().numpy(), vc_model.sr, format='WAV')
                buf.seek(0)
                return buf
        return await loop.run_in_executor(None, _blocking)


# ------------- Routes -------------
@router.post("/text_to_speech_stream")
async def text_to_speech(text: TextInput, request: Request):
    try:
        tts_model = request.app.state.tts_model
        if tts_model is None:
            raise HTTPException(status_code=503, detail="TTS model not loaded.")

        profiles = getattr(request.app.state, "voice_profiles", {})
        # profile_id = {"harry": "1", "sophie": "2"}.get(text.profile_name.lower(), "2")
        profile_data = profiles.get(text.profile_name.lower(), "sophie")
        cond_path = profile_data.get("cond_path")

        if not cond_path or not os.path.exists(cond_path):
            raise HTTPException(status_code=404, detail="Voice profile conditionals not found.")

        return StreamingResponse(
            tts_generate_stream(tts_model, text, cond_path),
            media_type="audio/wav"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.post("/text_to_speech")
async def text_to_speech(text: TextInput, request: Request):
    try:
        tts_model = request.app.state.tts_model
        if tts_model is None:
            raise HTTPException(status_code=503, detail="TTS model not loaded.")

        profiles = getattr(request.app.state, "voice_profiles", {})
        # profile_id = {"harry": "1", "sophie": "2"}.get(text.profile_name.lower(), "2")
        profile_data = profiles.get(text.profile_name.lower(), "sophie")
        cond_path = profile_data.get("cond_path")

        if not cond_path or not os.path.exists(cond_path):
            raise HTTPException(status_code=404, detail="Voice profile conditionals not found.")

        buf = await tts_generate(tts_model, text, cond_path)
        return StreamingResponse(buf, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/speech_to_speech")
async def speech_to_speech(request: Request,file: UploadFile = File(...),profile_name: str = Form(...),):
    try:
        vc_model = request.app.state.vc_model
        if vc_model is None:
            raise HTTPException(status_code=503, detail="VC model not loaded.")
        
        profiles = getattr(request.app.state, "voice_profiles", {})
        profile_data = profiles.get(profile_name.lower(), "sophie")

        cond_path = profile_data.get("cond_path")
        if not cond_path or not os.path.exists(cond_path):
            raise HTTPException(status_code=404, detail=f"Voice profile conditionals not found.")

        buf = await run_vc_generate(vc_model, file, cond_path)
        return StreamingResponse(buf, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Fetch voice to play
@router.get("/voice_fetch")
async def voice_fetch(request: Request, profile_name: str = Query(...)):
    try:
        vc_model = request.app.state.vc_model
        if vc_model is None:
            raise HTTPException(status_code=503, detail="VC model not loaded.")
        
        profiles = getattr(request.app.state, "voice_profiles", {})
        profile_data = profiles.get(profile_name.lower(), "sophie")
        audio_path = profile_data.get("audio_path")
        if not audio_path or not os.path.exists(audio_path):
            raise HTTPException(status_code=404, detail=f"Voice profile audio not found.")
        
        return FileResponse(path=audio_path, media_type="audio/wav", filename=os.path.basename(audio_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/voice_save")
async def voice_save(request: Request,file: UploadFile = File(...),profile_name: str = Form(...),):
    if not file.filename.endswith((".wav", ".mp3", ".flac", ".aac")):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    tts_model = request.app.state.tts_model
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded.")

    # Save user-specific prompt
    user_dir = VOICE_PROFILE_PATH / profile_name
    user_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename).suffix or ".wav"
    audio_path = user_dir / f"active_prompt{ext}"
    
    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Pre-compute and save conditionals ONCE during save
    try:
        print(f"Pre-computing conditionals for profile {profile_name}...")
        # Generate conditionals once and save them
        cond_path = user_dir / "conditionals.pt"
        tts_model.prepare_conditionals(str(audio_path), exaggeration=0.5)
        tts_model.conds.save(cond_path)
        print(f"Conditionals saved to {cond_path}")
        
        # Update app state with both paths
        if not hasattr(request.app.state, "voice_profiles"):
            request.app.state.voice_profiles = {}
        request.app.state.voice_profiles[profile_name] = {
            "audio_path": str(audio_path),
            "cond_path": str(cond_path)
        }

        payload = {
            "status": 10, 
            "message": "Voice profile saved with precomputed conditionals", 
            "profile_name": profile_name
        }
        return JSONResponse(content=payload, status_code=200)
        
    except Exception as e:
        # Cleanup on failure
        if audio_path.exists():
            audio_path.unlink()
        cond_path = user_dir / "conditionals.pt"
        if cond_path.exists():
            cond_path.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to process voice: {str(e)}")