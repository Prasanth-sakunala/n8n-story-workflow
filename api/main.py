"""
FastAPI service for n8n Animated Story YouTube Workflow.

NODE 1 — Story Trigger & Topic Selection:
  GET  /health                       — Health check
  GET  /api/characters               — Get character registry
  POST /api/characters               — Add/update a character
  GET  /api/story-seeds              — Get available story seeds
  POST /api/story-seeds              — Add a new story seed
  POST /api/fetch-story-ideas        — Fetch ideas from Reddit + local seeds
  POST /api/pick-story               — Score & pick the best story idea

CHARACTER CONSISTENCY SYSTEM:
  GET  /api/character-consistency           — Get canonical prompt library (frozen prompts)
  POST /api/character-consistency/validate  — Check a prompt for character drift
  POST /api/generate-reference-sheet        — Generate multi-pose reference sheet for one character
  POST /api/generate-all-reference-sheets   — Generate reference sheets for ALL characters

NODE 2 — Script Generation (Gemini 2.5 Flash → Groq Llama 4 → Ollama cascade):
  POST /api/generate-script          — Generate full episode script with scenes
  GET  /api/llm-status               — Check which LLM providers are available

NODE 3 — Scene Planner (builds generation-ready tasks per scene):
  POST /api/plan-scenes              — Build ComfyUI prompts + audio + timing per scene

NODE 4 — Image & Animation Generation (ComfyUI + AnimateDiff):
  POST /api/generate-visuals         — Generate all keyframes + animate + Ken Burns fallback
  GET  /api/comfyui-status           — Check ComfyUI availability + AnimateDiff nodes

NODE 5 — Audio Generation (Edge-TTS with word-level timestamps):
  POST /api/generate-audio           — Generate per-scene audio + synced subtitles
  GET  /api/tts-voices               — List available Edge-TTS voices
  GET  /api/voice-lock               — Get frozen voice assignments per character
  POST /api/mix-audio                — Mix voice + background music with ducking
  GET  /api/music-library            — List available background music tracks

CINEMATIC SYSTEMS:
  POST /api/plan-transitions         — Plan scene transitions based on emotional flow
  GET  /api/transitions              — Get transition library and emotion mappings
  POST /api/plan-motion              — Plan cinematic motion per scene (Ken Burns + parallax)

NODE 6 — Video Assembly (FFmpeg long-form with audio-video sync):
  POST /api/assemble-video           — Assemble scenes into final video with subtitles

NODE 7 — SEO & Metadata Generation (Gemini → Groq → Ollama cascade):
  POST /api/generate-seo             — Generate YouTube title, description, tags, category

NODE 8 — Thumbnail Generation (Pillow text overlay):
  POST /api/generate-thumbnail       — Add title + branding to ComfyUI thumbnail image

NODE 9 — YouTube Upload (YouTube Data API v3 + OAuth2):
  GET  /api/youtube-auth-url         — Get OAuth2 authorization URL
  POST /api/youtube-auth-callback    — Exchange auth code for token
  GET  /api/youtube-auth-status      — Check auth status
  POST /api/upload-youtube           — Upload video + thumbnail + SEO metadata
  POST /api/youtube-playlist-add     — Add video to series playlist

EPISODE CONTINUITY:
  GET  /api/continuity               — Get story state for next episode
  POST /api/continuity               — Record continuity after episode completion
"""

import asyncio
import json
import os
import random
import shutil
import subprocess
import time
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
import re

from fastapi import FastAPI, HTTPException
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel

# ============================================
# CONFIG
# ============================================
DATA_DIR = os.environ.get("DATA_DIR", "/data")
COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://host.docker.internal:8188")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434")
SADTALKER_URL = os.environ.get("SADTALKER_URL", "http://host.docker.internal:8189")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
POLLINATIONS_API_KEY = os.environ.get("POLLINATIONS_API_KEY", "")  # Primary image generation (free tier)
POLLINATIONS_MODEL = os.environ.get("POLLINATIONS_MODEL", "flux")  # flux | seedream | nanobanana
COMFYUI_CHECKPOINT = os.environ.get("COMFYUI_CHECKPOINT", "")  # Optional ComfyUI fallback checkpoint

# Font paths (Liberation Sans installed via Dockerfile — free Arial equivalent)
FONT_REGULAR = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"

CHARACTERS_FILE = os.path.join(DATA_DIR, "characters", "characters.json")
STORY_SEEDS_FILE = os.path.join(DATA_DIR, "stories", "story_seeds.json")
STORY_HISTORY_FILE = os.path.join(DATA_DIR, "stories", "history.json")
CONTINUITY_FILE = os.path.join(DATA_DIR, "stories", "continuity.json")

# Bundled defaults (copied to DATA_DIR on first run)
BUNDLED_CHARACTERS = os.path.join(os.path.dirname(__file__), "data", "characters.json")
BUNDLED_SEEDS = os.path.join(os.path.dirname(__file__), "data", "story_seeds.json")


# ============================================
# SECTION 2 — GLOBAL VISUAL STYLE LOCK
# This style is automatically appended to EVERY image generation prompt.
# It ensures visual consistency across all episodes, scenes, and emotions.
# ============================================
GLOBAL_STYLE_PROMPT = (
    "cinematic illustrated children's storybook, "
    "high-end animated movie concept art, "
    "soft volumetric lighting, warm cinematic lighting, "
    "stylized cartoon realism, clean lineart, "
    "consistent character design, storybook color palette, "
    "emotionally expressive faces, Pixar-inspired composition, "
    "high-quality cinematic framing, "
    "detailed environmental storytelling"
)

GLOBAL_NEGATIVE_PROMPT = (
    "photorealistic, photograph, 3d render, anime, "
    "inconsistent art style, low quality, blurry, "
    "deformed, extra limbs, bad anatomy, ugly, "
    "watermark, text, signature, disfigured, "
    "live action, real person, random style change"
)


# ============================================
# SECTION 6 — PROVIDER ABSTRACTION LAYER
# Provider-agnostic generation interfaces for future-proofing.
# Swap providers without touching workflow logic.
# ============================================

class ImageProvider:
    """Abstract interface for image generation providers."""
    name: str = "base"

    def generate(self, prompt: str, output_path: str, **kwargs) -> dict:
        """Generate image. Returns {"success": bool, "path": str, "error": str}."""
        raise NotImplementedError

    def is_available(self) -> bool:
        raise NotImplementedError


class PollinationsProvider(ImageProvider):
    """Pollinations.ai with FLUX/Seedream — primary provider (free tier, no key required)."""
    name = "pollinations"

    def is_available(self) -> bool:
        return True  # Always available — free tier works without API key

    def generate(self, prompt: str, output_path: str, **kwargs) -> dict:
        full_prompt = f"{GLOBAL_STYLE_PROMPT}, {prompt}"
        negative = kwargs.get("negative_prompt", GLOBAL_NEGATIVE_PROMPT)
        width = kwargs.get("width", 1024)
        height = kwargs.get("height", 576)
        model = kwargs.get("model", POLLINATIONS_MODEL)
        return _generate_pollinations(full_prompt, output_path, width, height, model, negative)


# Imagen3Provider removed — requires paid tier, use Pollinations (free) instead


class ComfyUIProvider(ImageProvider):
    """ComfyUI — fallback provider (only used when Gemini fails)."""
    name = "comfyui"

    def is_available(self) -> bool:
        return _comfyui_available()

    def generate(self, prompt: str, output_path: str, **kwargs) -> dict:
        task = {
            "id": kwargs.get("id", "scene"),
            "prompt": f"{GLOBAL_STYLE_PROMPT}, {prompt}",
            "negativePrompt": kwargs.get("negative_prompt", GLOBAL_NEGATIVE_PROMPT),
            "width": kwargs.get("width", 1024),
            "height": kwargs.get("height", 576),
            "steps": kwargs.get("steps", 25),
            "cfg": kwargs.get("cfg", 7.0),
            "seed": kwargs.get("seed", -1),
        }
        checkpoint = kwargs.get("checkpoint", COMFYUI_CHECKPOINT or "dreamshaper_8.safetensors")
        sampler = kwargs.get("sampler", "euler_ancestral")
        scheduler = kwargs.get("scheduler", "normal")
        loras = kwargs.get("loras", [])
        wf = _build_keyframe_workflow(task, checkpoint, sampler, scheduler, loras, force_batch_size=1)
        pid = _comfyui_queue(wf)
        if not pid:
            return {"success": False, "path": "", "error": "ComfyUI queue failed"}
        comp = _comfyui_wait(pid, max_wait=300)
        if comp.get("success") and comp.get("images"):
            ok = _comfyui_download(comp["images"][0], output_path)
            return {"success": ok, "path": output_path if ok else "", "error": ""}
        return {"success": False, "path": "", "error": comp.get("error", "No images")}


class ScriptProvider:
    """Abstract interface for LLM script generation."""
    name: str = "base"

    def generate(self, prompt: str, **kwargs) -> dict:
        """Returns {"success": bool, "text": str, "error": str}."""
        raise NotImplementedError

    def is_available(self) -> bool:
        raise NotImplementedError


class GeminiScriptProvider(ScriptProvider):
    name = "gemini"
    def is_available(self) -> bool:
        return bool(GEMINI_API_KEY)
    def generate(self, prompt: str, **kwargs) -> dict:
        model = kwargs.get("model", "gemini-2.5-flash")
        return _call_gemini(prompt, GEMINI_API_KEY, model)


class GroqScriptProvider(ScriptProvider):
    name = "groq"
    def is_available(self) -> bool:
        return bool(GROQ_API_KEY)
    def generate(self, prompt: str, **kwargs) -> dict:
        model = kwargs.get("model", "meta-llama/llama-4-scout-17b-16e-instruct")
        return _call_groq(prompt, GROQ_API_KEY, model)


class OllamaScriptProvider(ScriptProvider):
    name = "ollama"
    def is_available(self) -> bool:
        try:
            urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5)
            return True
        except Exception:
            return False
    def generate(self, prompt: str, **kwargs) -> dict:
        model = kwargs.get("model", "llama3.1:8b")
        return _call_ollama(prompt, model)


# Provider registry — order defines priority (first available wins)
IMAGE_PROVIDERS: list[ImageProvider] = [PollinationsProvider(), ComfyUIProvider()]
SCRIPT_PROVIDERS: list[ScriptProvider] = [GeminiScriptProvider(), GroqScriptProvider(), OllamaScriptProvider()]


def _generate_with_fallback(prompt: str, output_path: str, providers: list[ImageProvider] = None, **kwargs) -> dict:
    """Try image providers in priority order. Returns first success."""
    providers = providers or IMAGE_PROVIDERS
    errors = []
    for provider in providers:
        if not provider.is_available():
            errors.append({"provider": provider.name, "error": "not available"})
            continue
        result = provider.generate(prompt, output_path, **kwargs)
        if result["success"]:
            result["provider"] = provider.name
            return result
        errors.append({"provider": provider.name, "error": result.get("error", "unknown")})
    return {"success": False, "path": "", "error": f"All providers failed: {errors}", "provider": "none"}


# ============================================
# SECTION 9 — PRODUCTION STABILITY
# Retry logic, rate limit handling, and fault tolerance.
# ============================================

MAX_RETRIES = 4
RETRY_DELAYS = [5, 15, 30, 60]  # seconds between retries (exponential backoff)


def _retry_with_backoff(fn, max_retries: int = MAX_RETRIES, **kwargs) -> dict:
    """Execute fn with retry + exponential backoff. fn must return dict with 'success' key."""
    last_error = ""
    for attempt in range(max_retries):
        result = fn(**kwargs)
        if result.get("success"):
            if attempt > 0:
                print(f"[Retry] Succeeded on attempt {attempt + 1}")
            return result
        last_error = result.get("error", "unknown")
        if attempt >= max_retries - 1:
            break
        # Check for rate limit (429) or timeout — wait longer
        if "429" in str(last_error) or "rate" in str(last_error).lower():
            wait = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)] * 3
            print(f"[Retry] Rate limited, waiting {wait}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
        elif "timeout" in str(last_error).lower() or "timed out" in str(last_error).lower():
            wait = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)] * 2
            print(f"[Retry] Timeout, waiting {wait}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
        else:
            wait = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
            print(f"[Retry] Failed: {last_error[:200]}, retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
    return {"success": False, "error": f"Failed after {max_retries} attempts: {last_error}"}


app = FastAPI(title="n8n Animated Story API", version="2.0.0")


# ============================================
# JOB TRACKING SYSTEM — Progress monitoring for long-running tasks
# ============================================
_JOBS: dict[str, dict] = {}  # job_id -> {status, progress, total, current_step, result, error, started_at}
_JOBS_LOCK = threading.Lock()


def _create_job(total_steps: int = 0, description: str = "") -> str:
    """Create a new tracked job. Returns job_id."""
    job_id = str(uuid.uuid4())
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "status": "running",
            "progress": 0,
            "total": total_steps,
            "currentStep": "",
            "description": description,
            "startedAt": time.time(),
            "updatedAt": time.time(),
            "result": None,
            "error": None,
        }
    return job_id


def _update_job(job_id: str, progress: int = None, current_step: str = None, **kwargs):
    """Update job progress."""
    with _JOBS_LOCK:
        if job_id in _JOBS:
            if progress is not None:
                _JOBS[job_id]["progress"] = progress
            if current_step is not None:
                _JOBS[job_id]["currentStep"] = current_step
            _JOBS[job_id]["updatedAt"] = time.time()
            for k, v in kwargs.items():
                _JOBS[job_id][k] = v


def _complete_job(job_id: str, result: dict):
    """Mark job as completed with result."""
    with _JOBS_LOCK:
        if job_id in _JOBS:
            _JOBS[job_id]["status"] = "completed"
            _JOBS[job_id]["progress"] = _JOBS[job_id]["total"]
            _JOBS[job_id]["result"] = result
            _JOBS[job_id]["updatedAt"] = time.time()


def _fail_job(job_id: str, error: str):
    """Mark job as failed."""
    with _JOBS_LOCK:
        if job_id in _JOBS:
            _JOBS[job_id]["status"] = "failed"
            _JOBS[job_id]["error"] = error
            _JOBS[job_id]["updatedAt"] = time.time()


@app.get("/api/job-status/{job_id}")
def get_job_status(job_id: str):
    """Poll job progress. Returns current status, progress %, and result when done."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    elapsed = round(time.time() - job["startedAt"], 1)
    pct = round((job["progress"] / job["total"] * 100) if job["total"] > 0 else 0, 1)
    return {
        "jobId": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "total": job["total"],
        "percent": pct,
        "currentStep": job["currentStep"],
        "description": job.get("description", ""),
        "elapsedSeconds": elapsed,
        "result": job["result"] if job["status"] == "completed" else None,
        "error": job["error"] if job["status"] == "failed" else None,
    }


# ============================================
# STARTUP — initialize data files
# ============================================
@app.on_event("startup")
def _init_data_dirs():
    """Copy bundled defaults to DATA_DIR if they don't exist yet."""
    for src, dst_dir, dst_file in [
        (BUNDLED_CHARACTERS, os.path.join(DATA_DIR, "characters"), CHARACTERS_FILE),
        (BUNDLED_SEEDS, os.path.join(DATA_DIR, "stories"), STORY_SEEDS_FILE),
    ]:
        os.makedirs(dst_dir, exist_ok=True)
        if not os.path.exists(dst_file) and os.path.exists(src):
            "dialogue": [],

    # Ensure other dirs exist
    for d in ["images", "audio", "video", "credentials", "clips",
              "music", "thumbnails", "stories"]:
        os.makedirs(os.path.join(DATA_DIR, d), exist_ok=True)

    # Ensure history file
    if not os.path.exists(STORY_HISTORY_FILE):
        with open(STORY_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump({"episodes": []}, f)


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================
# HEALTH
# ============================================
@app.get("/health")
def health():
    """Health check — also reports status of Ollama + ComfyUI."""
    status = {"status": "ok", "ollama": "unknown", "comfyui": "unknown"}
    # Check Ollama
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                status["ollama"] = "ok"
    except Exception:
        status["ollama"] = "unreachable"
    # Check ComfyUI
    try:
        req = urllib.request.Request(f"{COMFYUI_URL}/system_stats", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                status["comfyui"] = "ok"
    except Exception:
        status["comfyui"] = "unreachable"
    return status


# ============================================
# CHARACTER REGISTRY
# ============================================
class CharacterCreate(BaseModel):
    id: str
    name: str
    description: str
    prompt: str
    loraFile: str = ""
    loraWeight: float = 0.8
    ipAdapterRef: str = ""
    role: str = "supporting"
    personality: str = ""
    catchphrase: str = ""
    age: str = ""
    voiceId: str = "en-US-DavisNeural"


@app.get("/api/characters")
def get_characters():
    """Return the full character registry including series config."""
    if not os.path.exists(CHARACTERS_FILE):
        raise HTTPException(status_code=404, detail="Character file not found. Start the container first.")
    return _load_json(CHARACTERS_FILE)


@app.post("/api/characters")
def add_or_update_character(char: CharacterCreate):
    """Add a new character or update an existing one by id."""
    data = _load_json(CHARACTERS_FILE)
    char_dict = char.model_dump()

    # Find existing by id
    existing_idx = next(
        (i for i, c in enumerate(data["characters"]) if c["id"] == char.id),
        None,
    )
    if existing_idx is not None:
        data["characters"][existing_idx] = char_dict
        action = "updated"
    else:
        data["characters"].append(char_dict)
        action = "added"

    _save_json(CHARACTERS_FILE, data)
    return {"success": True, "action": action, "character": char_dict}


class LocationCreate(BaseModel):
    id: str
    name: str
    description: str
    prompt: str


@app.post("/api/locations")
def add_or_update_location(loc: LocationCreate):
    """Add a new location or update an existing one by id."""
    data = _load_json(CHARACTERS_FILE)
    loc_dict = loc.model_dump()

    if "locations" not in data:
        data["locations"] = []

    existing_idx = next(
        (i for i, l in enumerate(data["locations"]) if l["id"] == loc.id),
        None,
    )
    if existing_idx is not None:
        data["locations"][existing_idx] = loc_dict
        action = "updated"
    else:
        data["locations"].append(loc_dict)
        action = "added"

    _save_json(CHARACTERS_FILE, data)
    return {"success": True, "action": action, "location": loc_dict}


# ============================================
# CHARACTER CONSISTENCY SYSTEM
# Ensures every character remains visually identical across ALL scenes.
# Provides: canonical prompt library, reference sheet generation,
# and consistency enforcement for image generation.
# ============================================

# --- Canonical Prompt Library ---
# These are the FROZEN character prompts. They NEVER change between scenes.
# Every image generation call MUST use these exact descriptions.

def _get_canonical_prompts() -> dict:
    """Load the frozen canonical character prompts from the registry.
    Returns {character_id: {prompt, negative, identity_tags, ...}}"""
    if not os.path.exists(CHARACTERS_FILE):
        return {}
    data = _load_json(CHARACTERS_FILE)
    canonical = {}
    for char in data.get("characters", []):
        cid = char["id"]
        canonical[cid] = {
            "id": cid,
            "name": char.get("name", ""),
            "canonicalPrompt": char.get("prompt", ""),
            "canonicalNegative": f"different {char.get('name', '')} design, inconsistent character, changing appearance, wrong outfit, wrong hair color, wrong eye color",
            "identityTags": _extract_identity_tags(char),
            "voiceId": char.get("voiceId", ""),
            "ipAdapterRef": char.get("ipAdapterRef", ""),
            "loraFile": char.get("loraFile", ""),
            "loraWeight": char.get("loraWeight", 0.8),
            # SECTION 1: Extended character memory fields
            "personality": char.get("personality", ""),
            "catchphrase": char.get("catchphrase", ""),
            "speakingStyle": char.get("speakingStyle", ""),
            "emotionalPatterns": char.get("emotionalPatterns", {}),
            "canonicalExpressions": char.get("canonicalExpressions", []),
            "canonicalPoses": char.get("canonicalPoses", []),
            "physicalIdentity": char.get("physicalIdentity", {}),
        }
    return canonical


def _extract_identity_tags(char: dict) -> list[str]:
    """Extract the key visual identity tags that must NEVER change."""
    prompt = char.get("prompt", "")
    desc = char.get("description", "")
    combined = f"{prompt} {desc}".lower()

    tags = []
    # Extract hair info
    hair_colors = ["brown hair", "black hair", "blonde hair", "red hair", "orange hair", "purple hair", "blue hair", "white hair", "pink hair"]
    for hc in hair_colors:
        if hc in combined:
            tags.append(hc)
    hair_styles = ["messy hair", "long hair", "short hair", "ponytail", "braids", "curly hair", "spiky hair"]
    for hs in hair_styles:
        if hs in combined:
            tags.append(hs)

    # Extract eye color
    eye_colors = ["blue eyes", "green eyes", "brown eyes", "yellow eyes", "red eyes", "purple eyes"]
    for ec in eye_colors:
        if ec in combined:
            tags.append(ec)

    # Extract key clothing/accessories from description
    clothing_keywords = ["hoodie", "scarf", "goggles", "belt", "backpack", "badge", "hat", "cape", "boots", "gloves"]
    for kw in clothing_keywords:
        if kw in combined:
            # Get surrounding context for color
            idx = combined.find(kw)
            context = combined[max(0, idx-20):idx+len(kw)+5]
            tags.append(context.strip().strip(",").strip())

    return tags


def _build_canonical_scene_prompt(
    character_ids: list[str],
    scene_context: dict,
    style: str = "",
) -> str:
    """Build a scene prompt with FROZEN canonical character descriptions.
    
    This is the SINGLE SOURCE OF TRUTH for how characters appear in images.
    It guarantees character identity never drifts across scenes.
    GLOBAL_STYLE_PROMPT is always prepended for visual consistency.
    
    Args:
        character_ids: List of character IDs in the scene
        scene_context: Dict with keys: emotion, cameraAngle, location, visualDescription
        style: Series visual style string (merged with global style)
    
    Returns:
        Complete positive prompt with characters described canonically
    """
    canonical = _get_canonical_prompts()
    if not style and os.path.exists(CHARACTERS_FILE):
        data = _load_json(CHARACTERS_FILE)
        style = data.get("series", {}).get("style", "")

    parts = []

    # 1. GLOBAL STYLE LOCK — always first, always present
    parts.append(f"(({GLOBAL_STYLE_PROMPT}:1.2))")

    # 2. Series-specific style (if different from global)
    if style and style not in GLOBAL_STYLE_PROMPT:
        parts.append(f"({style}:1.0)")

    # 2. Camera angle
    camera = scene_context.get("cameraAngle", "medium_shot")
    camera_keywords = {
        "wide_shot": "wide angle shot, full body visible, establishing shot",
        "medium_shot": "medium shot, waist up, conversational framing",
        "close_up": "close-up shot, face detail, expressive, portrait",
        "over_shoulder": "over the shoulder shot, depth, perspective",
        "birds_eye": "birds eye view, top-down angle",
        "low_angle": "low angle shot, dramatic, looking up",
    }
    parts.append(camera_keywords.get(camera, "medium shot"))

    # 3. CANONICAL character prompts (FROZEN — never modified)
    for cid in character_ids:
        if cid in canonical:
            cp = canonical[cid]["canonicalPrompt"]
            name = canonical[cid]["name"]
            # High weight ensures character identity dominates
            parts.append(f"(({cp}:1.3))")
            parts.append(f"({name}:1.1)")

    # 4. Emotion expression
    emotion = scene_context.get("emotion", "neutral")
    emotion_map = {
        "happy": "happy expression, smiling, joyful",
        "sad": "sad expression, downcast eyes, melancholy",
        "scared": "scared expression, wide eyes, fearful",
        "excited": "excited expression, energetic, dynamic pose",
        "curious": "curious expression, tilted head, inquisitive",
        "determined": "determined expression, confident stance, focused",
        "surprised": "surprised expression, open mouth, wide eyes",
        "angry": "angry expression, furrowed brows, intense",
        "neutral": "calm expression, relaxed pose",
    }
    parts.append(emotion_map.get(emotion, "expressive"))

    # 5. Scene visual description (from script)
    visual = scene_context.get("visualDescription", "")
    if visual:
        parts.append(visual)

    # 6. Location
    location = scene_context.get("location", "")
    if location:
        parts.append(f"setting: {location}")

    # 7. Quality tags
    parts.append("detailed background, sharp focus, vibrant colors, high quality, masterpiece")

    return ", ".join(p for p in parts if p)


def _build_canonical_negative(character_ids: list[str], base_negative: str = "") -> str:
    """Build negative prompt that specifically prevents character drift."""
    canonical = _get_canonical_prompts()

    neg_parts = []
    if base_negative:
        neg_parts.append(base_negative)
    else:
        neg_parts.append(
            "realistic, photo, 3d render, photorealistic, blurry, deformed, "
            "extra limbs, bad anatomy, ugly, distorted face, low quality, "
            "noisy, text, watermark, signature, live action, real person"
        )

    # Add per-character anti-drift negatives
    for cid in character_ids:
        if cid in canonical:
            neg_parts.append(canonical[cid]["canonicalNegative"])

    # Universal consistency negatives
    neg_parts.append(
        "inconsistent character design, changing appearance between frames, "
        "different outfit, different hair, different eyes, model sheet errors"
    )

    return ", ".join(neg_parts)


@app.get("/api/character-consistency")
def get_character_consistency():
    """Return the full canonical prompt library for all characters.
    Use this to verify what prompts will be injected into every scene."""
    canonical = _get_canonical_prompts()
    return {
        "success": True,
        "characters": canonical,
        "totalCharacters": len(canonical),
        "note": "These prompts are FROZEN. They are injected identically into every scene image generation.",
    }



@app.post("/api/character-consistency/validate")
def validate_scene_prompt(data: dict = {}):
    """Validate that a scene prompt contains all required canonical character elements.
    
    Send a prompt string + character_ids to check if any identity tags are missing.
    Useful for debugging consistency issues.
    """
    prompt = data.get("prompt", "")
    character_ids = data.get("characterIds", [])

    if not prompt:
        raise HTTPException(status_code=400, detail="No prompt provided")

    canonical = _get_canonical_prompts()
    prompt_lower = prompt.lower()

    results = []
    all_pass = True

    for cid in character_ids:
        if cid not in canonical:
            results.append({"characterId": cid, "status": "not_found"})
            all_pass = False
            continue

        char = canonical[cid]
        missing_tags = []
        for tag in char["identityTags"]:
            if tag.lower() not in prompt_lower:
                missing_tags.append(tag)

        # Check if canonical prompt is present
        has_canonical = char["canonicalPrompt"].lower()[:50] in prompt_lower

        status = "pass" if (not missing_tags and has_canonical) else "drift_detected"
        if status == "drift_detected":
            all_pass = False

        results.append({
            "characterId": cid,
            "name": char["name"],
            "status": status,
            "hasCanonicalPrompt": has_canonical,
            "missingIdentityTags": missing_tags,
            "identityTags": char["identityTags"],
        })

    return {
        "success": True,
        "allConsistent": all_pass,
        "results": results,
    }


class ReferenceSheetRequest(BaseModel):
    """Request to generate a character reference sheet (multiple poses/angles)."""
    characterId: str
    poses: list[str] = ["front_view", "three_quarter", "side_view", "back_view", "expression_happy", "expression_sad"]
    width: int = 512
    height: int = 512
    steps: int = 25
    cfg: float = 7.5
    checkpoint: str = ""  # ComfyUI fallback checkpoint (leave empty to use COMFYUI_CHECKPOINT env)
    sampler: str = "euler_ancestral"
    scheduler: str = "normal"
    useImagen: bool = False  # Deprecated; Imagen 3 is intentionally disabled


@app.post("/api/generate-reference-sheet")
def generate_reference_sheet(req: ReferenceSheetRequest):
    """
    Generate a multi-pose reference sheet for a character.
    
    Strategy: Pollinations primary (free tier) -> ComfyUI fallback.
    Creates individual images for each pose/angle, saves them to /data/characters/,
    and sets the first image as the IP-Adapter reference for future consistency.
    
    This should be run ONCE per character to establish their visual canon.
    """
    canonical = _get_canonical_prompts()
    if req.characterId not in canonical:
        raise HTTPException(status_code=404, detail=f"Character '{req.characterId}' not found in registry")

    char = canonical[req.characterId]
    char_prompt = char["canonicalPrompt"]
    char_name = char["name"]

    # Pollinations is PRIMARY (free, no key needed), ComfyUI is fallback
    has_pollinations = True  # Always available — free tier
    has_imagen = False  # Imagen 3 removed (paid tier only)
    # ComfyUI is optional fallback only
    comfyui_online = _comfyui_available() and bool(req.checkpoint or COMFYUI_CHECKPOINT)

    # Load series style
    series_style = ""
    if os.path.exists(CHARACTERS_FILE):
        data = _load_json(CHARACTERS_FILE)
        series_style = data.get("series", {}).get("style", "")
        neg_prompt = data.get("series", {}).get("negativePrompt", "")
    else:
        neg_prompt = ""

    # Pose-specific prompt modifiers
    pose_modifiers = {
        "front_view": "front view, facing viewer, symmetrical, full body, standing pose, character sheet",
        "three_quarter": "three-quarter view, slight angle, full body, natural pose, character sheet",
        "side_view": "side view, profile, full body, standing pose, character sheet",
        "back_view": "back view, rear angle, full body, standing pose, character sheet",
        "expression_happy": "front view, close-up face, happy expression, big smile, joyful, character sheet",
        "expression_sad": "front view, close-up face, sad expression, downcast eyes, melancholy, character sheet",
        "expression_surprised": "front view, close-up face, surprised expression, wide eyes, open mouth, character sheet",
        "expression_angry": "front view, close-up face, angry expression, furrowed brows, character sheet",
        "expression_scared": "front view, close-up face, scared expression, wide eyes, trembling, character sheet",
        "action_running": "dynamic running pose, side view, motion blur background, action, character sheet",
        "action_jumping": "jumping pose, mid-air, dynamic angle, action, character sheet",
    }

    ref_dir = os.path.join(DATA_DIR, "characters")
    os.makedirs(ref_dir, exist_ok=True)

    generated = []
    errors = []

    for pose in req.poses:
        pose_mod = pose_modifiers.get(pose, f"{pose}, character sheet")

        # Build the reference sheet prompt
        prompt = f"(({char_prompt}:1.4)), {series_style}, {pose_mod}, white background, clean, simple background, character reference, high quality, masterpiece, sharp focus"
        negative = f"{neg_prompt}, complex background, multiple characters, {char['canonicalNegative']}"

        filename = f"{req.characterId}_{pose}.png"
        output_path = os.path.join(ref_dir, filename)
        pose_generated = False

        # --- Try Pollinations first (free tier, high quality FLUX) ---
        if has_pollinations and not pose_generated:
            pollen_prompt = f"{char_prompt}, {series_style}, {pose_mod}, white background, clean simple background, character reference sheet, high quality, sharp focus"
            img_result = _generate_pollinations(
                prompt=pollen_prompt,
                output_path=output_path,
                width=512,
                height=512,
                model=POLLINATIONS_MODEL,
                negative_prompt=negative,
            )
            if img_result["success"]:
                pose_generated = True
                generated.append({
                    "pose": pose,
                    "filename": filename,
                    "path": output_path,
                    "method": "pollinations",
                })
                print(f"[RefSheet] {char_name} / {pose} — Pollinations success")

        # --- Try Imagen 3 as secondary --- (REMOVED — paid tier only)

        # --- Fallback to ComfyUI (only if Pollinations failed) ---
        if not pose_generated and comfyui_online:
            fallback_checkpoint = req.checkpoint or COMFYUI_CHECKPOINT
            print(f"[RefSheet] {char_name} / {pose} — Pollinations failed, trying ComfyUI ({fallback_checkpoint})")
            workflow = _build_reference_image_workflow(
                prompt=prompt,
                negative=negative,
                width=req.width,
                height=req.height,
                steps=req.steps,
                cfg=req.cfg,
                checkpoint=fallback_checkpoint,
                sampler=req.sampler,
                scheduler=req.scheduler,
                filename=filename,
            )

            prompt_id = _comfyui_queue(workflow)
            if not prompt_id:
                if not pose_generated:
                    errors.append({"pose": pose, "error": "Both Pollinations and ComfyUI failed"})
                continue

            result = _comfyui_wait(prompt_id, max_wait=120)
            if not result.get("success"):
                errors.append({"pose": pose, "error": result.get("error", "Unknown error")})
                continue

            gen_images = result.get("images", [])
            if gen_images:
                src_path = gen_images[0]
                if os.path.exists(src_path) and src_path != output_path:
                    shutil.copy2(src_path, output_path)

                generated.append({
                    "pose": pose,
                    "filename": filename,
                    "path": output_path,
                    "method": "comfyui",
                })
            else:
                errors.append({"pose": pose, "error": "No image generated"})

        elif not pose_generated:
            errors.append({"pose": pose, "error": "No image generator available"})

    # Set the front_view as the primary IP-Adapter reference
    front_ref = next((g for g in generated if g["pose"] == "front_view"), None)
    if front_ref:
        _update_character_reference(req.characterId, front_ref["filename"])

    return {
        "success": len(generated) > 0,
        "characterId": req.characterId,
        "characterName": char_name,
        "generated": generated,
        "errors": errors,
        "totalGenerated": len(generated),
        "totalRequested": len(req.poses),
        "ipAdapterRef": front_ref["filename"] if front_ref else char.get("ipAdapterRef", ""),
        "referenceDir": ref_dir,
    }


def _update_character_reference(character_id: str, filename: str):
    """Update the character's ipAdapterRef in the registry."""
    if not os.path.exists(CHARACTERS_FILE):
        return
    data = _load_json(CHARACTERS_FILE)
    for char in data.get("characters", []):
        if char["id"] == character_id:
            char["ipAdapterRef"] = filename
            break
    _save_json(CHARACTERS_FILE, data)


def _build_reference_image_workflow(
    prompt: str,
    negative: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    checkpoint: str,
    sampler: str,
    scheduler: str,
    filename: str,
) -> dict:
    """Build a simple ComfyUI workflow for a single reference image."""
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["1", 1]},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative, "clip": ["1", 1]},
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
                "seed": random.randint(0, 2**32 - 1),
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": 1.0,
            },
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {"images": ["6", 0], "filename_prefix": filename.replace(".png", "")},
        },
    }


@app.post("/api/generate-all-reference-sheets")
def generate_all_reference_sheets(data: dict = {}):
    """Generate reference sheets for ALL registered characters at once.
    
    Optional body params:
        poses: list of pose names (default: front, 3/4, side, expressions)
        checkpoint: model to use
    """
    poses = data.get("poses", ["front_view", "three_quarter", "side_view", "expression_happy", "expression_sad", "expression_surprised"])
    checkpoint = data.get("checkpoint", "")

    canonical = _get_canonical_prompts()
    if not canonical:
        raise HTTPException(status_code=404, detail="No characters registered")

    all_results = []
    for cid in canonical:
        req = ReferenceSheetRequest(
            characterId=cid,
            poses=poses,
            checkpoint=checkpoint,
        )
        result = generate_reference_sheet(req)
        all_results.append(result)

    total_generated = sum(r.get("totalGenerated", 0) for r in all_results)
    total_errors = sum(len(r.get("errors", [])) for r in all_results)

    return {
        "success": total_generated > 0,
        "characters": all_results,
        "totalCharacters": len(all_results),
        "totalImagesGenerated": total_generated,
        "totalErrors": total_errors,
    }


# ============================================
# STORY SEEDS
# ============================================
class StorySeedCreate(BaseModel):
    title: str
    premise: str
    genre: str = "adventure"
    themes: list[str] = []
    characters: list[str] = []
    locations: list[str] = []
    mood: str = ""


@app.get("/api/story-seeds")
def get_story_seeds():
    """Return all story seeds with used/unused status."""
    if not os.path.exists(STORY_SEEDS_FILE):
        raise HTTPException(status_code=404, detail="Story seeds file not found.")
    data = _load_json(STORY_SEEDS_FILE)
    unused = [s for s in data.get("seeds", []) if not s.get("used")]
    return {
        "totalSeeds": len(data.get("seeds", [])),
        "unusedSeeds": len(unused),
        "seeds": data.get("seeds", []),
    }


@app.post("/api/story-seeds")
def add_story_seed(seed: StorySeedCreate):
    """Add a new story seed to the database."""
    data = _load_json(STORY_SEEDS_FILE)
    seeds = data.get("seeds", [])

    new_id = f"seed_{len(seeds) + 1:03d}"
    seed_dict = seed.model_dump()
    seed_dict["id"] = new_id
    seed_dict["used"] = False

    seeds.append(seed_dict)
    data["seeds"] = seeds
    _save_json(STORY_SEEDS_FILE, data)
    return {"success": True, "seed": seed_dict}


# ============================================
# FETCH STORY IDEAS (Reddit + Local Seeds)
# ============================================
class FetchStoryIdeasRequest(BaseModel):
    useReddit: bool = True
    useLocalSeeds: bool = True
    useAI: bool = True  # Generate ideas via Gemini (fallback: Groq)
    redditLimit: int = 25
    preferGenre: str = ""
    maxResults: int = 20
    aiIdeaCount: int = 5  # Number of AI-generated ideas to request


def _fetch_reddit_prompts(subreddits: list[str], limit: int) -> list[dict]:
    """Fetch story prompts from Reddit. Returns list of idea dicts."""
    ideas = []
    subs_str = "+".join(subreddits)
    url = (
        f"https://www.reddit.com/r/{subs_str}.json"
        f"?limit={limit}&t=week&sort=hot"
    )
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "n8n-story-bot/1.0"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return ideas

    for child in data.get("data", {}).get("children", []):
        post = child.get("data", {})
        title = post.get("title", "")
        selftext = post.get("selftext", "")
        ups = post.get("ups", 0)
        sub = post.get("subreddit", "")

        if not title or len(title) < 15:
            continue

        # Skip mod posts, meta, and NSFW
        if post.get("over_18"):
            continue
        flair = (post.get("link_flair_text") or "").lower()
        if flair in ("meta", "modpost", "off topic"):
            continue

        # Clean up [WP] [SP] etc. prefixes
        clean_title = title
        for prefix in ["[WP]", "[SP]", "[EU]", "[CW]", "[RF]", "[TT]", "[PI]", "[OT]"]:
            clean_title = clean_title.replace(prefix, "").strip()

        ideas.append({
            "source": "reddit",
            "subreddit": sub,
            "title": clean_title,
            "premise": selftext[:500] if selftext else clean_title,
            "upvotes": ups,
            "url": f"https://reddit.com{post.get('permalink', '')}",
            "genre": _guess_genre(clean_title, selftext),
        })

    return ideas


def _guess_genre(title: str, text: str) -> str:
    """Rough genre classification from title/text keywords."""
    combined = (title + " " + text).lower()
    if any(w in combined for w in ["dragon", "wizard", "magic", "sword", "quest", "kingdom", "elf"]):
        return "fantasy_adventure"
    if any(w in combined for w in ["spaceship", "alien", "planet", "galaxy", "robot", "ai", "future"]):
        return "scifi_adventure"
    if any(w in combined for w in ["detective", "murder", "clue", "mystery", "disappear", "secret"]):
        return "mystery_adventure"
    if any(w in combined for w in ["funny", "accidentally", "oops", "wrong", "mix-up", "misunderstand"]):
        return "comedy_adventure"
    if any(w in combined for w in ["ghost", "dark", "creature", "shadow", "fear", "night"]):
        return "mystery_fantasy"
    return "adventure"


def _generate_ai_story_ideas(count: int = 5, prefer_genre: str = "") -> list[dict]:
    """
    Generate story ideas using Gemini (primary) or Groq (fallback).
    Returns list of idea dicts with source='ai_gemini' or 'ai_groq'.
    """
    # Load characters for context
    characters_context = ""
    if os.path.exists(CHARACTERS_FILE):
        char_data = _load_json(CHARACTERS_FILE)
        chars = char_data.get("characters", [])
        if chars:
            char_names = [c.get("name", c.get("id", "")) for c in chars]
            characters_context = f"Main characters: {', '.join(char_names)}. "

    # Load continuity for context
    continuity_context = ""
    continuity_file = os.path.join(DATA_DIR, "stories", "continuity.json")
    if os.path.exists(continuity_file):
        cont_data = _load_json(continuity_file)
        last_ep = cont_data.get("lastEpisode", {})
        if last_ep:
            continuity_context = f"Previous episode was about: {last_ep.get('title', 'unknown')}. Avoid repeating similar themes. "

    genre_hint = f"Preferred genre: {prefer_genre}. " if prefer_genre else ""

    prompt = f"""Generate exactly {count} unique, creative story ideas for an animated children's YouTube series.

{characters_context}{continuity_context}{genre_hint}

The series features Captain Finn (brave 8-year-old explorer), Squeaky (clever orange mouse sidekick), and Misty (purple cat rival who is secretly kind). They go on adventures together.

Each idea must be original, exciting, and suitable for children ages 4-10. Mix genres: adventure, mystery, fantasy, comedy, discovery.

Return ONLY a JSON array with this exact structure:
[
  {{
    "title": "Episode Title Here",
    "premise": "2-3 sentence description of the story premise and conflict",
    "genre": "genre_category",
    "themes": ["theme1", "theme2"],
    "characters": ["captain_finn", "squeaky", "misty"],
    "locations": ["location1"],
    "mood": "emotional tone description"
  }}
]

Genre options: fantasy_adventure, scifi_adventure, mystery_adventure, comedy_adventure, discovery_adventure, mystery_fantasy, adventure.
Return ONLY the JSON array, no markdown formatting or extra text."""

    # --- Try Gemini first ---
    if GEMINI_API_KEY:
        result = _call_gemini(prompt, GEMINI_API_KEY, model="gemini-2.5-flash")
        if result["success"] and result["text"]:
            ideas = _parse_ai_ideas(result["text"], source="ai_gemini")
            if ideas:
                return ideas

    # --- Fallback: Groq (Llama) ---
    if GROQ_API_KEY:
        result = _call_groq(prompt, GROQ_API_KEY)
        if result["success"] and result["text"]:
            ideas = _parse_ai_ideas(result["text"], source="ai_groq")
            if ideas:
                return ideas

    return []


def _parse_ai_ideas(raw_text: str, source: str) -> list[dict]:
    """Parse AI-generated JSON ideas into standardized idea dicts."""
    try:
        # Strip markdown code fences if present
        text = raw_text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        ideas_data = json.loads(text)
        if not isinstance(ideas_data, list):
            return []

        parsed = []
        for idea in ideas_data:
            if not isinstance(idea, dict):
                continue
            title = idea.get("title", "")
            premise = idea.get("premise", "")
            if not title or not premise:
                continue
            parsed.append({
                "source": source,
                "title": title,
                "premise": premise,
                "genre": idea.get("genre", "adventure"),
                "themes": idea.get("themes", []),
                "characters": idea.get("characters", []),
                "locations": idea.get("locations", []),
                "mood": idea.get("mood", ""),
                "upvotes": 0,
            })
        return parsed
    except (json.JSONDecodeError, ValueError):
        return []


@app.post("/api/fetch-story-ideas")
def fetch_story_ideas(req: FetchStoryIdeasRequest):
    """
    Node 1 core: Fetch story ideas from multiple sources.
    Returns a ranked list of story candidates.
    """
    all_ideas: list[dict] = []

    # --- Source 1: Local story seeds (curated, high quality) ---
    if req.useLocalSeeds and os.path.exists(STORY_SEEDS_FILE):
        seeds_data = _load_json(STORY_SEEDS_FILE)
        for seed in seeds_data.get("seeds", []):
            if seed.get("used"):
                continue
            all_ideas.append({
                "source": "local_seed",
                "seedId": seed["id"],
                "title": seed["title"],
                "premise": seed["premise"],
                "genre": seed.get("genre", "adventure"),
                "themes": seed.get("themes", []),
                "characters": seed.get("characters", []),
                "locations": seed.get("locations", []),
                "mood": seed.get("mood", ""),
                "upvotes": 0,
            })

    # --- Source 2: Reddit writing prompts ---
    if req.useReddit:
        reddit_subs = ["WritingPrompts", "shortstories", "SimplePrompts"]
        if os.path.exists(STORY_SEEDS_FILE):
            custom_subs = _load_json(STORY_SEEDS_FILE).get("redditSources", [])
            if custom_subs:
                reddit_subs = custom_subs
        reddit_ideas = _fetch_reddit_prompts(reddit_subs, req.redditLimit)
        all_ideas.extend(reddit_ideas)

    # --- Source 3: AI-Generated Ideas (Gemini -> Groq fallback) ---
    if req.useAI:
        ai_ideas = _generate_ai_story_ideas(
            count=req.aiIdeaCount,
            prefer_genre=req.preferGenre,
        )
        all_ideas.extend(ai_ideas)

    return {
        "success": True,
        "totalIdeas": len(all_ideas),
        "ideas": all_ideas[:req.maxResults],
        "sources": {
            "localSeeds": sum(1 for i in all_ideas if i["source"] == "local_seed"),
            "reddit": sum(1 for i in all_ideas if i["source"] == "reddit"),
            "ai_gemini": sum(1 for i in all_ideas if i["source"] == "ai_gemini"),
            "ai_groq": sum(1 for i in all_ideas if i["source"] == "ai_groq"),
        },
        "fetchedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ============================================
# PICK BEST STORY (scoring + selection)
# ============================================
class PickStoryRequest(BaseModel):
    ideas: list[dict]
    preferGenre: str = ""
    preferLocal: bool = True
    randomness: float = 0.2


@app.post("/api/pick-story")
def pick_story(req: PickStoryRequest):
    """
    Node 1 final: Score and select the best story idea.
    Considers genre preference, character compatibility, novelty, and virality.
    """
    if not req.ideas:
        raise HTTPException(status_code=400, detail="No ideas provided to pick from.")

    # Load character registry to check compatibility
    characters = {}
    if os.path.exists(CHARACTERS_FILE):
        char_data = _load_json(CHARACTERS_FILE)
        characters = {c["id"]: c for c in char_data.get("characters", [])}

    # Load history to avoid repeats
    used_titles: set[str] = set()
    if os.path.exists(STORY_HISTORY_FILE):
        history = _load_json(STORY_HISTORY_FILE)
        used_titles = {ep.get("title", "").lower() for ep in history.get("episodes", [])}

    scored: list[dict] = []
    for idea in req.ideas:
        score = 50.0  # base score

        # --- Boost local seeds (curated = higher quality) ---
        if idea.get("source") == "local_seed":
            score += 20
            if req.preferLocal:
                score += 10

        # --- Boost by upvotes (Reddit virality signal) ---
        ups = idea.get("upvotes", 0)
        if ups > 100:
            score += min(15, ups / 200)
        if ups > 1000:
            score += 10
        if ups > 5000:
            score += 10

        # --- Genre preference ---
        idea_genre = idea.get("genre", "")
        if req.preferGenre and idea_genre:
            if req.preferGenre.lower() in idea_genre.lower():
                score += 15
            elif idea_genre.lower() in req.preferGenre.lower():
                score += 10

        # --- Character compatibility --- 
        idea_chars = idea.get("characters", [])
        if idea_chars and characters:
            matched = sum(1 for c in idea_chars if c in characters)
            score += matched * 10  # bonus per matching character

        # --- Story quality signals ---
        title = idea.get("title", "")
        premise = idea.get("premise", "")

        # Good length premise = more material to work with
        if 50 < len(premise) < 500:
            score += 10
        elif len(premise) >= 500:
            score += 5

        # Curiosity/engagement keywords
        engagement_words = [
            "discover", "mysterious", "secret", "hidden", "suddenly",
            "never", "impossible", "ancient", "magical", "strange",
            "adventure", "quest", "treasure", "danger", "escape",
        ]
        for word in engagement_words:
            if word in title.lower() or word in premise.lower():
                score += 3

        # --- Penalize already used stories ---
        if title.lower() in used_titles:
            score -= 80  # heavy penalty, nearly disqualifies

        # --- Penalize very short titles ---
        if len(title) < 10:
            score -= 20

        # --- Add controlled randomness for variety ---
        score += random.uniform(-req.randomness * 20, req.randomness * 20)

        score = max(0, min(100, score))
        scored.append({**idea, "score": round(score, 1)})

    scored.sort(key=lambda x: x["score"], reverse=True)
    best = scored[0]

    # If the best is a local seed, mark it as used
    if best.get("source") == "local_seed" and best.get("seedId"):
        _mark_seed_used(best["seedId"])

    # Always assign registered characters - this ensures character consistency
    # Even Reddit-sourced stories get our fixed characters assigned
    selected_characters = best.get("characters", [])
    if not selected_characters or not any(c in characters for c in selected_characters):
        # Force all registered characters into every story (Finn, Squeaky, Misty)
        selected_characters = list(characters.keys())[:3]

    return {
        "success": True,
        "selectedStory": {
            "title": best.get("title", ""),
            "premise": best.get("premise", ""),
            "genre": best.get("genre", "adventure"),
            "themes": best.get("themes", []),
            "characters": selected_characters,
            "locations": best.get("locations", []),
            "mood": best.get("mood", "exciting"),
            "source": best.get("source", ""),
            "score": best.get("score", 0),
        },
        "characterDetails": {
            cid: {
                "name": characters[cid]["name"],
                "prompt": characters[cid]["prompt"],
                "personality": characters[cid]["personality"],
                "catchphrase": characters[cid]["catchphrase"],
                "voiceId": characters[cid]["voiceId"],
                "loraFile": characters[cid].get("loraFile", ""),
                "loraWeight": characters[cid].get("loraWeight", 0.8),
                "ipAdapterRef": characters[cid].get("ipAdapterRef", ""),
            }
            for cid in selected_characters
            if cid in characters
        },
        "seriesConfig": _get_series_config(),
        "rankedIdeas": scored[:10],
        "totalScored": len(scored),
        "pickedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _mark_seed_used(seed_id: str) -> None:
    """Mark a local story seed as used so it is not re-picked."""
    if not os.path.exists(STORY_SEEDS_FILE):
        return
    data = _load_json(STORY_SEEDS_FILE)
    for seed in data.get("seeds", []):
        if seed["id"] == seed_id:
            seed["used"] = True
            break
    _save_json(STORY_SEEDS_FILE, data)


def _get_series_config() -> dict:
    """Extract series-level config from characters file."""
    if not os.path.exists(CHARACTERS_FILE):
        return {}
    data = _load_json(CHARACTERS_FILE)
    series = data.get("series", {})
    return {
        "name": series.get("name", "Finn, Squeaky & Misty Adventures"),
        "genre": series.get("genre", "adventure"),
        "style": series.get("style", "2d cartoon, anime cel shading, clean lineart, flat colors, vibrant palette"),
        "checkpoint": series.get("checkpoint", COMFYUI_CHECKPOINT),
        "negativePrompt": series.get("negativePrompt", ""),
    }


# ============================================
# STORY HISTORY
# ============================================
@app.get("/api/story-history")
def get_story_history():
    """Get all previously generated episodes."""
    if not os.path.exists(STORY_HISTORY_FILE):
        return {"episodes": [], "totalEpisodes": 0}
    data = _load_json(STORY_HISTORY_FILE)
    return {
        "episodes": data.get("episodes", []),
        "totalEpisodes": len(data.get("episodes", [])),
    }


class RecordEpisodeRequest(BaseModel):
    title: str
    premise: str
    genre: str = ""
    characters: list[str] = []
    youtubeId: str = ""
    youtubeUrl: str = ""


@app.post("/api/story-history")
def record_episode(ep: RecordEpisodeRequest):
    """Record a completed episode in history (prevents re-use)."""
    if not os.path.exists(STORY_HISTORY_FILE):
        data = {"episodes": []}
    else:
        data = _load_json(STORY_HISTORY_FILE)

    episode = ep.model_dump()
    episode["episodeNumber"] = len(data.get("episodes", [])) + 1
    episode["createdAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    data.setdefault("episodes", []).append(episode)
    _save_json(STORY_HISTORY_FILE, data)

    # Also update episode count in characters.json
    if os.path.exists(CHARACTERS_FILE):
        char_data = _load_json(CHARACTERS_FILE)
        char_data.setdefault("series", {})["episodeCount"] = len(data["episodes"])
        _save_json(CHARACTERS_FILE, char_data)

    return {"success": True, "episode": episode}


# ============================================
# NODE 2 — SCRIPT GENERATION
# Cascade: Gemini → Groq → Ollama
# ============================================

SCRIPT_PROMPT_TEMPLATE = """You are a master children's storyteller creating emotionally engaging animated episodes for YouTube.

Your #1 goal: make children FEEL something — wonder, excitement, warmth, laughter, courage.
Keep language simple. Kids aged 4-10 must understand every word.
Characters must feel like real friends the audience wants to see again.

SERIES: {series_name}
GENRE: {genre}
EPISODE #{episode_number}: "{story_title}"
VISUAL STYLE: {style}

STORY PREMISE:
{premise}

CHARACTERS IN THIS EPISODE:
{character_descriptions}

MOOD: {mood}
THEMES: {themes}

CHARACTER CONSISTENCY RULES (CRITICAL):
- Every character MUST match their canonical description EXACTLY in every scene.
- Never change their appearance, outfit, accessories, or personality mid-episode.
- Characters must react in-character: Finn is brave and encouraging, Squeaky is clever and energetic, Misty is sarcastic but secretly kind.
- Use each character's catchphrase naturally at least once.

EMOTIONAL STORYTELLING RULES:
- Start with a strong emotional hook (mystery, danger, wonder) in the first 5 seconds.
- Give the episode ONE clear problem, ONE ticking pressure, and ONE emotional want for the hero.
- Make every scene answer one question and create the next question.
- Build emotional progression: curiosity → excitement → challenge → teamwork → triumph → warmth.
- Include at least ONE moment that makes kids laugh out loud.
- Include at least ONE moment of genuine wonder or "wow".
- Include at least ONE moment of teamwork solving a problem.
- End with emotional warmth + a tease that makes them want the next episode.
- Keep sentences SHORT and punchy. Simple words. Clear emotions.
- Avoid generic lines like "we can do anything together" unless earned by a specific action.

YOUR TASK: Write a complete animated episode script (target length: {target_duration} minutes).
The script will be turned into an animated video with AI-generated visuals, so EVERY scene must have detailed visual descriptions.

✅ CRITICAL REQUIREMENT FOR FIX 4: You MUST generate EXACTLY {scene_count} separate scenes.
Each scene will be animated individually with AnimateDiff. Do NOT combine scenes.
Format each scene as a separate JSON object in the "scenes" array.

You MUST respond with ONLY valid JSON (no markdown, no code fences):

{{
  "title": "Episode title (max 80 chars, curiosity-triggering)",
  "description": "YouTube description with hooks, max 300 words. Include episode summary, character names, and series name.",
  "tags": ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7", "tag8", "tag9", "tag10"],
  "scenes": [
    {{
      "sceneNumber": 1,
      "narration": "What the narrator says during this scene. 2-4 vivid sentences, 25-45 words. Use concrete action, suspense, and sensory detail.",
      "dialogue": [
        {{"character": "Character Name", "line": "What they say"}},
        {{"character": "Character Name", "line": "Their response"}}
      ],
      "visualDescription": "Detailed description of what we SEE in this scene. Include: character positions, expressions, background, lighting, camera angle. 30-60 words.",
      "charactersInScene": ["character_id_1", "character_id_2"],
      "emotion": "primary emotion (happy, scared, excited, curious, sad, determined, surprised)",
      "cameraAngle": "wide_shot | medium_shot | close_up | over_shoulder | birds_eye | low_angle",
      "location": "Where this scene takes place",
      "textOverlay": "",
      "duration": 8
    }}
  ],
  "fullNarration": "All narration text combined in order",
  "thumbnailDescription": "Describe the ideal thumbnail image: main character in dramatic pose + key visual element",
  "cliffhanger": "Final line teasing the next episode (for end card)",
  "totalScenes": 0,
  "estimatedDuration": 0
}}

SCENE STRUCTURE FOR A {target_duration}-MINUTE EPISODE:
✅ FIX 4: Generate EXACTLY {scene_count} scenes (not fewer, not combined):
- COLD OPEN (1 scene): Start mid-action or with a mystery. Hook the viewer in 5 seconds.
- INTRO/SETUP (1-2 scenes): Establish the situation. Show characters in their normal world.
- INCITING INCIDENT (1 scene): Something unexpected happens that launches the adventure.
- RISING ACTION (4-6 scenes): The characters face challenges, discover things, work together.
  Include: at least 1 funny moment, 1 "wow" discovery, 1 obstacle they must overcome.
- CLIMAX (1-2 scenes): The biggest challenge. Tension at maximum.
- RESOLUTION (1-2 scenes): How they solve it. Show teamwork and character growth.
- ENDING + HOOK (1 scene): Wrap up warmly, then tease something for the next episode.
  End with: "But that is a story for another day..." or similar cliffhanger.

Each scene is 6-12 seconds long. Total narration should be {word_count} words.
Every scene is SEPARATE (has its own narration, visuals, and emotions).

NARRATION RULES:
- Narrator speaks in third person, like a warm storybook: "Captain Finn looked up at the sky..."
- Use SIMPLE words a 5-year-old understands. No complex vocabulary.
- Each scene has 2-3 vivid sentences of narration (18-32 words per scene).
- Use sensory language: sounds, colors, textures, temperatures.
- Build atmosphere — describe what characters see, hear, and feel.
- Show emotions through actions: "His heart raced" not "He felt anxious"
- Pause between dramatic moments (use "..." in narration).
- This episode is narration-only. Do not write spoken character dialogue.

DIALOGUE RULES:
- Keep dialogue as an empty array for every scene.
- If reactions are needed, express them via narrator text only.

VISUAL DESCRIPTION RULES:
- Describe exactly what the camera sees, like a storyboard artist.
- ALWAYS include FULL character appearance description (hair, eyes, outfit, accessories) for EVERY character in the scene — this ensures AI image consistency.
- Always mention character expressions (big smile, worried eyes, surprised mouth).
- Include background details (environment, weather, time of day).
- Specify camera angles for variety: alternate between wide, medium, close-up.
- Close-ups for emotional moments. Wide shots for adventure/discovery. Medium for dialogue.
- Use character prompts for visual consistency (NEVER deviate from these):
{character_prompts}

IMPORTANT: 
- Respond with ONLY the JSON object. No text before or after.
- Every scene must be unique and separate.
- Do NOT skip scenes or combine them.
- Scene array length MUST equal {scene_count}."""


def _build_script_prompt(data: dict) -> str:
    """Build the LLM prompt from story context + character data.
    Automatically injects episode continuity for multi-episode consistency.
    
    SECTION 3: Supports 'retention' pacing (2-4 min, fewer scenes, faster pace)
    and 'standard' pacing (5-8 min, more scenes).
    """
    target_min = data.get("targetDuration", 3)
    pacing_mode = data.get("pacingMode", "retention")
    
    if pacing_mode == "retention":
        # Retention-optimized: fewer scenes, faster cuts, tighter story
        target_min = min(target_min, 2)  # Cap at 2 minutes
        scene_count = max(10, min(14, int(target_min * 6)))
        word_count = f"{target_min * 90}-{target_min * 120}"
    else:
        # Standard pacing
        scene_count = max(10, min(20, int(target_min * 2)))
        word_count = f"{target_min * 140}-{target_min * 170}"

    char_prompts = ""
    for cid, cdata in (data.get("characters") or {}).items():
        char_prompts += f"  - {cdata.get('name', cid)}: {cdata.get('prompt', '')}\n"

    base_prompt = SCRIPT_PROMPT_TEMPLATE.format(
        series_name=data.get("seriesName", "Untitled Series"),
        genre=data.get("seriesGenre", "adventure"),
        episode_number=data.get("episodeNumber", 1),
        story_title=data.get("storyTitle", "Untitled"),
        style=data.get("seriesStyle", "cartoon style, vibrant colors"),
        premise=data.get("storyPremise", "An exciting adventure awaits."),
        character_descriptions=data.get("characterDescriptions", "No characters defined."),
        mood=data.get("storyMood", "exciting"),
        themes=", ".join(data.get("storyThemes", ["adventure"])),
        target_duration=target_min,
        scene_count=scene_count,  # FIX 4 + FIX 9: Explicit, reduced scene count
        word_count=word_count,
        character_prompts=char_prompts or "  (no character prompts defined)",
    )

    # Inject episode continuity (story state from previous episodes)
    continuity_prompt = ""
    if os.path.exists(CONTINUITY_FILE):
        try:
            cont_data = _load_json(CONTINUITY_FILE)
            episodes = cont_data.get("episodes", [])
            if episodes:
                state = _build_cumulative_state(episodes)
                continuity_prompt = _build_continuity_prompt(state)
        except Exception:
            pass

    if continuity_prompt:
        base_prompt = f"{base_prompt}\n\n{continuity_prompt}"

    # SECTION 3 + 8: Retention engine injection
    if pacing_mode == "retention":
        retention_rules = """

RETENTION OPTIMIZATION RULES (HIGH PRIORITY):
- First 5 seconds MUST hook the viewer: start mid-action, mid-mystery, or with a startling visual.
- NO slow openings. No "once upon a time" beginnings. Start with action or intrigue.
- Every scene must advance the story. ZERO filler. Cut anything that doesn't serve the plot.
- Scene transitions should create curiosity gaps: end each scene with a question or tension.
- Include ONE recurring catchphrase per character (audience recognition + engagement).
- Pacing: fast cuts for action, slow moments only for emotional payoff.
- End with a STRONG curiosity loop: "What will happen next? Find out in the next episode!"
- Each scene should be 6-12 seconds max. Keep energy HIGH.
- Emotional arc must be COMPLETE within the short duration: hook → conflict → payoff.
- Use cliffhanger-style scene transitions (mini hooks between scenes).
"""
        base_prompt = f"{base_prompt}\n{retention_rules}"

    return base_prompt




def _parse_llm_json(text: str):
    if not text:
        return None

    # Try finding the first '{' and the last '}'
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        return None
    
    json_text = text[start_idx:end_idx+1]
    
    # Try direct parse
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        # Try cleaning trailing commas or other common issues
        import re
        # Remove trailing commas before closing braces/brackets
        json_text = re.sub(r',\s*([\]}])', r'\1', json_text)
        try:
            return json.loads(json_text)
        except:
            pass
    
    return None

# --- Provider 1: Google Gemini 2.5 Flash (best quality, free tier) ---

def _call_gemini(prompt: str, api_key: str, model: str = "gemini-2.5-flash") -> dict:
    """Call Google Gemini API. Returns {"success": bool, "text": str, "error": str}."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
        f":generateContent?key={api_key}"
    )
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.85,
            "topP": 0.95,
            "topK": 64,
            "maxOutputTokens": 16384,
            "responseMimeType": "application/json",
        },
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode())
        # Extract text from Gemini response
        candidates = result.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                return {"success": True, "text": parts[0].get("text", ""), "error": ""}
        return {"success": False, "text": "", "error": "No candidates in Gemini response"}
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        return {"success": False, "text": "", "error": f"Gemini HTTP {e.code}: {body[:300]}"}
    except Exception as e:
        return {"success": False, "text": "", "error": f"Gemini error: {str(e)[:300]}"}


# --- Pollinations.ai (FLUX/Seedream — primary free-tier image gen) ---

def _generate_pollinations(
    prompt: str,
    output_path: str,
    width: int = 1024,
    height: int = 576,
    model: str = "",
    negative_prompt: str = "",
) -> dict:
    """
    Generate an image using Pollinations.ai.
    Returns {"success": bool, "path": str, "error": str}.

    Models: flux (best for illustration), flux-realism, turbo
    Free tier — works without API key via image.pollinations.ai URL endpoint.
    If POLLINATIONS_API_KEY is set, uses the OpenAI-compatible endpoint for higher limits.
    """
    import base64

    use_model = model or POLLINATIONS_MODEL or "flux"
    key = POLLINATIONS_API_KEY

    # Build prompt with negative prompt guidance
    full_prompt = prompt
    if negative_prompt:
        full_prompt = f"{prompt}. Avoid: {negative_prompt}"

    # --- Strategy 1: If API key is available, use OpenAI-compatible endpoint ---
    if key:
        url = "https://gen.pollinations.ai/v1/images/generations"
        payload = json.dumps({
            "prompt": full_prompt,
            "model": use_model,
            "size": f"{width}x{height}",
            "quality": "hd",
            "response_format": "b64_json",
            "n": 1,
        }).encode()

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                result = json.loads(resp.read().decode())

            data_list = result.get("data", [])
            if data_list:
                image_b64 = data_list[0].get("b64_json", "")
                if image_b64:
                    image_bytes = base64.b64decode(image_b64)
                    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
                    with open(output_path, "wb") as f:
                        f.write(image_bytes)
                    print(f"[Pollinations/{use_model}] Generated (API): {output_path} ({len(image_bytes)} bytes)")
                    return {"success": True, "path": output_path, "error": ""}
                # Try URL in response
                img_url = data_list[0].get("url", "")
                if img_url:
                    img_req = urllib.request.Request(img_url)
                    with urllib.request.urlopen(img_req, timeout=60) as img_resp:
                        image_bytes = img_resp.read()
                    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
                    with open(output_path, "wb") as f:
                        f.write(image_bytes)
                    print(f"[Pollinations/{use_model}] Generated (API/URL): {output_path} ({len(image_bytes)} bytes)")
                    return {"success": True, "path": output_path, "error": ""}
        except Exception as e:
            print(f"[Pollinations] API endpoint failed: {str(e)[:200]}, falling back to URL endpoint")

    # --- Strategy 2: Free URL-based endpoint (no key required) ---
    encoded_prompt = urllib.parse.quote(full_prompt[:1500])  # Limit prompt length for URL
    img_url = (
        f"https://image.pollinations.ai/prompt/{encoded_prompt}"
        f"?width={width}&height={height}&model={use_model}&nologo=true&enhance=true&seed={random.randint(1, 999999)}"
    )

    try:
        req = urllib.request.Request(img_url, headers={
            "User-Agent": "n8n-story-api/2.0",
            "Accept": "image/*",
        })
        with urllib.request.urlopen(req, timeout=240) as resp:
            content_type = resp.headers.get("Content-Type", "")
            image_bytes = resp.read()

        # Validate we got an actual image (not an error page)
        if len(image_bytes) < 5000:
            return {"success": False, "path": "", "error": f"Pollinations returned too-small response ({len(image_bytes)} bytes) — likely rate limited"}

        if "text/html" in content_type or "application/json" in content_type:
            return {"success": False, "path": "", "error": f"Pollinations returned non-image content: {content_type}"}

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(image_bytes)

        print(f"[Pollinations/{use_model}] Generated (URL): {output_path} ({len(image_bytes)} bytes)")
        return {"success": True, "path": output_path, "error": ""}

    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        print(f"[Pollinations] HTTP {e.code}: {body[:300]}")
        return {"success": False, "path": "", "error": f"Pollinations HTTP {e.code}: {body[:300]}"}
    except Exception as e:
        print(f"[Pollinations] Error: {str(e)[:300]}")
        return {"success": False, "path": "", "error": f"Pollinations error: {str(e)[:300]}"}


# --- Provider 2: Groq (Llama 4 Scout, fast, free tier) ---

def _call_groq(prompt: str, api_key: str, model: str = "meta-llama/llama-4-scout-17b-16e-instruct") -> dict:
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert animated story script writer for children's YouTube series. You create emotionally engaging, easy-to-understand stories with consistent characters, strong pacing, and cinematic scene descriptions. Return ONLY valid JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.85,
        "max_tokens": 8192,
        "top_p": 0.95
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "n8n-app/1.0"
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode())

        choices = result.get("choices", [])
        if choices:
            return {"success": True, "text": choices[0]["message"]["content"], "error": ""}

        return {"success": False, "text": "", "error": "No choices in Groq response"}

    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        return {"success": False, "text": "", "error": f"Groq HTTP {e.code}: {body}"}

    except Exception as e:
        return {"success": False, "text": "", "error": str(e)}


# --- Provider 3: Ollama (local fallback, always available) ---

def _call_ollama(prompt: str, model: str = "llama3.1:8b") -> dict:
    """Call local Ollama API. Returns {"success": bool, "text": str, "error": str}."""
    url = f"{OLLAMA_URL}/api/generate"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.8,
            "num_predict": 8192,
            "top_p": 0.92,
            "repeat_penalty": 1.1,
        },
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            result = json.loads(resp.read().decode())
        text = result.get("response", "")
        return {"success": True, "text": text, "error": ""}
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        return {"success": False, "text": "", "error": f"Ollama HTTP {e.code}: {body[:300]}"}
    except Exception as e:
        return {"success": False, "text": "", "error": f"Ollama error: {str(e)[:300]}"}


# --- LLM Status Check ---

@app.get("/api/llm-status")
def llm_status():
    """Check which LLM providers are available and configured."""
    status = {
        "pollinations": {"configured": bool(POLLINATIONS_API_KEY), "model": POLLINATIONS_MODEL, "status": "configured" if POLLINATIONS_API_KEY else "not configured"},
        "gemini": {"configured": bool(GEMINI_API_KEY), "status": "unknown"},
        "groq": {"configured": bool(GROQ_API_KEY), "status": "unknown"},
        "ollama": {"configured": True, "status": "unknown"},
        "recommended": "none",
    }

    # Test Gemini
    if GEMINI_API_KEY:
        try:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models"
                f"?key={GEMINI_API_KEY}"
            )
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    status["gemini"]["status"] = "ok"
        except Exception:
            status["gemini"]["status"] = "error"

    # Test Groq
    if GROQ_API_KEY:
        try:
            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/models",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "User-Agent": "n8n-app/1.0"
                },
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    status["groq"]["status"] = "ok"
        except urllib.error.HTTPError as e:
            status["groq"]["status"] = f"error: {e.code}"
        except Exception as e:
            status["groq"]["status"] = f"error: {str(e)}"

    # Test Ollama
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                models = json.loads(resp.read().decode()).get("models", [])
                status["ollama"]["status"] = "ok"
                status["ollama"]["models"] = [m.get("name", "") for m in models]
    except Exception:
        status["ollama"]["status"] = "unreachable"

    # Pick recommended
    if status["gemini"]["status"] == "ok":
        status["recommended"] = "gemini"
    elif status["groq"]["status"] == "ok":
        status["recommended"] = "groq"
    elif status["ollama"]["status"] == "ok":
        status["recommended"] = "ollama"

    return status


# --- Script Generation Endpoint ---

class GenerateScriptRequest(BaseModel):
    # Story context (from Node 1 output)
    storyTitle: str
    storyPremise: str
    storyGenre: str = "adventure"
    storyThemes: list[str] = []
    storyMood: str = "exciting"
    characters: dict = {}
    characterDescriptions: str = ""
    characterIds: list[str] = []
    seriesName: str = "Finn, Squeaky & Misty Adventures"
    seriesStyle: str = "2d cartoon, anime cel shading, clean lineart, flat colors, vibrant palette"
    seriesGenre: str = "adventure"
    checkpoint: str = ""  # ComfyUI fallback checkpoint (leave empty for env default)
    negativePrompt: str = ""
    episodeNumber: int = 1
    # Generation settings
    targetDuration: int = 3  # minutes (SECTION 3: optimized for retention)
    pacingMode: str = "retention"  # "retention" (2-4 min, fast pace) | "standard" (5-8 min)
    narrationOnly: bool = True
    provider: str = "auto"   # auto | gemini | groq | ollama
    geminiModel: str = "gemini-2.5-flash"
    groqModel: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    ollamaModel: str = "llama3.1:8b"
    geminiApiKey: str = ""
    groqApiKey: str = ""


@app.post("/api/generate-script")
def generate_script(req: GenerateScriptRequest):
    """
    Node 2: Generate a full episode script using LLM cascade.
    Tries Gemini first (best quality), then Groq, then Ollama.
    """
    prompt = _build_script_prompt(req.model_dump())

    # Resolve API keys: request > environment
    gemini_key = req.geminiApiKey or GEMINI_API_KEY
    groq_key = req.groqApiKey or GROQ_API_KEY

    # Build provider order
    if req.provider == "auto":
        providers = []
        if gemini_key:
            providers.append(("gemini", gemini_key))
        if groq_key:
            providers.append(("groq", groq_key))
        providers.append(("ollama", ""))
    elif req.provider == "gemini":
        providers = [("gemini", gemini_key)]
    elif req.provider == "groq":
        providers = [("groq", groq_key)]
    elif req.provider == "ollama":
        providers = [("ollama", "")]
    else:
        providers = [("ollama", "")]

    # Try each provider in order
    errors: list[dict] = []
    for provider_name, api_key in providers:
        if provider_name == "gemini":
            if not api_key:
                errors.append({"provider": "gemini", "error": "No API key configured"})
                continue
            result = _call_gemini(prompt, api_key, req.geminiModel)
        elif provider_name == "groq":
            if not api_key:
                errors.append({"provider": "groq", "error": "No API key configured"})
                continue
            result = _call_groq(prompt, api_key, req.groqModel)
        else:
            result = _call_ollama(prompt, req.ollamaModel)

        if not result["success"]:
            errors.append({"provider": provider_name, "error": result["error"]})
            continue

        # Parse the JSON response
        script = _parse_llm_json(result["text"])
        if script is None:
            errors.append({
                "provider": provider_name,
                "error": f"Failed to parse JSON from response (length={len(result['text'])})",
            })
            continue

        # Validate and fix the script
        script = _validate_script(script, req.model_dump())

        return {
            "success": True,
            "provider": provider_name,
            "script": script,
            "prompt_length": len(prompt),
            "response_length": len(result["text"]),
            "errors": errors,
            "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    # All providers failed
    raise HTTPException(
        status_code=502,
        detail={
            "message": "All LLM providers failed to generate script",
            "errors": errors,
        },
    )


def _validate_script(script: dict, context: dict) -> dict:
    """Validate and fix the generated script, ensuring required fields."""

    # Ensure title
    if not script.get("title"):
        script["title"] = context.get("storyTitle", "Untitled Episode")

    # Ensure description
    if not script.get("description"):
        script["description"] = f"Episode #{context.get('episodeNumber', 1)} of {context.get('seriesName', 'Series')}: {context.get('storyTitle', '')}"

    # Ensure tags
    tags = script.get("tags", [])
    if not isinstance(tags, list) or len(tags) == 0:
        tags = ["animation", "cartoon", "story", "kids", "adventure", "animated story"]
    script["tags"] = [str(t)[:50] for t in tags[:15]]

    # Validate scenes
    scenes = script.get("scenes", [])
    if not isinstance(scenes, list):
        scenes = []

    pacing_mode = str(context.get("pacingMode", "retention")).lower()
    retention_mode = pacing_mode == "retention"
    min_scenes = 8 if retention_mode else 10
    max_scenes = 14 if retention_mode else 20
    wpm = 170 if retention_mode else 150
    min_scene_sec = 6.0 if retention_mode else 7.0
    max_scene_sec = 10.0 if retention_mode else 16.0
    pause_sec = 0.8 if retention_mode else 1.0

    def _scene_duration_from_text(text: str) -> float:
        words = len((text or "").split())
        if words <= 0:
            return min_scene_sec
        estimate = (words / wpm) * 60 + pause_sec
        return round(max(min_scene_sec, min(max_scene_sec, estimate)), 2)

    def _norm_text(text: str) -> str:
        t = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
        return re.sub(r"\s+", " ", t).strip()

    def _jaccard_similarity(a: str, b: str) -> float:
        sa = set(_norm_text(a).split())
        sb = set(_norm_text(b).split())
        if not sa and not sb:
            return 1.0
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / max(1, len(sa | sb))

    validated_scenes = []
    for i, scene in enumerate(scenes[:max_scenes]):
        if not isinstance(scene, dict):
            continue
        narration_text = str(scene.get("narration", ""))[:600]
        validated_scenes.append({
            "sceneNumber": i + 1,
            "narration": narration_text,
            # Narration-only pipeline: keep all spoken content in narrator track.
            "dialogue": [],
            "visualDescription": str(scene.get("visualDescription", ""))[:400],
            "charactersInScene": scene.get("charactersInScene", []) if isinstance(scene.get("charactersInScene"), list) else [],
            "emotion": str(scene.get("emotion", "neutral"))[:30],
            "cameraAngle": str(scene.get("cameraAngle", "medium_shot"))[:30],
            "location": str(scene.get("location", ""))[:100],
            # Keep generated scenes text-free; subtitles stay in files, not in the artwork.
            "textOverlay": "",
            "duration": _scene_duration_from_text(narration_text),
        })

    # If the LLM returns too few scenes, add short non-repetitive bridge scenes.
    if len(validated_scenes) < min_scenes:
        defaults = [
            {"narration": f"A strange clue appeared in {context.get('storyTitle', 'the adventure')}, and the team rushed to understand it.", "emotion": "curious", "cameraAngle": "wide_shot"},
            {"narration": "A small mistake made the challenge harder, but the heroes adapted quickly.", "emotion": "determined", "cameraAngle": "medium_shot"},
            {"narration": "They noticed a hidden detail others would miss, and everything started to make sense.", "emotion": "surprised", "cameraAngle": "close_up"},
            {"narration": "One risky move changed the momentum, and hope returned.", "emotion": "excited", "cameraAngle": "low_angle"},
            {"narration": "The path ahead was still unknown, but they moved forward together.", "emotion": "warm", "cameraAngle": "wide_shot"},
        ]
        while len(validated_scenes) < min_scenes:
            idx = len(validated_scenes)
            d = defaults[idx % len(defaults)]
            validated_scenes.append({
                "sceneNumber": idx + 1,
                "narration": d["narration"],
                "dialogue": [],
                "visualDescription": f"Scene showing the characters in a {context.get('seriesStyle', 'cartoon')} style",
                "charactersInScene": context.get("characterIds", [])[:2],
                "emotion": d["emotion"],
                "cameraAngle": d["cameraAngle"],
                "location": "unknown",
                "textOverlay": "",
                "duration": _scene_duration_from_text(d["narration"]),
            })

    # Engagement normalization: enforce a stronger opener + de-duplicate near-repeated narration.
    if validated_scenes:
        first = validated_scenes[0].get("narration", "")
        first_norm = _norm_text(first)
        hook_tokens = ("suddenly", "alarm", "mystery", "before", "warning", "vanish", "secret", "storm")
        if not any(tok in first_norm for tok in hook_tokens):
            validated_scenes[0]["narration"] = f"Suddenly, a new mystery burst open. {first}".strip()
            validated_scenes[0]["duration"] = _scene_duration_from_text(validated_scenes[0]["narration"])

    bridge_variants = [
        "A fresh clue changed their direction, and the tension rose.",
        "The plan shifted quickly as a hidden detail came into focus.",
        "For one breath, everything paused, then the adventure surged ahead.",
        "A new obstacle appeared, forcing a smarter move than before.",
    ]
    for i in range(1, len(validated_scenes)):
        cur = validated_scenes[i].get("narration", "")
        prev = validated_scenes[i - 1].get("narration", "")
        if _jaccard_similarity(cur, prev) > 0.82:
            replacement = bridge_variants[(i - 1) % len(bridge_variants)]
            validated_scenes[i]["narration"] = replacement
            validated_scenes[i]["duration"] = _scene_duration_from_text(replacement)

    script["scenes"] = validated_scenes
    script["totalScenes"] = len(validated_scenes)

    # Rebuild full narration from scenes
    all_narration = []
    for s in validated_scenes:
        if s["narration"]:
            all_narration.append(s["narration"])
        for d in s.get("dialogue", []):
            if isinstance(d, dict) and d.get("line"):
                all_narration.append(f'{d.get("character", "")}: {d["line"]}')
    script["fullNarration"] = " ".join(all_narration)

    # Estimated duration
    total_dur = sum(s["duration"] for s in validated_scenes)
    script["estimatedDuration"] = round(total_dur, 1)

    # Thumbnail fallback
    if not script.get("thumbnailDescription"):
        script["thumbnailDescription"] = f"Dramatic scene from {script['title']}, cartoon style, vibrant colors"

    # Cliffhanger fallback
    if not script.get("cliffhanger"):
        script["cliffhanger"] = "But that is a story for another day..."

    return script


# ============================================
# NODE 3 — SCENE PLANNER
# Transforms script scenes into generation-ready tasks:
#   - ComfyUI image prompts (with LoRA/IP-Adapter)
#   - AnimateDiff animation parameters
#   - Audio assignments (per-character TTS)
#   - Timing (duration based on word count)
#   - Ken Burns fallback settings
# ============================================

class PlanScenesRequest(BaseModel):
    # From "Structure Script Output" (Node 2 output)
    title: str
    description: str = ""
    tags: list[str] = []
    scenes: list[dict]
    fullNarration: str = ""
    thumbnailDescription: str = ""
    cliffhanger: str = ""
    totalScenes: int = 0
    estimatedDuration: float = 0
    # Series / character context
    seriesName: str = "Finn, Squeaky & Misty Adventures"
    seriesStyle: str = "2d cartoon, anime cel shading, clean lineart, flat colors, vibrant palette, children book illustration, consistent character design, soft shadows, simple background"
    checkpoint: str = ""  # ComfyUI fallback checkpoint (leave empty for env default)
    negativePrompt: str = "realistic, photo, 3d render, photorealistic, blurry, deformed, extra limbs, bad anatomy, ugly, distorted face, inconsistent character, low quality, noisy, text, watermark, signature, live action, real person"
    episodeNumber: int = 1
    characters: dict = {}
    characterIds: list[str] = []
    # Generation settings (ComfyUI fallback only)
    imageWidth: int = 512
    imageHeight: int = 384
    animateFrames: int = 16   # 16 frames = 1.3s at 12fps, smooth and lightweight
    animateSteps: int = 15    # 15 steps for cartoon style
    imageSteps: int = 20
    imageCfg: float = 7.0
    animateCfg: float = 6.5
    motionModule: str = "v3_sd15_mm.ckpt"
    sampler: str = "euler_ancestral"
    scheduler: str = "normal"
    # Output directories
    imageDir: str = "/data/images"
    clipDir: str = "/data/clips"
    audioDir: str = "/data/audio"
    videoDir: str = "/data/video"
    # TTS settings
    narratorVoice: str = "en-US-DavisNeural"  # FIXED: Single narrator voice
    ttsRate: str = "+0%"  # CHANGED from '+15%' to natural pace (FIX 5 - was causing audio issues)
    ttsPitch: str = "+0Hz"
    # Video format (16:9 horizontal for long-form YouTube)
    videoWidth: int = 1920
    videoHeight: int = 1080
    fps: int = 12  # CHANGED from 24 to reduce processing load (FIX 6)
    # Words-per-minute for duration estimation
    wordsPerMinute: int = 150
    pacingMode: str = "retention"


def _estimate_duration(
    text: str,
    wpm: int,
    min_seconds: float = 5.0,
    max_seconds: float = 14.0,
    pause_seconds: float = 1.0,
) -> float:
    """Estimate speech duration in seconds from text and WPM."""
    if not text:
        return min_seconds
    words = len(text.split())
    dur = (words / wpm) * 60
    return max(min_seconds, min(max_seconds, round(dur + pause_seconds, 2)))


# STEP 1: New structured prompt function using CANONICAL CONSISTENCY SYSTEM
def _build_structured_prompt(scene: dict, characters: dict, series_style: str) -> str:
    """Build scene prompt using the canonical character consistency system.
    
    Delegates to _build_canonical_scene_prompt to ensure character identity
    NEVER drifts across scenes. Every scene uses the exact same frozen
    character descriptions regardless of context.
    """
    character_ids = scene.get("charactersInScene", [])
    
    # Use the canonical prompt builder (single source of truth)
    prompt = _build_canonical_scene_prompt(
        character_ids=character_ids,
        scene_context={
            "emotion": scene.get("emotion", "neutral"),
            "cameraAngle": scene.get("cameraAngle", "medium_shot"),
            "location": scene.get("location", ""),
            "visualDescription": scene.get("visualDescription", ""),
        },
        style=series_style,
    )
    
    return prompt


def _build_comfyui_scene_prompt(
    scene: dict,
    style: str,
    negative: str,
    characters: dict,
) -> dict:
    """Build a complete ComfyUI prompt bundle for one scene with strong character consistency."""
    visual = scene.get("visualDescription", "")
    emotion = scene.get("emotion", "neutral")
    camera = scene.get("cameraAngle", "medium_shot")
    location = scene.get("location", "")
    scene_chars = scene.get("charactersInScene", [])

    # -- Build positive prompt --
    parts = []

    # Style prefix with cartoon emphasis
    parts.append(f"(({style}:1.2))")

    # Camera angle mapping
    camera_keywords = {
        "wide_shot": "wide angle shot, full body visible, establishing shot",
        "medium_shot": "medium shot, waist up, conversational framing",
        "close_up": "close-up shot, face detail, expressive, portrait",
        "over_shoulder": "over the shoulder shot, depth, perspective",
        "birds_eye": "birds eye view, top-down angle, overhead shot",
        "low_angle": "low angle shot, dramatic, looking up, powerful",
    }
    cam_kw = camera_keywords.get(camera, "medium shot")
    parts.append(cam_kw)

    # Character prompts with STRONG emphasis weighting for consistency
    lora_list = []
    for cid in scene_chars:
        cdata = characters.get(cid, {})
        if not cdata:
            cdata = scene.get("characterPrompts", {}).get(cid, {})
        if cdata:
            char_prompt = cdata.get("prompt", "")
            char_name = cdata.get("name", "")
            if char_prompt:
                # Wrap character prompt with emphasis for consistency
                parts.append(f"(({char_prompt}:1.3))")
                # Add character name as extra anchor
                if char_name:
                    parts.append(f"({char_name}:1.1)")
            lora_file = cdata.get("loraFile", "")
            if lora_file:
                weight = cdata.get("loraWeight", 0.8)
                lora_list.append({"file": lora_file, "weight": weight})

    # Emotion keywords
    emotion_map = {
        "happy": "happy expression, smiling, joyful",
        "sad": "sad expression, downcast eyes, melancholy",
        "scared": "scared expression, wide eyes, fearful",
        "excited": "excited expression, energetic, dynamic pose",
        "curious": "curious expression, tilted head, inquisitive",
        "determined": "determined expression, confident stance, focused",
        "surprised": "surprised expression, open mouth, wide eyes",
        "angry": "angry expression, furrowed brows, intense",
        "neutral": "calm expression, relaxed pose",
    }
    emotion_kw = emotion_map.get(emotion, "expressive")
    parts.append(emotion_kw)

    # Visual description (from the LLM script)
    if visual:
        parts.append(visual)

    # Location
    if location:
        parts.append(f"setting: {location}")

    # Quality boosters — cartoon-specific
    parts.append("detailed background, sharp focus, vibrant colors, high quality, masterpiece, anime cel shading, clean lineart, consistent character design")

    positive_prompt = ", ".join(p for p in parts if p)

    # -- Build motion hint for AnimateDiff (emotion → motion mapping) --
    motion_map = {
        "happy": "character smiling, gentle swaying, happy movement",
        "sad": "character looking down, slow subtle movement, melancholy atmosphere",
        "scared": "character trembling slightly, eyes darting, tense atmosphere",
        "excited": "character moving energetically, dynamic motion, bouncy animation",
        "curious": "character leaning forward, head tilting, exploring movement",
        "determined": "character standing firm, confident stride, powerful stance",
        "surprised": "character jolting back, wide eyes opening, shock reaction",
        "angry": "character clenching fists, intense stare, aggressive posture",
        "neutral": "character breathing naturally, gentle idle animation, subtle wind",
    }
    motion_hint = motion_map.get(emotion, "gentle subtle movement, natural animation")

    # Camera motion hints
    camera_motion = {
        "wide_shot": "slow camera pan, establishing movement",
        "medium_shot": "slight camera drift, conversational framing",
        "close_up": "slow zoom in, focus pull, intimate framing",
        "over_shoulder": "subtle camera sway, depth shift",
        "birds_eye": "slow rotation, aerial drift",
        "low_angle": "dramatic camera tilt, upward sweep",
    }
    cam_motion = camera_motion.get(camera, "gentle camera movement")
    motion_hint = f"{motion_hint}, {cam_motion}"

    # -- Negative prompt — cartoon-specific --
    neg_base = negative or "realistic, photo, blurry, deformed, extra limbs, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra fingers, ugly, watermark, text, signature"
    neg_cartoon = f"{neg_base}, 3d render, photorealistic, live action, real person, photograph, different character design, inconsistent character, changing appearance"

    return {
        "positivePrompt": positive_prompt[:1000],
        "negativePrompt": neg_cartoon[:500],
        "loras": lora_list,
        "motionHint": motion_hint,
    }


def _build_audio_tasks(scene: dict, characters: dict, narrator_voice: str) -> list[dict]:
    """Build TTS task list for one scene: narration + character dialogue.
    Uses the voice lock system to ensure permanent voice identity per character."""
    tasks = []
    sn = scene.get("sceneNumber", 0)

    # Narration (always present — uses locked narrator voice)
    narration = scene.get("narration", "")
    if narration:
        tasks.append({
            "type": "narration",
            "text": narration,
            "voice": narrator_voice,
            "filename": f"scene_{sn:02d}_narration.m4a",
        })

    # Narration-only pipeline: dialogue TTS is intentionally disabled.

    return tasks


@app.post("/api/plan-scenes")
def plan_scenes(req: PlanScenesRequest):
    """
    Node 3: Transform script scenes into generation-ready task lists.

    For each scene, produces:
      - imageTask:     ComfyUI prompt for keyframe image generation
      - animationTask: AnimateDiff parameters to animate the keyframe
      - audioTasks:    TTS tasks (narration + per-character dialogue)
      - timing:        Duration estimates based on word count
      - kenBurns:      Fallback settings if animation fails
      - assembly:      Per-scene FFmpeg assembly parameters
    """
    # Load character data (prefer request data, fall back to file)
    characters = req.characters or {}
    if not characters and os.path.exists(CHARACTERS_FILE):
        char_data = _load_json(CHARACTERS_FILE)
        characters = {c["id"]: c for c in char_data.get("characters", [])}

    planned_scenes: list[dict] = []
    total_estimated_dur = 0.0

    for scene in req.scenes:
        sn = scene.get("sceneNumber", len(planned_scenes) + 1)
        narration = scene.get("narration", "")
        dialogue = scene.get("dialogue", [])

        # --- Duration estimation ---
        # Narration-only pacing with tighter per-scene caps in retention mode.
        all_text = narration
        if str(req.pacingMode).lower() == "retention":
            duration = _estimate_duration(all_text, req.wordsPerMinute, min_seconds=6.0, max_seconds=10.0, pause_seconds=0.8)
        else:
            duration = _estimate_duration(all_text, req.wordsPerMinute, min_seconds=7.0, max_seconds=16.0, pause_seconds=1.0)
        total_estimated_dur += duration

        # --- Image generation task ---
        # STEP 1: Use new structured prompt for consistency
        structured_prompt = _build_structured_prompt(scene, characters, req.seriesStyle)
        
        # Get LoRA info from old prompt bundle for backward compatibility
        prompt_bundle = _build_comfyui_scene_prompt(
            scene, req.seriesStyle, req.negativePrompt, characters,
        )

        # STEP 4: Extract first character's IP-Adapter reference (if available)
        ipadapter_ref = next(
            (characters[cid].get("ipAdapterRef") for cid in (scene.get("charactersInScene", [])) 
             if characters.get(cid) and characters[cid].get("ipAdapterRef")), 
            ""
        )
        
        # Build canonical negative prompt (prevents character drift)
        scene_char_ids = scene.get("charactersInScene", [])
        canonical_negative = _build_canonical_negative(scene_char_ids, req.negativePrompt)
        
        image_task = {
            "id": f"scene_{sn:02d}",
            "prompt": structured_prompt,  # STEP 1: Canonical consistency prompt
            "negativePrompt": canonical_negative,  # STEP 2: Anti-drift negative
            "loras": prompt_bundle["loras"],
            "width": req.imageWidth,
            "height": req.imageHeight,
            "steps": req.imageSteps,
            "cfg": req.imageCfg,
            "seed": -1,
            "filename": f"scene_{sn:02d}.png",
            "checkpoint": req.checkpoint,
            "sampler": req.sampler,
            "scheduler": req.scheduler,
            # STEP 4: IP-Adapter configuration (only if character reference exists)
            "useIPAdapter": bool(ipadapter_ref),
            "ipAdapterWeight": 0.8,
            "ipAdapterNoise": 0.3,
            "ipAdapterImage": ipadapter_ref,
            "referenceDir": "/data/characters",
        }

        # --- Second keyframe for longer scenes (different angle/expression) ---
        # This gives us 2 images to crossfade, avoiding repetitive loop artifacts
        alt_cameras = {"wide_shot": "medium_shot", "medium_shot": "close_up",
                       "close_up": "medium_shot", "over_shoulder": "close_up",
                       "birds_eye": "wide_shot", "low_angle": "medium_shot"}
        alt_camera = alt_cameras.get(scene.get("cameraAngle", "medium_shot"), "medium_shot")
        alt_prompt = structured_prompt.replace(
            scene.get("cameraAngle", "medium_shot").replace("_", " "),
            alt_camera.replace("_", " ")
        )
        second_keyframe_task = {
            "id": f"scene_{sn:02d}_b",
            "prompt": alt_prompt,
            "negativePrompt": image_task["negativePrompt"],
            "loras": prompt_bundle["loras"],
            "width": req.imageWidth,
            "height": req.imageHeight,
            "steps": req.imageSteps,
            "cfg": req.imageCfg,
            "seed": -1,
            "filename": f"scene_{sn:02d}_b.png",
            "checkpoint": req.checkpoint,
            "sampler": req.sampler,
            "scheduler": req.scheduler,
            "useIPAdapter": bool(ipadapter_ref),
            "ipAdapterWeight": 0.8,
            "ipAdapterNoise": 0.3,
            "ipAdapterImage": ipadapter_ref,
            "referenceDir": "/data/characters",
        }

        # --- Animation task (with motion hints for better AnimateDiff) ---
        animation_task = {
            "id": f"anim_{sn:02d}",
            "sourceImage": f"{req.imageDir}/scene_{sn:02d}.png",
            "prompt": structured_prompt,
            "negativePrompt": image_task["negativePrompt"],
            "motionHint": prompt_bundle.get("motionHint", "smooth natural motion"),
            "loras": prompt_bundle["loras"],
            "width": req.imageWidth,
            "height": req.imageHeight,
            # AnimateDiff settings for ComfyUI fallback
            "frames": req.animateFrames,  # 16 frames = 1.3s at 12fps, lightweight
            "steps": req.animateSteps,    # 15 steps (fast, clean)
            "cfg": req.animateCfg,        # 6.5 for cartoon consistency
            "sampler": "euler_ancestral",
            "denoise": 0.45,  # Low denoise = character stays consistent with keyframe
            "seed": -1,
            "motionModule": req.motionModule,
            "checkpoint": req.checkpoint,
            "scheduler": req.scheduler,
            "filename": f"anim_{sn:02d}.mp4",
            # IP-Adapter configuration for animation (character consistency)
            "useIPAdapter": bool(ipadapter_ref),
            "ipAdapterWeight": 0.7,  # Slightly lower for animation to allow motion
            "ipAdapterNoise": 0.4,   # More noise tolerance for motion
            "ipAdapterImage": ipadapter_ref,
            "referenceDir": "/data/characters",
        }

        # --- Audio tasks ---
        audio_tasks = _build_audio_tasks(scene, characters, req.narratorVoice)

        # --- Ken Burns fallback (smart pattern based on camera + emotion) ---
        _kb_camera_patterns = {
            "wide_shot": ["pan_right", "pan_left", "zoom_in_pan"],
            "medium_shot": ["zoom_in", "zoom_out", "zoom_in_pan"],
            "close_up": ["zoom_in", "zoom_out"],
            "over_shoulder": ["pan_right", "pan_left"],
            "birds_eye": ["zoom_out", "zoom_out_pan"],
            "low_angle": ["zoom_in", "zoom_in_pan"],
        }
        cam = scene.get("cameraAngle", "medium_shot")
        kb_options = _kb_camera_patterns.get(cam, ["zoom_in", "pan_right", "zoom_in_pan"])
        kb_pattern = kb_options[(sn - 1) % len(kb_options)]

        ken_burns = {
            "enabled": True,
            "pattern": kb_pattern,
            "zoomRange": [1.0, 1.25],
            "duration": duration,
        }

        # --- Assembly info ---
        assembly = {
            "sceneNumber": sn,
            "imagePath": f"{req.imageDir}/scene_{sn:02d}.png",
            "clipPath": f"{req.clipDir}/anim_{sn:02d}.mp4",
            "audioDir": req.audioDir,
            "audioFiles": [t["filename"] for t in audio_tasks],
            "duration": duration,
            "textOverlay": "",
            "narration": narration,
        }

        planned_scenes.append({
            "sceneNumber": sn,
            "duration": duration,
            "emotion": scene.get("emotion", "neutral"),
            "cameraAngle": scene.get("cameraAngle", "medium_shot"),
            "location": scene.get("location", ""),
            "imageTask": image_task,
            "secondKeyframeTask": second_keyframe_task,
            "animationTask": animation_task,
            "audioTasks": audio_tasks,
            "kenBurns": ken_burns,
            "assembly": assembly,
            "textOverlay": "",
            "narration": narration,
            "dialogue": dialogue,
            "charactersInScene": scene.get("charactersInScene", []),
            # IP-Adapter image references for this scene
            "ipAdapterImages": [
                f"/data/characters/{characters[cid].get('ipAdapterRef')}"
                for cid in scene.get("charactersInScene", [])
                if characters.get(cid) and characters[cid].get("ipAdapterRef")
            ],
        })

    # --- Thumbnail task ---
    thumbnail_task = {
        "id": "thumbnail",
        "prompt": f"{req.seriesStyle}, {req.thumbnailDescription}, dramatic pose, bold composition, vibrant colors, eye-catching, masterpiece",
        "negativePrompt": req.negativePrompt or "blurry, deformed, text, watermark",
        "loras": [],
        "width": 1280,
        "height": 720,
        "steps": req.imageSteps,
        "cfg": req.imageCfg,
        "seed": -1,
        "filename": "thumbnail.png",
        "checkpoint": req.checkpoint,
        "sampler": req.sampler,
        "scheduler": req.scheduler,
    }
    # Add protagonist LoRA to thumbnail
    if req.characterIds and characters:
        main_char = characters.get(req.characterIds[0], {})
        if main_char.get("prompt"):
            thumbnail_task["prompt"] = f"{main_char['prompt']}, {thumbnail_task['prompt']}"
        if main_char.get("loraFile"):
            thumbnail_task["loras"].append({
                "file": main_char["loraFile"],
                "weight": main_char.get("loraWeight", 0.8),
            })

    # --- Full narration TTS task ---
    full_narration_task = {
        "text": req.fullNarration,
        "voice": req.narratorVoice,
        "rate": req.ttsRate,
        "pitch": req.ttsPitch,
        "filename": "full_narration.m4a",
    }

    # --- Summary stats ---
    total_audio_tasks = sum(len(s["audioTasks"]) for s in planned_scenes)
    has_loras = any(
        len(s["imageTask"]["loras"]) > 0 for s in planned_scenes
    )
    manual_image_guide = {
        "directory": "/data/images/manual",
        "requiredFiles": [
            {
                "scene": s["sceneNumber"],
                "filename": s["imageTask"]["filename"],
                "prompt": s["imageTask"]["prompt"],
                "negativePrompt": s["imageTask"]["negativePrompt"],
            }
            for s in planned_scenes
        ],
        "optionalFiles": [
            {
                "scene": s["sceneNumber"],
                "filename": s["secondKeyframeTask"]["filename"],
                "prompt": s["secondKeyframeTask"]["prompt"],
                "negativePrompt": s["secondKeyframeTask"]["negativePrompt"],
            }
            for s in planned_scenes
            if s.get("secondKeyframeTask")
        ],
        "thumbnailFile": "thumbnail.png",
    }

    return {
        "success": True,
        "totalScenes": len(planned_scenes),
        "estimatedDuration": round(total_estimated_dur, 1),
        "totalImageTasks": len(planned_scenes),
        "totalAnimationTasks": len(planned_scenes),
        "totalAudioTasks": total_audio_tasks,
        "hasLoRAs": has_loras,
        "scenes": planned_scenes,
        "thumbnailTask": thumbnail_task,
        "manualImageGuide": manual_image_guide,
        "fullNarrationTask": full_narration_task,
        "metadata": {
            "title": req.title,
            "description": req.description,
            "tags": req.tags,
            "cliffhanger": req.cliffhanger,
            "episodeNumber": req.episodeNumber,
            "seriesName": req.seriesName,
        },
        "settings": {
            "checkpoint": req.checkpoint,
            "motionModule": req.motionModule,
            "imageSize": f"{req.imageWidth}x{req.imageHeight}",
            "videoSize": f"{req.videoWidth}x{req.videoHeight}",
            "fps": req.fps,
            "sampler": req.sampler,
        },
        "directories": {
            "images": req.imageDir,
            "clips": req.clipDir,
            "audio": req.audioDir,
            "video": req.videoDir,
        },
        "plannedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# =============================================================================
# NODE 4 — Image & Animation Generation (Pollinations/manual + ComfyUI fallback)
# =============================================================================
# Primary: Pollinations.ai FLUX — free tier, no GPU needed
#   - Used for keyframe images + thumbnails when not in manual mode
#
# Fallback: ComfyUI (SD1.5) — requires local GPU, only used when Pollinations fails
#   Motion:     AnimateDiff v3 — smoothest motion
#   Adapter:    IP-Adapter Plus SD1.5 — character consistency from reference image
#
# Strategy:  1) Use manual images if imageSource=manual
#            2) Otherwise try Pollinations for all images
#            3) If Pollinations fails -> ComfyUI fallback (SD1.5 + LoRA/IP-Adapter)
#            3) Animate keyframe (AnimateDiff via ComfyUI, 16 frames)
#            4) If AnimateDiff OOMs → Ken Burns zoom/pan fallback (CPU only)
# =============================================================================

class GenerateVisualsRequest(BaseModel):
    """Input from Collect Scene Plan node — full scene plan with tasks."""
    # Scene plan data
    imageTasks: list[dict | None] | None = []
    animationTasks: list[dict | None] | None = []
    scenes: list[dict]
    thumbnailTask: dict | None = {}
    animateFrames: int = 16
    # Image generation strategy
    imageSource: str = "pollinations"  # pollinations | manual
    manualImageDir: str = "/data/images/manual"
    useImagen: bool = False  # Deprecated; Imagen 3 is intentionally disabled
    imagenAspectRatio: str = "16:9"
    # ComfyUI fallback settings (only used when Pollinations fails)
    checkpoint: str = ""  # Leave empty to use COMFYUI_CHECKPOINT env var
    motionModule: str = "v3_sd15_mm.ckpt"
    sampler: str = "euler_ancestral"
    scheduler: str = "normal"
    # IP-Adapter (character consistency without LoRA)
    useIPAdapter: bool = False
    ipAdapterModel: str = "ip-adapter-plus_sd15.safetensors"
    ipAdapterClipVision: str = "clip-vit-h-14-laion2B-s32B-b79K.safetensors"
    # Directories
    imageDir: str = "/data/images"
    clipDir: str = "/data/clips"
    referenceDir: str = "/data/characters"
    # Video
    fps: int = 12
    # Ken Burns fallback (enabled as safety net when AnimateDiff fails)
    kenBurnsEnabled: bool = True  # Fallback to prevent blank screens
    videoWidth: int = 1920
    videoHeight: int = 1080


# ---- ComfyUI helpers (shared by image + animation) ----

def _comfyui_available() -> bool:
    """Check if ComfyUI is reachable."""
    try:
        urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=5)
        return True
    except Exception:
        return False


def _resolve_manual_image(manual_dir: str, filename: str) -> str:
    """Find a manually supplied scene image by exact name or common image extension."""
    if not filename:
        return ""
    exact_path = os.path.join(manual_dir, filename)
    if os.path.exists(exact_path):
        return exact_path
    stem, _ = os.path.splitext(filename)
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = os.path.join(manual_dir, f"{stem}{ext}")
        if os.path.exists(candidate):
            return candidate
    return ""


def _comfyui_queue(workflow: dict) -> str | None:
    """Submit a workflow to ComfyUI and return the prompt_id."""
    payload = json.dumps({
        "prompt": workflow,
        "client_id": str(uuid.uuid4()),
    }).encode()
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            prompt_id = result.get("prompt_id")
            if prompt_id:
                print(f"[ComfyUI] Queued workflow: {prompt_id}")
            else:
                print(f"[ComfyUI] No prompt_id in response: {result}")
            return prompt_id
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"[ComfyUI] HTTP {e.code}: {body[:500]}")
        return None
    except Exception as e:
        print(f"[ComfyUI] Error: {type(e).__name__}: {str(e)[:500]}")
        return None


def _comfyui_wait(prompt_id: str, max_wait: int = 300) -> dict:
    """Poll ComfyUI /history until the prompt completes or times out."""
    start = time.time()
    while time.time() - start < max_wait:
        try:
            r = urllib.request.Request(
                f"{COMFYUI_URL}/history/{prompt_id}", method="GET",
            )
            with urllib.request.urlopen(r, timeout=10) as resp:
                history = json.loads(resp.read().decode())
            if prompt_id in history:
                data = history[prompt_id]
                status = data.get("status", {})
                if status.get("completed") or status.get("status_str") == "success":
                    for _, out in data.get("outputs", {}).items():
                        if "images" in out:
                            return {"success": True, "images": out["images"]}
                    return {"success": True, "images": []}
                if status.get("status_str") == "error":
                    msg = status.get("messages", [])
                    return {"success": False, "error": f"ComfyUI error: {msg}"}
        except Exception:
            pass
        time.sleep(2)
    return {"success": False, "error": "Timeout waiting for ComfyUI"}


def _comfyui_upload_image(image_path: str) -> bool:
    """Upload an image to ComfyUI's input folder so LoadImage can find it."""
    if not os.path.exists(image_path):
        print(f"[ComfyUI] Upload failed: File not found: {image_path}")
        return False
    try:
        filename = os.path.basename(image_path)
        with open(image_path, "rb") as f:
            files = {"image": (filename, f), "overwrite": (None, "true")}
            r = requests.post(f"{COMFYUI_URL}/upload/image", files=files, timeout=30)
            if r.status_code == 200:
                print(f"[ComfyUI] Successfully uploaded {filename}")
                return True
            else:
                print(f"[ComfyUI] Upload failed with status {r.status_code}: {r.text}")
                return False
    except Exception as e:
        print(f"[ComfyUI] Upload exception: {e}")
        return False


def _comfyui_download(image_info: dict, output_path: str) -> bool:
    """Download a generated image from ComfyUI /view endpoint."""
    fn = image_info.get("filename", "")
    sf = image_info.get("subfolder", "")
    it = image_info.get("type", "output")
    params = urllib.parse.urlencode({"filename": fn, "subfolder": sf, "type": it})
    url = f"{COMFYUI_URL}/view?{params}"
    try:
        r = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(r, timeout=30) as resp:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(resp.read())
        return True
    except Exception:
        return False


def _detect_animatediff_nodes() -> dict:
    """Detect available AnimateDiff node types in ComfyUI."""
    try:
        r = urllib.request.Request(f"{COMFYUI_URL}/object_info", method="GET")
        with urllib.request.urlopen(r, timeout=15) as resp:
            info = json.loads(resp.read().decode())
    except Exception:
        return {}

    available: dict[str, str] = {}
    for name in ["ADE_AnimateDiffLoaderGen1", "ADE_LoadAnimateDiffModel",
                 "AnimateDiffLoaderV1"]:
        if name in info:
            available["loader"] = name
            break
    for name in ["ADE_ApplyAnimateDiffModelSimple", "ADE_ApplyAnimateDiffModel"]:
        if name in info:
            available["applier"] = name
            break
    if "VHS_VideoCombine" in info:
        available["video_combine"] = "VHS_VideoCombine"
    # IP-Adapter nodes
    for name in ["IPAdapterApply", "IPAdapter", "IPAdapterModelLoader"]:
        if name in info:
            available["ip_adapter"] = name
            break
    if "CLIPVisionLoader" in info:
        available["clip_vision_loader"] = "CLIPVisionLoader"
    return available


# ---- Workflow builders ----

def _build_keyframe_workflow(
    task: dict, checkpoint: str, sampler: str, scheduler: str,
    loras: list[dict] | None = None,
    force_batch_size: int = 1,
) -> dict:
    """
    Build ComfyUI API workflow for a single keyframe image.
    Supports: checkpoint → optional LoRA chain → CLIP encode → KSampler → VAE → Save
    """
    seed = task.get("seed", -1)
    if seed == -1:
        seed = int.from_bytes(os.urandom(4), "big") % (2**31)

    wf: dict[str, Any] = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint},
        },
    }

    # Model output node — may change if LoRAs are chained
    model_out = ["1", 0]
    clip_out = ["1", 1]

    # Chain LoRAs (if any)
    if loras:
        for i, lora in enumerate(loras):
            node_id = f"lora_{i}"
            wf[node_id] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "lora_name": lora["file"],
                    "strength_model": lora.get("weight", 0.8),
                    "strength_clip": lora.get("weight", 0.8),
                    "model": model_out,
                    "clip": clip_out,
                },
            }
            model_out = [node_id, 0]
            clip_out = [node_id, 1]

    wf.update({
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": task.get("prompt", ""), "clip": clip_out},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": task.get("negativePrompt", ""), "clip": clip_out},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": task.get("width", 576),
                "height": task.get("height", 1024),
                "batch_size": force_batch_size,
            },
        },
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": task.get("steps", 25),
                "cfg": task.get("cfg", 7.0),
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": 1.0,
                "model": model_out,
                "positive": ["3", 0],
                "negative": ["4", 0],
                "latent_image": ["5", 0],
            },
        },
        "7": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["6", 0], "vae": ["1", 2]},
        },
        "8": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": task.get("id", "scene"),
                "images": ["7", 0],
            },
        },
    })

    # --- IP-Adapter Implementation ---
    if task.get("useIPAdapter") and task.get("ipAdapterImage"):
        ref_image = task.get("ipAdapterImage")
        ref_dir = task.get("referenceDir", "/data/characters")
        ref_path = os.path.join(ref_dir, ref_image)

        # Upload reference image to ComfyUI (handles both full path and basename)
        uploaded = False
        if os.path.exists(ref_path):
            uploaded = _comfyui_upload_image(ref_path)
        elif os.path.exists(ref_image):
            uploaded = _comfyui_upload_image(ref_image)
            ref_image = os.path.basename(ref_image)

        if uploaded:
            # Use only the filename for ComfyUI's LoadImage node
            upload_name = os.path.basename(ref_path) if os.path.exists(ref_path) else os.path.basename(ref_image)

            wf.update({
                "ip_model": {
                    "class_type": "IPAdapterModelLoader",
                    "inputs": {"ipadapter_file": "ip-adapter-plus_sd15.safetensors"},
                },
                "ip_clip": {
                    "class_type": "CLIPVisionLoader",
                    "inputs": {"clip_name": "clip-vit-h-14-laion2B-s32B-b79K.safetensors"},
                },
                "ip_load": {
                    "class_type": "LoadImage",
                    "inputs": {"image": upload_name},
                },
                "ip_apply": {
                    "class_type": "IPAdapterApply",
                    "inputs": {
                        "ipadapter": ["ip_model", 0],
                        "clip_vision": ["ip_clip", 0],
                        "image": ["ip_load", 0],
                        "model": model_out,
                        "weight": task.get("ipAdapterWeight", 0.8),
                        "noise": task.get("ipAdapterNoise", 0.3),
                    },
                },
            })
            # Redirect KSampler model input to IP-Adapter output
            wf["6"]["inputs"]["model"] = ["ip_apply", 0]

    return wf


def _build_animatediff_workflow(
    task: dict, checkpoint: str, motion_module: str,
    sampler: str, scheduler: str, ad_nodes: dict,
    loras: list[dict] | None = None,
    source_image_path: str = "",
) -> dict:
    """
    Build ComfyUI API workflow for AnimateDiff animation.
    Uses img2img mode: loads keyframe → VAE encode → AnimateDiff with denoise < 1.0
    This ensures the animation stays consistent with the keyframe image.
    Falls back to txt2video if no source image.
    """
    seed = task.get("seed", -1)
    if seed == -1:
        seed = int.from_bytes(os.urandom(4), "big") % (2**31)

    frames = task.get("frames", 24)
    use_img2img = bool(source_image_path)

    wf: dict[str, Any] = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint},
        },
    }

    model_out = ["1", 0]
    clip_out = ["1", 1]

    # Chain LoRAs
    if loras:
        for i, lora in enumerate(loras):
            nid = f"lora_{i}"
            wf[nid] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "lora_name": lora["file"],
                    "strength_model": lora.get("weight", 0.8),
                    "strength_clip": lora.get("weight", 0.8),
                    "model": model_out,
                    "clip": clip_out,
                },
            }
            model_out = [nid, 0]
            clip_out = [nid, 1]

    # AnimateDiff loader — depends on which nodes are installed
    loader = ad_nodes.get("loader", "ADE_AnimateDiffLoaderGen1")

    if loader in ("ADE_AnimateDiffLoaderGen1", "AnimateDiffLoaderV1"):
        wf["ad_load"] = {
            "class_type": loader,
            "inputs": {
                "model": model_out,
                "model_name": motion_module,
                "beta_schedule": "sqrt_linear (AnimateDiff)",
            },
        }
        model_out = ["ad_load", 0]
    else:
        wf["ad_load"] = {
            "class_type": "ADE_LoadAnimateDiffModel",
            "inputs": {"model_name": motion_module},
        }
        applier = ad_nodes.get("applier", "ADE_ApplyAnimateDiffModelSimple")
        wf["ad_apply"] = {
            "class_type": applier,
            "inputs": {"motion_model": ["ad_load", 0], "model": model_out},
        }
        model_out = ["ad_apply", 0]

    # Build motion-enhanced prompt from scene context
    motion_prompt = task.get("prompt", "")
    motion_hint = task.get("motionHint", "")
    if motion_hint:
        motion_prompt = f"{motion_prompt}, {motion_hint}"

    wf.update({
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": motion_prompt, "clip": clip_out},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": task.get("negativePrompt", ""), "clip": clip_out},
        },
    })

    # --- IP-Adapter Implementation for AnimateDiff ---
    if task.get("useIPAdapter") and task.get("ipAdapterImage"):
        ref_image = task.get("ipAdapterImage")
        ref_dir = task.get("referenceDir", "/data/characters")
        ref_path = os.path.join(ref_dir, ref_image)

        # Upload reference image to ComfyUI
        uploaded = False
        if os.path.exists(ref_path):
            uploaded = _comfyui_upload_image(ref_path)
        elif os.path.exists(ref_image):
            uploaded = _comfyui_upload_image(ref_image)

        if uploaded:
            upload_name = os.path.basename(ref_path) if os.path.exists(ref_path) else os.path.basename(ref_image)
            wf.update({
                "ip_model": {
                    "class_type": "IPAdapterModelLoader",
                    "inputs": {"ipadapter_file": "ip-adapter-plus_sd15.safetensors"},
                },
                "ip_clip": {
                    "class_type": "CLIPVisionLoader",
                    "inputs": {"clip_name": "clip-vit-h-14-laion2B-s32B-b79K.safetensors"},
                },
                "ip_load": {
                    "class_type": "LoadImage",
                    "inputs": {"image": upload_name},
                },
                "ip_apply": {
                    "class_type": "IPAdapterApply",
                    "inputs": {
                        "ipadapter": ["ip_model", 0],
                        "clip_vision": ["ip_clip", 0],
                        "image": ["ip_load", 0],
                        "model": model_out,
                        "weight": task.get("ipAdapterWeight", 0.7),
                        "noise": task.get("ipAdapterNoise", 0.4),
                    },
                },
            })
            model_out = ["ip_apply", 0]

    if use_img2img:
        # IMG2IMG MODE: Load keyframe → repeat batch → VAE encode → KSampler with denoise < 1.0
        wf["img_load"] = {
            "class_type": "LoadImage",
            "inputs": {"image": os.path.basename(source_image_path)},
        }
        wf["img_resize"] = {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["img_load", 0],
                "width": task.get("width", 512),
                "height": task.get("height", 768),
                "upscale_method": "bilinear",
                "crop": "center",
            },
        }
        wf["img_batch"] = {
            "class_type": "RepeatImageBatch",
            "inputs": {
                "image": ["img_resize", 0],
                "amount": frames,
            },
        }
        wf["5"] = {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["img_batch", 0],
                "vae": ["1", 2],
            },
        }
        denoise = task.get("denoise", 0.45)
    else:
        # TXT2VIDEO fallback: empty latent
        wf["5"] = {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": task.get("width", 512),
                "height": task.get("height", 768),
                "batch_size": frames,
            },
        }
        denoise = 1.0

    wf.update({
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": task.get("steps", 20),
                "cfg": task.get("cfg", 7.0),
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": denoise,
                "model": model_out,
                "positive": ["3", 0],
                "negative": ["4", 0],
                "latent_image": ["5", 0],
            },
        },
        "7": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["6", 0], "vae": ["1", 2]},
        },
        "8": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": task.get("id", "anim"),
                "images": ["7", 0],
            },
        },
    })

    return wf


def _frames_to_clip(frame_paths: list[str], output_path: str, fps: int = 12) -> bool:
    """Combine AnimateDiff frames into a smooth MP4 clip with motion interpolation.
    
    Uses minterpolate to double the frame count for smoother cartoon-like motion.
    24 frames at 12fps = 2s base, interpolated to 24fps = smooth 2s animation.
    """
    if not frame_paths:
        return False
    list_path = output_path + ".frames.txt"
    try:
        sorted_frames = sorted(frame_paths)
        with open(list_path, "w", encoding="utf-8") as f:
            for fp in sorted_frames:
                f.write(f"file '{fp}'\nduration {1.0 / fps}\n")
            f.write(f"file '{sorted_frames[-1]}'\n")

        # First pass: create base clip from frames
        # Second pass: use minterpolate for smooth motion between frames
        r = subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
             "-vf", f"minterpolate=fps={fps * 2}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
             "-c:v", "libx264", "-preset", "fast", "-crf", "18",
             "-pix_fmt", "yuv420p", "-r", str(fps * 2), output_path],
            capture_output=True, text=True, timeout=180,
        )
        if r.returncode == 0:
            return True

        # Fallback: simple concat without interpolation if minterpolate fails
        r = subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
             "-c:v", "libx264", "-preset", "fast", "-crf", "18",
             "-pix_fmt", "yuv420p", "-r", str(fps), output_path],
            capture_output=True, text=True, timeout=120,
        )
        return r.returncode == 0
    except Exception:
        return False
    finally:
        try:
            os.remove(list_path)
        except Exception:
            pass


def _ken_burns_fallback(
    image_path: str, output_path: str,
    pattern: str = "zoom_in", duration: float = 5.0,
    zoom_range: tuple[float, float] = (1.0, 1.25),
    width: int = 1920, height: int = 1080, fps: int = 24,
) -> bool:
    """
    CPU-only Ken Burns effect: zoom/pan on a static image.
    Uses FFmpeg zoompan filter — no GPU needed.
    Enhanced: larger zoom range, more patterns, smoother output.
    """
    if not os.path.exists(image_path):
        return False

    total_frames = int(duration * fps)
    z_start, z_end = zoom_range

    # Build zoompan expression based on pattern — all use eased motion
    if pattern == "zoom_in":
        zp = f"zoompan=z='min({z_start}+({z_end}-{z_start})*on/{total_frames},{z_end})':d={total_frames}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={width}x{height}:fps={fps}"
    elif pattern == "zoom_out":
        zp = f"zoompan=z='max({z_end}-({z_end}-{z_start})*on/{total_frames},{z_start})':d={total_frames}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={width}x{height}:fps={fps}"
    elif pattern == "pan_right":
        zp = f"zoompan=z='{(z_start + z_end) / 2}':d={total_frames}:x='(iw-iw/zoom)*on/{total_frames}':y='ih/2-(ih/zoom/2)':s={width}x{height}:fps={fps}"
    elif pattern == "pan_left":
        zp = f"zoompan=z='{(z_start + z_end) / 2}':d={total_frames}:x='(iw-iw/zoom)*(1-on/{total_frames})':y='ih/2-(ih/zoom/2)':s={width}x{height}:fps={fps}"
    elif pattern == "zoom_in_pan":
        # Zoom in while panning right — most dynamic
        zp = f"zoompan=z='min({z_start}+({z_end}-{z_start})*on/{total_frames},{z_end})':d={total_frames}:x='(iw-iw/zoom)*on/{total_frames}':y='ih/3-(ih/zoom/3)':s={width}x{height}:fps={fps}"
    elif pattern == "zoom_out_pan":
        zp = f"zoompan=z='max({z_end}-({z_end}-{z_start})*on/{total_frames},{z_start})':d={total_frames}:x='(iw-iw/zoom)*(1-on/{total_frames})':y='ih/3-(ih/zoom/3)':s={width}x{height}:fps={fps}"
    else:
        zp = f"zoompan=z='min({z_start}+({z_end}-{z_start})*on/{total_frames},{z_end})':d={total_frames}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={width}x{height}:fps={fps}"

    # Scale up source image first for zoompan headroom
    scale_w, scale_h = width * 2, height * 2
    vf = f"scale={scale_w}:{scale_h}:flags=lanczos:force_original_aspect_ratio=increase,crop={scale_w}:{scale_h},{zp}"

    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", image_path,
             "-vf", vf,
             "-c:v", "libx264", "-preset", "fast", "-crf", "18",
             "-pix_fmt", "yuv420p", output_path],
            capture_output=True, text=True, timeout=180,
        )
        return r.returncode == 0
    except Exception:
        return False


# =============================================================================
# SECTION 7 — ENHANCED MOTION: Parallax Depth Simulation
# Creates the illusion of depth by moving foreground/background at different rates.
# Uses FFmpeg crop offsets at different speeds to simulate parallax layers.
# =============================================================================

def _parallax_motion(
    image_path: str, output_path: str,
    duration: float = 5.0,
    direction: str = "right",  # right, left, up, down
    depth_intensity: float = 1.5,  # How much parallax separation
    width: int = 1920, height: int = 1080, fps: int = 24,
) -> bool:
    """
    Simulated parallax motion using split-crop technique.
    
    Splits the image into foreground (bottom 40%) and background (top 60%),
    moves them at different speeds to create depth illusion.
    Uses FFmpeg overlay with motion-offset backgrounds.
    """
    if not os.path.exists(image_path):
        return False

    total_frames = int(duration * fps)
    
    # Calculate pixel movement
    bg_move = int(width * 0.05 * depth_intensity)  # Background moves less
    fg_move = int(width * 0.12 * depth_intensity)  # Foreground moves more
    
    if direction in ("left", "up"):
        bg_move, fg_move = -bg_move, -fg_move

    # Use zoompan with offset for parallax-like movement
    # Background layer (top 60%): slow pan
    # Foreground layer (bottom 40%): faster pan
    # Combined via split + overlay at different rates
    
    # Simpler approach: use zoompan with x-offset animation for cinematic drift
    # The "parallax feel" comes from combining zoom + directional pan
    if direction in ("right", "left"):
        x_expr = f"(iw-iw/zoom)*on/{total_frames}" if direction == "right" else f"(iw-iw/zoom)*(1-on/{total_frames})"
        y_expr = "ih/2-(ih/zoom/2)"
    else:
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = f"(ih-ih/zoom)*on/{total_frames}" if direction == "down" else f"(ih-ih/zoom)*(1-on/{total_frames})"

    # Combine subtle zoom with directional pan for depth feel
    z_start = 1.05
    z_end = 1.15 * depth_intensity
    
    scale_w, scale_h = width * 2, height * 2
    zp = f"zoompan=z='min({z_start}+({z_end}-{z_start})*on/{total_frames},{z_end})':d={total_frames}:x='{x_expr}':y='{y_expr}':s={width}x{height}:fps={fps}"
    vf = f"scale={scale_w}:{scale_h}:flags=lanczos:force_original_aspect_ratio=increase,crop={scale_w}:{scale_h},{zp}"

    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", image_path,
             "-vf", vf,
             "-c:v", "libx264", "-preset", "fast", "-crf", "18",
             "-pix_fmt", "yuv420p", output_path],
            capture_output=True, text=True, timeout=180,
        )
        return r.returncode == 0
    except Exception:
        return False


def _cinematic_drift(
    image_path: str, output_path: str,
    duration: float = 5.0,
    drift_type: str = "gentle",  # gentle, dramatic, breathing
    width: int = 1920, height: int = 1080, fps: int = 24,
) -> bool:
    """
    Subtle cinematic drift effects for emotional moments.
    
    - gentle: very slow zoom with slight drift (contemplative moments)
    - dramatic: faster zoom-in with slight rotation feel
    - breathing: zoom in then out cyclically (creates living feel)
    """
    if not os.path.exists(image_path):
        return False

    total_frames = int(duration * fps)
    scale_w, scale_h = width * 2, height * 2

    if drift_type == "gentle":
        z_start, z_end = 1.0, 1.06
        zp = f"zoompan=z='min({z_start}+({z_end}-{z_start})*on/{total_frames},{z_end})':d={total_frames}:x='iw/2-(iw/zoom/2)+sin(on/30)*5':y='ih/2-(ih/zoom/2)':s={width}x{height}:fps={fps}"
    elif drift_type == "dramatic":
        z_start, z_end = 1.0, 1.2
        zp = f"zoompan=z='min({z_start}+({z_end}-{z_start})*on/{total_frames},{z_end})':d={total_frames}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={width}x{height}:fps={fps}"
    else:  # breathing
        # Sine wave zoom: in and out
        mid = 1.05
        amp = 0.04
        zp = f"zoompan=z='{mid}+{amp}*sin(on*3.14159/{total_frames})':d={total_frames}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={width}x{height}:fps={fps}"

    vf = f"scale={scale_w}:{scale_h}:flags=lanczos:force_original_aspect_ratio=increase,crop={scale_w}:{scale_h},{zp}"

    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", image_path,
             "-vf", vf,
             "-c:v", "libx264", "-preset", "fast", "-crf", "18",
             "-pix_fmt", "yuv420p", output_path],
            capture_output=True, text=True, timeout=180,
        )
        return r.returncode == 0
    except Exception:
        return False


@app.get("/api/comfyui-status")
def comfyui_status():
    """Check ComfyUI availability and detect AnimateDiff + IP-Adapter nodes."""
    reachable = _comfyui_available()
    if not reachable:
        return {"online": False, "animateDiff": False, "ipAdapter": False, "nodes": {}}

    nodes = _detect_animatediff_nodes()
    return {
        "online": True,
        "animateDiff": bool(nodes.get("loader")),
        "ipAdapter": bool(nodes.get("ip_adapter")),
        "videoCombine": bool(nodes.get("video_combine")),
        "nodes": nodes,
    }


@app.post("/api/generate-visuals")
def generate_visuals(req: GenerateVisualsRequest):
    """
    Node 4: Generate all scene keyframes + animate them.

    Pipeline per scene (sequential to fit 6GB VRAM):
      1. Use manual image if imageSource=manual, otherwise try Pollinations
      2. If Pollinations fails -> fall back to ComfyUI (SD1.5 + LoRA)
      3. Submit AnimateDiff workflow for animation (if ComfyUI available)
      4. Wait, download frames, combine to MP4 clip
      5. If AnimateDiff fails → Ken Burns zoom/pan fallback

    Returns per-scene results with paths to images and clips.
    """
    comfyui_online = _comfyui_available()
    # Pollinations is always available (free tier, no key required)

    # Resolve checkpoint: request value > env var > skip ComfyUI if neither set
    if not req.checkpoint and COMFYUI_CHECKPOINT:
        req.checkpoint = COMFYUI_CHECKPOINT
    if not req.checkpoint and comfyui_online:
        print("[Visuals] Warning: No COMFYUI_CHECKPOINT configured — ComfyUI fallback disabled")
        comfyui_online = False

    os.makedirs(req.imageDir, exist_ok=True)
    os.makedirs(req.clipDir, exist_ok=True)

    animate_frames = getattr(req, "animateFrames", 16)
    if req.animationTasks:
        for task in req.animationTasks:
            if task:
                task["frames"] = task.get("frames") or animate_frames

    # Detect AnimateDiff capability (only if ComfyUI is online)
    ad_nodes = _detect_animatediff_nodes() if comfyui_online else {}
    has_animatediff = bool(ad_nodes.get("loader"))

    image_results: list[dict] = []
    clip_results: list[dict] = []
    started_at = time.time()
    manual_used = 0
    pollinations_used = 0
    comfyui_used = 0

    # SECTION 1: Scene Reference Chaining
    # Previous scene's image becomes IP-Adapter reference for next scene (ComfyUI).
    # For Pollinations, we maintain style consistency via the global style lock.
    previous_scene_image: str = ""  # Path to last successful keyframe

    # ---- PHASE 1: Generate ALL keyframe images in parallel (3 concurrent) ----
    safe_image_tasks = [t for t in (req.imageTasks or []) if t]
    safe_anim_tasks = req.animationTasks or []
    safe_scenes = req.scenes or []
    image_source = (req.imageSource or "pollinations").lower().strip()

    print(f"[Generate Visuals] Starting image generation for {len(safe_image_tasks)} scenes (staggered to avoid rate limits)...")

    def _generate_one_keyframe(idx: int, img_task: dict) -> dict:
        """Generate a single scene's keyframe image. Thread-safe for Pollinations."""
        scene_id = img_task.get("id", f"scene_{idx:02d}")
        scene_num = idx + 1
        target_img = os.path.join(req.imageDir, img_task.get("filename", f"{scene_id}.png"))
        result = {"id": scene_id, "scene": scene_num, "target_img": target_img}

        imagen_prompt = img_task.get("prompt", "")
        imagen_negative = img_task.get("negativePrompt", "")
        print(f"[Scene {scene_num}] Generating keyframe via Pollinations ({POLLINATIONS_MODEL})...")

        pollen_result = _retry_with_backoff(
            lambda **kw: _generate_pollinations(**kw),
            max_retries=4,
            prompt=f"{GLOBAL_STYLE_PROMPT}, {imagen_prompt}",
            output_path=target_img,
            width=int(img_task.get("width", 512)),
            height=int(img_task.get("height", 384)),
            model=POLLINATIONS_MODEL,
            negative_prompt=imagen_negative or GLOBAL_NEGATIVE_PROMPT,
        )
        if pollen_result["success"]:
            result.update({
                "success": True, "path": target_img,
                "filename": img_task.get("filename", ""),
                "isLast": img_task.get("isLast", False),
                "method": "pollinations",
            })
            print(f"[Scene {scene_num}] Pollinations success")
            return result

        # ComfyUI fallback (sequential safety — only if needed)
        if comfyui_online:
            print(f"[Scene {scene_num}] Pollinations failed, trying ComfyUI...")
            loras = img_task.get("loras", [])
            wf = _build_keyframe_workflow(
                img_task, req.checkpoint, req.sampler, req.scheduler, loras,
                force_batch_size=1
            )
            pid = _comfyui_queue(wf)
            if pid:
                comp = _comfyui_wait(pid, max_wait=300)
                if comp.get("success") and comp.get("images"):
                    ok = _comfyui_download(comp["images"][0], target_img)
                    if ok:
                        result.update({
                            "success": True, "path": target_img,
                            "filename": img_task.get("filename", ""),
                            "isLast": img_task.get("isLast", False),
                            "method": "comfyui",
                        })
                        return result

        result.update({
            "success": False,
            "error": f"All image providers failed for scene {scene_num}",
            "isLast": img_task.get("isLast", False),
        })
        return result

    # Run image generation with staggered parallelism (2 concurrent + delay between submissions)
    # Pollinations free tier rate-limits aggressively — stagger to avoid 429 errors
    keyframe_results: list[dict] = [None] * len(safe_image_tasks)
    if image_source == "manual":
        missing_images: list[str] = []
        for idx, img_task in enumerate(safe_image_tasks):
            scene_id = img_task.get("id", f"scene_{idx + 1:02d}")
            scene_num = idx + 1
            filename = img_task.get("filename", f"scene_{scene_num:02d}.png")
            manual_path = _resolve_manual_image(req.manualImageDir, filename)
            if not manual_path:
                missing_images.append(filename)
                keyframe_results[idx] = {
                    "id": scene_id,
                    "scene": scene_num,
                    "success": False,
                    "error": f"Missing manual image: {os.path.join(req.manualImageDir, filename)}",
                    "target_img": "",
                }
                continue
            keyframe_results[idx] = {
                "id": scene_id,
                "scene": scene_num,
                "success": True,
                "path": manual_path,
                "filename": os.path.basename(manual_path),
                "isLast": img_task.get("isLast", False),
                "method": "manual",
                "target_img": manual_path,
            }
        if missing_images:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Manual image mode is missing required scene images.",
                    "manualImageDir": req.manualImageDir,
                    "missingImages": missing_images,
                },
            )
    PARALLEL_WORKERS = 2  # Only 2 concurrent to avoid rate limits
    STAGGER_DELAY = 8  # seconds between starting each new request

    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        futures = {}
        for idx, img_task in enumerate([] if image_source == "manual" else safe_image_tasks):
            if not img_task or not isinstance(img_task, dict):
                keyframe_results[idx] = {
                    "id": f"scene_{idx:02d}", "scene": idx + 1,
                    "success": False, "error": "Null imageTask — skipped",
                    "target_img": "",
                }
                continue
            # Stagger submissions to avoid hitting Pollinations rate limits
            if idx > 0:
                time.sleep(STAGGER_DELAY)
            future = executor.submit(_generate_one_keyframe, idx, img_task)
            futures[future] = idx

        for future in as_completed(futures):
            idx = futures[future]
            try:
                keyframe_results[idx] = future.result()
            except Exception as e:
                keyframe_results[idx] = {
                    "id": f"scene_{idx:02d}", "scene": idx + 1,
                    "success": False, "error": str(e), "target_img": "",
                }

    img_gen_elapsed = round(time.time() - started_at, 1)
    successful_imgs = sum(1 for r in keyframe_results if r and r.get("success"))
    print(f"[Generate Visuals] Image generation complete: {successful_imgs}/{len(safe_image_tasks)} in {img_gen_elapsed}s")

    # ---- PHASE 2: Generate second keyframes (sequential to avoid rate limits) ----
    def _generate_second_keyframe(idx: int) -> str:
        """Generate second keyframe for a scene. Returns path or empty string."""
        scene_data = safe_scenes[idx] if idx < len(safe_scenes) else {}
        second_kf_task = scene_data.get("secondKeyframeTask")
        if not second_kf_task or not isinstance(second_kf_task, dict):
            return ""
        scene_num = idx + 1
        s_filename = second_kf_task.get("filename", f"scene_{scene_num:02d}_b.png")
        s_output = os.path.join(req.imageDir, s_filename)
        if image_source == "manual":
            return _resolve_manual_image(req.manualImageDir, s_filename)
        s_prompt = second_kf_task.get("prompt", "")
        s_neg = second_kf_task.get("negativePrompt", "")
        s_result = _generate_pollinations(
            f"{GLOBAL_STYLE_PROMPT}, {s_prompt}",
            s_output,
            int(second_kf_task.get("width", 512)),
            int(second_kf_task.get("height", 384)),
            POLLINATIONS_MODEL,
            s_neg,
        )
        if s_result["success"]:
            return s_output
        return ""

    second_kf_paths: list[str] = [""] * len(safe_image_tasks)
    for idx in range(len(safe_image_tasks)):
        if keyframe_results[idx] and keyframe_results[idx].get("success"):
            scene_data = safe_scenes[idx] if idx < len(safe_scenes) else {}
            if scene_data.get("secondKeyframeTask"):
                if image_source != "manual":
                    time.sleep(5)  # Rate limit protection
                second_kf_paths[idx] = _generate_second_keyframe(idx)
                if second_kf_paths[idx]:
                    pollinations_used += 1
                    print(f"[Scene {idx+1}] Second keyframe generated")

    # ---- PHASE 3: Animate each scene (sequential for ComfyUI, parallel for Ken Burns) ----
    print(f"[Generate Visuals] Starting animation phase...")

    for idx in range(len(safe_image_tasks)):
        img_task = safe_image_tasks[idx] if idx < len(safe_image_tasks) else {}
        kf_result = keyframe_results[idx]
        scene_num = idx + 1

        if not kf_result or not kf_result.get("success"):
            image_results.append(kf_result or {
                "id": f"scene_{idx:02d}", "scene": scene_num,
                "success": False, "error": "Image generation failed",
            })
            clip_results.append({
                "id": f"anim_{scene_num:02d}", "scene": scene_num,
                "success": False, "error": "Skipped — keyframe generation failed",
            })
            continue

        # Count successful providers
        if kf_result.get("method") == "pollinations":
            pollinations_used += 1
        elif kf_result.get("method") == "manual":
            manual_used += 1
        elif kf_result.get("method") == "comfyui":
            comfyui_used += 1

        image_results.append(kf_result)
        target_img = kf_result["target_img"]
        previous_scene_image = target_img
        second_img_path = second_kf_paths[idx]

        # ========== Animate keyframe ==========
        scene_data = safe_scenes[idx] if idx < len(safe_scenes) else {}
        anim_task = safe_anim_tasks[idx] if idx < len(safe_anim_tasks) else {}
        if not anim_task or not isinstance(anim_task, dict):
            anim_task = {}
        clip_id = anim_task.get("id", f"anim_{scene_num:02d}")
        clip_filename = anim_task.get("filename", f"{clip_id}.mp4")
        target_clip = os.path.join(req.clipDir, clip_filename)
        clip_result = {"id": clip_id, "scene": scene_num, "method": "none"}

        anim_success = False
        if has_animatediff and anim_task:
            anim_loras = anim_task.get("loras", [])
            _comfyui_upload_image(target_img)
            ad_wf = _build_animatediff_workflow(
                anim_task, req.checkpoint, req.motionModule,
                req.sampler, req.scheduler, ad_nodes, anim_loras,
                source_image_path=target_img,
            )
            ad_pid = _comfyui_queue(ad_wf)
            if ad_pid:
                ad_comp = _comfyui_wait(ad_pid, max_wait=600)
                if ad_comp.get("success") and ad_comp.get("images"):
                    frame_dir = os.path.join(req.clipDir, f"frames_{clip_id}")
                    os.makedirs(frame_dir, exist_ok=True)
                    frame_paths: list[str] = []
                    for fi, finfo in enumerate(ad_comp["images"]):
                        fp = os.path.join(frame_dir, f"frame_{fi:04d}.png")
                        if _comfyui_download(finfo, fp):
                            frame_paths.append(fp)
                    anim_success = _frames_to_clip(frame_paths, target_clip, req.fps)
                    for fp in frame_paths:
                        try: os.remove(fp)
                        except: pass
                    try: os.rmdir(frame_dir)
                    except: pass
                    if anim_success:
                        clip_result.update({
                            "success": True, "method": "animatediff",
                            "path": target_clip, "filename": clip_filename,
                            "frames": len(frame_paths),
                            "secondKeyframePath": second_img_path,
                            "isLast": anim_task.get("isLast", False) or img_task.get("isLast", False),
                        })

        # Ken Burns fallback
        if not anim_success:
            kb = scene_data.get("kenBurns", {})
            kb_pattern = kb.get("pattern", "zoom_in")
            kb_zoom = kb.get("zoomRange", [1.0, 1.25])
            kb_dur = kb.get("duration", 5.0)
            kb_ok = _ken_burns_fallback(
                target_img, target_clip,
                pattern=kb_pattern, duration=kb_dur,
                zoom_range=tuple(kb_zoom) if len(kb_zoom) >= 2 else (1.0, 1.25),
                width=req.videoWidth, height=req.videoHeight, fps=req.fps,
            )
            if kb_ok:
                clip_result.update({
                    "success": True, "method": "ken_burns",
                    "path": target_clip, "filename": clip_filename,
                    "secondKeyframePath": second_img_path,
                    "isLast": anim_task.get("isLast", False) or img_task.get("isLast", False),
                })
                anim_success = True
                print(f"[Scene {scene_num}] Ken Burns fallback succeeded")
            else:
                clip_result.update({
                    "success": False, "method": "none",
                    "error": "Both AnimateDiff and Ken Burns failed.",
                })

        if not clip_result.get("success") and not anim_success:
            clip_result.setdefault("success", False)
            cause = "AnimateDiff not available" if not has_animatediff else "Animation failed"
            clip_result.setdefault("error", cause)
            clip_result["isLast"] = anim_task.get("isLast", False) or img_task.get("isLast", False)

        clip_results.append(clip_result)
        print(f"[Scene {scene_num}] Animation complete: {clip_result.get('method', 'none')}")

    # ---- Thumbnail generation ----
    thumbnail_result = None
    if req.thumbnailTask:
        os.makedirs(req.imageDir, exist_ok=True)
        thumb_file = req.thumbnailTask.get("filename", "thumbnail.png")
        thumb_path = os.path.join(req.imageDir, thumb_file)
        thumb_generated = False

        if image_source == "manual":
            manual_thumb = _resolve_manual_image(req.manualImageDir, thumb_file)
            if not manual_thumb:
                manual_thumb = _resolve_manual_image(req.manualImageDir, "thumbnail.png")
            if manual_thumb:
                thumbnail_result = {"success": True, "path": manual_thumb, "filename": os.path.basename(manual_thumb), "method": "manual"}
                thumb_generated = True

        # Try Pollinations for thumbnail outside manual mode
        if not thumb_generated and image_source != "manual":
            t_prompt = req.thumbnailTask.get("prompt", "")
            t_neg = req.thumbnailTask.get("negativePrompt", "")
            t_result = _generate_pollinations(f"{GLOBAL_STYLE_PROMPT}, {t_prompt}", thumb_path, 768, 432, POLLINATIONS_MODEL, t_neg)
            if t_result["success"]:
                thumbnail_result = {"success": True, "path": thumb_path, "filename": thumb_file, "method": "pollinations"}
                thumb_generated = True
                pollinations_used += 1

        if thumbnail_result and thumbnail_result.get("method") == "pollinations":
            thumb_generated = True

        # Fallback to ComfyUI
        if not thumb_generated and comfyui_online:
            thumb_loras = req.thumbnailTask.get("loras", [])
            twf = _build_keyframe_workflow(
                req.thumbnailTask, req.checkpoint, req.sampler, req.scheduler, thumb_loras,
                force_batch_size=1
            )
            tpid = _comfyui_queue(twf)
            if tpid:
                tc = _comfyui_wait(tpid, max_wait=300)
                if tc.get("success") and tc.get("images"):
                    tok = _comfyui_download(tc["images"][0], thumb_path)
                    thumbnail_result = {
                        "success": tok,
                        "path": thumb_path if tok else "",
                        "filename": thumb_file,
                        "method": "comfyui",
                    }
                else:
                    thumbnail_result = {"success": False, "error": tc.get("error", "No image")}
            else:
                thumbnail_result = {"success": False, "error": "Queue failed"}

        if not thumb_generated and not thumbnail_result:
            thumbnail_result = {"success": False, "error": "No image generator available"}

    elapsed = round(time.time() - started_at, 1)
    total_images = sum(1 for r in image_results if r.get("success"))
    total_clips = sum(1 for r in clip_results if r.get("success"))
    ad_clips = sum(1 for r in clip_results if r.get("method") == "animatediff")
    kb_clips = sum(1 for r in clip_results if r.get("method") == "ken_burns")

    return {
        "success": total_images > 0 and total_clips > 0,
        "totalScenes": len(safe_image_tasks),
        "imagesGenerated": total_images,
        "clipsGenerated": total_clips,
        "animatediffClips": ad_clips,
        "kenBurnsClips": kb_clips,
        "manualImages": manual_used,
        "pollinationsImages": pollinations_used,
        "comfyuiImages": comfyui_used,
        "primaryGenerator": image_source if image_source in {"manual", "pollinations"} else "comfyui",
        "elapsedSeconds": elapsed,
        "hasAnimateDiff": has_animatediff,
        "images": image_results,
        "clips": clip_results,
        "thumbnail": thumbnail_result,
        "directories": {
            "images": req.imageDir,
            "clips": req.clipDir,
        },
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# =============================================================================
# NODE 5 — Audio Generation (Edge-TTS with word-level timestamps + subtitles)
# =============================================================================
# Key improvements over the shorts workflow:
#   1. Rate "+0%" (natural pace) — NOT "+10%" which was too fast
#   2. Word-level timestamps via edge-tts WordBoundary events
#   3. SRT subtitles generated from ACTUAL audio timing, not estimates
#   4. Per-scene audio + per-scene .srt → assembly node can sync exactly
#   5. Inter-segment pauses (300ms) between narration & dialogue
#   6. Actual duration measured via ffprobe → video clips stretched to match
#
# Best free TTS voices for storytelling (Microsoft Edge Neural):
#   Narrator:    en-US-DavisNeural  (deep, warm, documentary style)
#   Male char:   en-US-GuyNeural    (natural, conversational)
#   Female char: en-US-JennyNeural  (clear, expressive)
#   Child char:  en-US-AnaNeural    (young, energetic)
#   Alt narrator: en-GB-RyanNeural  (British, authoritative)
#   Villain:     en-US-TonyNeural   (older, gravelly)
# =============================================================================

class AudioSegment(BaseModel):
    """One TTS segment (narration line or dialogue line)."""
    type: str  # "narration" or "dialogue"
    text: str
    voice: str = "en-US-DavisNeural"
    character: str = ""  # only for dialogue
    filename: str = ""
    sceneNumber: int = 0

class GenerateAudioRequest(BaseModel):
    """Input from Collect Visual Results node."""
    # Per-scene audio tasks (from scene planner)
    audioTasks: list[dict] = []
    # Full narration (optional — combined narration track)
    fullNarrationTask: dict = {}
    # Scenes with visual results (for sync data)
    scenes: list[dict] = []
    # TTS settings
    narratorVoice: str = "en-GB-RyanNeural"
    rate: str = "-5%"
    pitch: str = "+0Hz"
    # Subtitle settings
    maxWordsPerLine: int = 6
    maxCharsPerLine: int = 35
    # Output
    audioDir: str = "/data/audio"
    subtitleDir: str = "/data/audio"


# ---- Audio helpers ----

def _audio_duration(path: str) -> float:
    """Get exact audio duration in seconds via ffprobe."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=10,
        )
        return round(float(r.stdout.strip()), 3)
    except Exception:
        return 0.0


def _audio_has_valid_stream(path: str) -> bool:
    """Verify the file contains a decodable audio stream."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_name",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=10,
        )
        return r.returncode == 0 and len(r.stdout.strip()) > 0
    except Exception:
        return False


def _convert_to_aac(input_path: str, output_path: str) -> bool:
    """Convert audio to AAC/M4A for guaranteed FFmpeg compatibility."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", input_path,
             "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
             "-movflags", "+faststart", output_path],
            capture_output=True, text=True, timeout=60,
        )
        return (r.returncode == 0
                and os.path.exists(output_path)
                and os.path.getsize(output_path) > 100)
    except Exception:
        return False


async def _tts_with_timestamps(
    text: str, output_path: str, voice: str, rate: str, pitch: str
) -> dict:
    """
    Generate TTS audio with word-level timestamps using edge-tts.

    Returns: {
        "success": bool,
        "duration": float (seconds),
        "engine": str,
        "wordTimestamps": [{"word": str, "start": float, "end": float}, ...]
    }
    """
    import edge_tts

    raw_path = output_path + ".raw.mp3"
    word_timestamps: list[dict] = []

    try:
        comm = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)

        # Stream audio + capture WordBoundary events for timestamps
        with open(raw_path, "wb") as audio_file:
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    audio_file.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    # edge-tts gives offset/duration in 100ns ticks
                    offset_ticks = chunk.get("offset", 0)
                    duration_ticks = chunk.get("duration", 0)
                    word_text = chunk.get("text", "")
                    start_sec = offset_ticks / 10_000_000
                    end_sec = (offset_ticks + duration_ticks) / 10_000_000

                    word_timestamps.append({
                        "word": word_text,
                        "start": round(start_sec, 3),
                        "end": round(end_sec, 3),
                    })

        if not os.path.exists(raw_path) or os.path.getsize(raw_path) < 100:
            return {"success": False, "engine": "edge-tts", "error": "Empty audio",
                    "duration": 0, "wordTimestamps": []}

        # Convert to AAC for reliable playback
        ok = _convert_to_aac(raw_path, output_path)
        if not ok or not _audio_has_valid_stream(output_path):
            return {"success": False, "engine": "edge-tts", "error": "Conversion failed",
                    "duration": 0, "wordTimestamps": []}

        duration = _audio_duration(output_path)
        return {
            "success": True,
            "engine": "edge-tts",
            "duration": duration,
            "wordTimestamps": word_timestamps,
        }

    except Exception as e:
        return {"success": False, "engine": "edge-tts", "error": str(e),
                "duration": 0, "wordTimestamps": []}
    finally:
        try:
            os.remove(raw_path)
        except Exception:
            pass


async def _tts_gtts_fallback(text: str, output_path: str) -> dict:
    """Fallback TTS via gTTS (no timestamps, always works)."""
    from gtts import gTTS

    raw_path = output_path + ".raw.mp3"
    try:
        tts = gTTS(text=text, lang="en", slow=False)
        tts.save(raw_path)
        if not os.path.exists(raw_path) or os.path.getsize(raw_path) < 100:
            return {"success": False, "engine": "gtts", "error": "Empty audio",
                    "duration": 0, "wordTimestamps": []}

        ok = _convert_to_aac(raw_path, output_path)
        if not ok:
            return {"success": False, "engine": "gtts", "error": "Conversion failed",
                    "duration": 0, "wordTimestamps": []}

        duration = _audio_duration(output_path)

        # Estimate word timestamps from duration (gTTS has no word events)
        words = text.split()
        if words and duration > 0:
            time_per_word = duration / len(words)
            timestamps = []
            for i, w in enumerate(words):
                timestamps.append({
                    "word": w,
                    "start": round(i * time_per_word, 3),
                    "end": round((i + 1) * time_per_word, 3),
                })
        else:
            timestamps = []

        return {
            "success": True,
            "engine": "gtts-fallback",
            "duration": duration,
            "wordTimestamps": timestamps,
            "timestampsEstimated": True,
        }
    except Exception as e:
        return {"success": False, "engine": "gtts", "error": str(e),
                "duration": 0, "wordTimestamps": []}
    finally:
        try:
            os.remove(raw_path)
        except Exception:
            pass


async def _tts_generate(
    text: str, output_path: str, voice: str, rate: str, pitch: str
) -> dict:
    """Try edge-tts first (with timestamps), fall back to gTTS."""
    # Attempt edge-tts (best quality + word timestamps)
    for attempt in range(2):
        result = await _tts_with_timestamps(text, output_path, voice, rate, pitch)
        if result["success"]:
            return result
        err = result.get("error", "")
        if "403" in err or "Invalid response" in err:
            break  # Permission error, skip retry
        if attempt == 0:
            await asyncio.sleep(1)

    # Fallback to gTTS
    return await _tts_gtts_fallback(text, output_path)


def _build_srt_from_timestamps(
    word_timestamps: list[dict],
    max_words: int = 6,
    max_chars: int = 35,
    offset_sec: float = 0.0,
) -> str:
    """
    Build SRT subtitle content from word-level timestamps.
    Groups words into subtitle lines with proper duration.

    This ensures subtitles match EXACTLY to the audio timing —
    no estimation, no drift.
    """
    if not word_timestamps:
        return ""

    lines: list[dict] = []
    current_words: list[str] = []
    current_start: float = 0.0

    for wt in word_timestamps:
        word = wt["word"]
        start = wt["start"] + offset_sec

        if not current_words:
            current_start = start

        # Check if adding this word would exceed limits
        test_line = " ".join(current_words + [word])
        if (len(current_words) >= max_words or len(test_line) > max_chars) and current_words:
            # Flush current line
            last_end = wt["start"] + offset_sec  # This word's start = previous group's end
            lines.append({
                "start": current_start,
                "end": last_end,
                "text": " ".join(current_words),
            })
            current_words = [word]
            current_start = start
        else:
            current_words.append(word)

    # Flush remaining
    if current_words and word_timestamps:
        last_ts = word_timestamps[-1]
        lines.append({
            "start": current_start,
            "end": last_ts["end"] + offset_sec,
            "text": " ".join(current_words),
        })

    # Format as SRT
    srt_parts = []
    for i, line in enumerate(lines, 1):
        s = _format_srt_time(line["start"])
        e = _format_srt_time(line["end"])
        srt_parts.append(f"{i}\n{s} --> {e}\n{line['text']}\n")

    return "\n".join(srt_parts)


def _format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT time format: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _concat_audio_with_gaps(
    audio_paths: list[str], output_path: str, gap_ms: int = 300
) -> bool:
    """
    Concatenate multiple audio files with silence gaps between them.
    The gap ensures narration and dialogue don't run together.
    """
    if not audio_paths:
        return False
    if len(audio_paths) == 1:
        import shutil
        try:
            shutil.copy2(audio_paths[0], output_path)
            return True
        except Exception:
            return False

    # Build FFmpeg filter for concatenation with gaps
    filter_parts = []
    inputs = []
    for i, path in enumerate(audio_paths):
        inputs.extend(["-i", path])
        filter_parts.append(f"[{i}:a]aformat=sample_rates=44100:channel_layouts=stereo[a{i}];")

    # Create silence gap audio
    gap_sec = gap_ms / 1000
    gap_filter = f"aevalsrc=0:d={gap_sec}:s=44100:c=stereo[gap];"

    # Interleave audio segments with gaps
    concat_inputs = []
    for i in range(len(audio_paths)):
        concat_inputs.append(f"[a{i}]")
        if i < len(audio_paths) - 1:
            concat_inputs.append("[gap]")

    total_streams = len(audio_paths) + (len(audio_paths) - 1)
    concat_filter = "".join(concat_inputs) + f"concat=n={total_streams}:v=0:a=1[out]"

    full_filter = "".join(filter_parts) + gap_filter + concat_filter

    try:
        cmd = ["ffmpeg", "-y"] + inputs + [
            "-filter_complex", full_filter,
            "-map", "[out]",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
            output_path,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return r.returncode == 0 and os.path.exists(output_path)
    except Exception:
        return False


@app.get("/api/tts-voices")
def tts_voices():
    """List recommended Edge-TTS voices for storytelling."""
    return {
        "recommended": [
            {"id": "en-US-DavisNeural", "use": "narrator", "style": "deep, warm, documentary"},
            {"id": "en-US-GuyNeural", "use": "male character", "style": "natural, conversational"},
            {"id": "en-US-JennyNeural", "use": "female character", "style": "clear, expressive"},
            {"id": "en-US-AnaNeural", "use": "child character", "style": "young, energetic"},
            {"id": "en-GB-RyanNeural", "use": "alt narrator", "style": "British, authoritative"},
            {"id": "en-US-TonyNeural", "use": "villain/elder", "style": "older, gravelly"},
            {"id": "en-US-AriaNeural", "use": "female narrator", "style": "warm, neutral"},
            {"id": "en-US-ChristopherNeural", "use": "male hero", "style": "confident, strong"},
        ],
        "defaultNarrator": "en-US-DavisNeural",
        "pacingTips": {
            "rate": "+0% is natural pace for stories (recommended)",
            "fast": "+10% to +15% is typical for YouTube Shorts (NOT for stories)",
            "slow": "-5% to -10% for dramatic emphasis or children content",
        },
    }


@app.post("/api/generate-audio")
async def generate_audio(req: GenerateAudioRequest):
    """
    Node 5: Generate per-scene audio with word-level subtitle sync.

    For each audio task:
      1. Generate TTS audio via Edge-TTS (with WordBoundary timestamps)
      2. Build SRT subtitle file from actual word timing (not estimates)
      3. Concatenate per-scene segments (narration + dialogue) with 300ms gaps
      4. Measure actual duration via ffprobe → used by Node 6 for video sync

    Pacing philosophy:
      - Rate "+0%" = natural speech pace (not rushed like shorts)
      - Word-level timestamps = subtitles match audio exactly
      - Assembly node (Node 6) stretches/trims video clips to match audio duration
    """
    os.makedirs(req.audioDir, exist_ok=True)
    os.makedirs(req.subtitleDir, exist_ok=True)

    scene_results: list[dict] = []
    overall_srt_entries: list[dict] = []  # For combined full-video subtitle
    cumulative_offset: float = 0.0
    started_at = time.time()

    # ---- Group audio tasks by scene ----
    tasks_by_scene: dict[int, list[dict]] = {}
    for task in req.audioTasks:
        sn = task.get("sceneNumber", 0)
        if sn not in tasks_by_scene:
            tasks_by_scene[sn] = []
        tasks_by_scene[sn].append(task)

    # Determine scene order from the scenes list (or audioTasks order)
    scene_numbers: list[int] = []
    if req.scenes:
        scene_numbers = [s.get("sceneNumber", i + 1) for i, s in enumerate(req.scenes)]
    else:
        scene_numbers = sorted(tasks_by_scene.keys())

    # ---- Process each scene's audio segments ----
    for sn in scene_numbers:
        scene_tasks = tasks_by_scene.get(sn, [])
        if not scene_tasks:
            scene_results.append({
                "sceneNumber": sn, "success": False,
                "error": "No audio tasks for this scene",
            })
            continue

        segment_paths: list[str] = []
        segment_timestamps: list[dict] = []
        all_word_timestamps: list[dict] = []
        local_offset: float = 0.0

        for task in scene_tasks:
            text = task.get("text", "").strip()
            if not text:
                continue

            # Voice lock enforcement: resolve from locked map, never random
            if task.get("type") == "dialogue" and task.get("character"):
                voice = _resolve_voice(task["character"], req.narratorVoice)
            else:
                voice = task.get("voice", req.narratorVoice)
            filename = task.get("filename", f"scene_{sn:02d}_{task.get('type', 'audio')}.m4a")
            seg_path = os.path.join(req.audioDir, filename)

            # Generate TTS with word timestamps
            tts_result = await _tts_generate(text, seg_path, voice, req.rate, req.pitch)

            if tts_result["success"]:
                segment_paths.append(seg_path)
                seg_dur = tts_result["duration"]

                # Offset word timestamps for this segment within the scene
                seg_word_ts = []
                for wt in tts_result.get("wordTimestamps", []):
                    seg_word_ts.append({
                        "word": wt["word"],
                        "start": round(wt["start"] + local_offset, 3),
                        "end": round(wt["end"] + local_offset, 3),
                    })
                all_word_timestamps.extend(seg_word_ts)

                segment_timestamps.append({
                    "type": task.get("type", "audio"),
                    "character": task.get("character", ""),
                    "filename": filename,
                    "duration": seg_dur,
                    "offsetInScene": round(local_offset, 3),
                    "engine": tts_result["engine"],
                    "wordCount": len(tts_result.get("wordTimestamps", [])),
                })

                local_offset += seg_dur + 0.3  # 300ms gap between segments
            else:
                segment_timestamps.append({
                    "type": task.get("type", "audio"),
                    "character": task.get("character", ""),
                    "filename": filename,
                    "success": False,
                    "error": tts_result.get("error", "TTS failed"),
                })

        # ---- Concatenate scene audio segments ----
        combined_filename = f"scene_{sn:02d}_combined.m4a"
        combined_path = os.path.join(req.audioDir, combined_filename)

        if len(segment_paths) > 1:
            concat_ok = _concat_audio_with_gaps(segment_paths, combined_path, gap_ms=300)
        elif len(segment_paths) == 1:
            import shutil
            try:
                shutil.copy2(segment_paths[0], combined_path)
                concat_ok = True
            except Exception:
                concat_ok = False
        else:
            concat_ok = False

        scene_duration = _audio_duration(combined_path) if concat_ok else 0.0

        # ---- Build per-scene SRT subtitle ----
        scene_srt = _build_srt_from_timestamps(
            all_word_timestamps,
            max_words=req.maxWordsPerLine,
            max_chars=req.maxCharsPerLine,
            offset_sec=0.0,  # Scene-local timing
        )
        scene_srt_filename = f"scene_{sn:02d}.srt"
        scene_srt_path = os.path.join(req.subtitleDir, scene_srt_filename)
        if scene_srt:
            with open(scene_srt_path, "w", encoding="utf-8") as f:
                f.write(scene_srt)

        # ---- Build entries for full-video SRT (with cumulative offset) ----
        for wt in all_word_timestamps:
            overall_srt_entries.append({
                "word": wt["word"],
                "start": round(wt["start"] + cumulative_offset, 3),
                "end": round(wt["end"] + cumulative_offset, 3),
            })

        cumulative_offset += scene_duration + 0.5  # 500ms gap between scenes

        scene_results.append({
            "sceneNumber": sn,
            "success": concat_ok and scene_duration > 0,
            "combinedAudioPath": combined_path if concat_ok else "",
            "combinedAudioFilename": combined_filename,
            "duration": scene_duration,
            "subtitlePath": scene_srt_path if scene_srt else "",
            "subtitleFilename": scene_srt_filename if scene_srt else "",
            "segments": segment_timestamps,
            "totalSegments": len(segment_paths),
            "wordTimestamps": all_word_timestamps,
        })

    # ---- Generate full combined narration (optional) ----
    full_narration_result = None
    if req.fullNarrationTask and req.fullNarrationTask.get("text"):
        fn_text = req.fullNarrationTask["text"]
        fn_voice = req.fullNarrationTask.get("voice", req.narratorVoice)
        fn_rate = req.fullNarrationTask.get("rate", req.rate)
        fn_pitch = req.fullNarrationTask.get("pitch", req.pitch)
        fn_filename = req.fullNarrationTask.get("filename", "full_narration.m4a")
        fn_path = os.path.join(req.audioDir, fn_filename)

        fn_result = await _tts_generate(fn_text, fn_path, fn_voice, fn_rate, fn_pitch)

        if fn_result["success"]:
            # Build SRT for full narration
            fn_srt = _build_srt_from_timestamps(
                fn_result.get("wordTimestamps", []),
                max_words=req.maxWordsPerLine,
                max_chars=req.maxCharsPerLine,
            )
            fn_srt_path = os.path.join(req.subtitleDir, "full_narration.srt")
            if fn_srt:
                with open(fn_srt_path, "w", encoding="utf-8") as f:
                    f.write(fn_srt)

            full_narration_result = {
                "success": True,
                "path": fn_path,
                "filename": fn_filename,
                "duration": fn_result["duration"],
                "engine": fn_result["engine"],
                "subtitlePath": fn_srt_path if fn_srt else "",
                "subtitleFilename": "full_narration.srt" if fn_srt else "",
            }
        else:
            full_narration_result = {
                "success": False,
                "error": fn_result.get("error", "Full narration TTS failed"),
            }

    # ---- Build full-video SRT (all scenes combined with offsets) ----
    full_video_srt = _build_srt_from_timestamps(
        overall_srt_entries,
        max_words=req.maxWordsPerLine,
        max_chars=req.maxCharsPerLine,
    )
    full_video_srt_path = os.path.join(req.subtitleDir, "full_video.srt")
    if full_video_srt:
        with open(full_video_srt_path, "w", encoding="utf-8") as f:
            f.write(full_video_srt)

    elapsed = round(time.time() - started_at, 1)
    total_scenes_ok = sum(1 for r in scene_results if r.get("success"))
    total_duration = sum(r.get("duration", 0) for r in scene_results)

    return {
        "success": total_scenes_ok > 0,
        "totalScenes": len(scene_numbers),
        "scenesGenerated": total_scenes_ok,
        "totalDuration": round(total_duration, 2),
        "elapsedSeconds": elapsed,
        "rate": req.rate,
        "narratorVoice": req.narratorVoice,
        "scenes": scene_results,
        "fullNarration": full_narration_result,
        "fullVideoSubtitle": {
            "path": full_video_srt_path if full_video_srt else "",
            "filename": "full_video.srt" if full_video_srt else "",
        },
        "syncInfo": {
            "description": "Each scene has exact word timestamps and audio duration. "
                           "Node 6 (Assembly) should extend/trim each video clip to match "
                           "the scene audio duration for perfect sync.",
            "sceneDurations": {
                r["sceneNumber"]: r["duration"]
                for r in scene_results if r.get("success")
            },
            "interSceneGap": 0.5,
        },
        "directories": {
            "audio": req.audioDir,
            "subtitles": req.subtitleDir,
        },
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# =============================================================================
# NODE 5B — SadTalker Lip Sync (applied to dialogue scenes)
# =============================================================================
# Runs on host machine via HTTP (needs GPU).
# Takes: face image + audio clip → outputs video with lip sync + head motion.
#
# Integration:
#   - Only applied to DIALOGUE segments (not narration)
#   - Uses the scene keyframe image as the face source
#   - Audio from the character's TTS dialogue line
#   - Output: short video clip that replaces static keyframe during dialogue
#
# SadTalker server API (runs on host:8189):
#   POST /api/animate
#     body: { "image": base64_png, "audio": base64_wav, "still_mode": false }
#     returns: { "success": true, "video": base64_mp4 }
# =============================================================================


def _sadtalker_available() -> bool:
    """Check if SadTalker server is reachable."""
    try:
        urllib.request.urlopen(f"{SADTALKER_URL}/health", timeout=5)
        return True
    except Exception:
        return False


def _sadtalker_animate(image_path: str, audio_path: str, output_path: str,
                       still_mode: bool = False, expression_scale: float = 1.0) -> bool:
    """
    Send image + audio to SadTalker server and save the lip-synced video.
    
    Args:
        image_path: Path to face/character image (PNG)
        audio_path: Path to dialogue audio (M4A/WAV/MP3)
        output_path: Where to save the output video (MP4)
        still_mode: If True, only mouth moves (no head motion) — better for cartoon
        expression_scale: How much expression to add (0.5-1.5, cartoon = 1.0)
    
    Returns: True if successful
    """
    import base64

    if not os.path.exists(image_path) or not os.path.exists(audio_path):
        return False

    try:
        # Read and encode image
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        # Convert audio to WAV first (SadTalker expects WAV)
        wav_path = output_path + ".temp.wav"
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1",
             "-c:a", "pcm_s16le", wav_path],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0 or not os.path.exists(wav_path):
            return False

        with open(wav_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")

        # Call SadTalker server
        payload = json.dumps({
            "image": image_b64,
            "audio": audio_b64,
            "still_mode": still_mode,
            "expression_scale": expression_scale,
            "preprocess": "crop",  # crop face region for better results
        }).encode("utf-8")

        req_obj = urllib.request.Request(
            f"{SADTALKER_URL}/api/animate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urllib.request.urlopen(req_obj, timeout=120)
        result = json.loads(resp.read().decode("utf-8"))

        if result.get("success") and result.get("video"):
            video_b64 = result["video"]
            with open(output_path, "wb") as f:
                f.write(base64.b64decode(video_b64))
            return os.path.exists(output_path) and os.path.getsize(output_path) > 100
        return False

    except Exception as e:
        print(f"[SadTalker] Error: {e}")
        return False
    finally:
        # Cleanup temp wav
        try:
            os.remove(wav_path)
        except Exception:
            pass


class LipSyncRequest(BaseModel):
    """Request to apply SadTalker lip sync to dialogue scenes."""
    scenes: list[dict]  # Scenes with imageResult, clipResult, audioResult
    imageDir: str = "/data/images"
    audioDir: str = "/data/audio"
    clipDir: str = "/data/clips"
    stillMode: bool = True  # True = only mouth moves (best for cartoon)
    expressionScale: float = 1.0


@app.post("/api/lip-sync")
def apply_lip_sync(req: LipSyncRequest):
    """
    Node 5B: Apply SadTalker lip sync to dialogue scenes.
    
    For each scene with dialogue:
      1. Take the keyframe image (face source)
      2. Take the dialogue audio clip
      3. Run SadTalker → get lip-synced video
      4. Replace/augment the scene clip with lip-synced version
    
    Only processes scenes where characters are speaking (has dialogue audio).
    Skips if SadTalker is not available (graceful fallback).
    """
    if not _sadtalker_available():
        return {
            "success": True,
            "skipped": True,
            "reason": "SadTalker not available — lip sync skipped (video will use static faces)",
            "scenesProcessed": 0,
            "scenes": req.scenes,
        }

    os.makedirs(req.clipDir, exist_ok=True)
    results = []
    processed = 0

    for scene in req.scenes:
        sn = scene.get("sceneNumber", 0)
        image_result = scene.get("imageResult", {})
        image_path = image_result.get("path", "")
        audio_result = scene.get("audioResult", {})

        # Only apply lip sync if scene has dialogue audio files
        dialogue_files = audio_result.get("dialogueFiles", [])
        if not dialogue_files or not image_path or not os.path.exists(image_path):
            results.append({"scene": sn, "lipSync": False, "reason": "No dialogue or image"})
            continue

        # Use the first dialogue audio for lip sync (primary character speaking)
        primary_dialogue = None
        for df in dialogue_files:
            dial_path = df if os.path.isabs(df) else os.path.join(req.audioDir, df)
            if os.path.exists(dial_path):
                primary_dialogue = dial_path
                break

        if not primary_dialogue:
            results.append({"scene": sn, "lipSync": False, "reason": "Dialogue audio not found"})
            continue

        # Generate lip-synced clip
        lipsync_output = os.path.join(req.clipDir, f"scene_{sn:02d}_lipsync.mp4")
        success = _sadtalker_animate(
            image_path=image_path,
            audio_path=primary_dialogue,
            output_path=lipsync_output,
            still_mode=req.stillMode,
            expression_scale=req.expressionScale,
        )

        if success:
            processed += 1
            # Store lip sync clip path in scene data
            scene.setdefault("clipResult", {})["lipSyncPath"] = lipsync_output
            scene.setdefault("clipResult", {})["hasLipSync"] = True
            results.append({"scene": sn, "lipSync": True, "path": lipsync_output})
            print(f"[Scene {sn}] Lip sync applied successfully")
        else:
            results.append({"scene": sn, "lipSync": False, "reason": "SadTalker failed"})

    return {
        "success": True,
        "skipped": False,
        "scenesProcessed": processed,
        "totalScenes": len(req.scenes),
        "results": results,
        "scenes": req.scenes,  # Pass through with lipSync paths added
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


@app.get("/api/sadtalker-status")
def sadtalker_status():
    """Check if SadTalker server is available."""
    available = _sadtalker_available()
    return {
        "online": available,
        "url": SADTALKER_URL,
        "note": "SadTalker adds lip sync to dialogue scenes. Install on host with GPU." if not available else "Ready for lip sync",
    }


# =============================================================================
# VOICE LOCK SYSTEM (Task 9)
# Ensures each character's voice NEVER changes between scenes or episodes.
# Voice assignments are frozen in characters.json and enforced here.
# =============================================================================

# Locked voice map — loaded from characters.json, never randomized
_VOICE_LOCK_CACHE: dict = {}


def _get_locked_voices() -> dict:
    """Get the frozen voice assignments for all characters.
    Returns {character_id: voice_id, "__narrator__": narrator_voice}"""
    global _VOICE_LOCK_CACHE
    if _VOICE_LOCK_CACHE:
        return _VOICE_LOCK_CACHE

    voices = {"__narrator__": "en-GB-RyanNeural"}
    if os.path.exists(CHARACTERS_FILE):
        data = _load_json(CHARACTERS_FILE)
        for char in data.get("characters", []):
            cid = char["id"]
            voice = char.get("voiceId", "")
            if voice:
                voices[cid] = voice
                # Also map by name (case-insensitive) for dialogue matching
                voices[char.get("name", "").lower()] = voice
    _VOICE_LOCK_CACHE = voices
    return voices


def _resolve_voice(character_name: str, fallback: str = "en-US-DavisNeural") -> str:
    """Resolve a character name to their locked voice. Never returns random voice."""
    voices = _get_locked_voices()
    # Try exact name match (case-insensitive)
    name_lower = character_name.lower().strip()
    if name_lower in voices:
        return voices[name_lower]
    # Try partial match (e.g., "Finn" matches "captain_finn")
    for key, voice in voices.items():
        if name_lower in key or key in name_lower:
            return voice
    # Narrator fallback
    return voices.get("__narrator__", fallback)


@app.get("/api/voice-lock")
def get_voice_lock():
    """Return the frozen voice assignments for all characters."""
    voices = _get_locked_voices()
    return {
        "success": True,
        "voiceMap": voices,
        "note": "These voice assignments are PERMANENT. Never swap voices between characters.",
        "narrator": voices.get("__narrator__", "en-GB-RyanNeural"),
    }


# =============================================================================
# BACKGROUND MUSIC SYSTEM (Task 11)
# Selects mood-appropriate background music for each scene.
# Sources: pre-downloaded royalty-free library in /data/music/
# Each mood maps to specific tracks. Fallback: silence.
# =============================================================================

# Mood-to-music mapping (tracks stored in /data/music/)
MOOD_MUSIC_MAP = {
    "adventure": ["adventure_theme.mp3", "exploration.mp3", "quest_begin.mp3"],
    "mystery": ["mystery_ambient.mp3", "curious_discovery.mp3", "suspense_light.mp3"],
    "happy": ["cheerful_morning.mp3", "playful_friends.mp3", "sunny_day.mp3"],
    "sad": ["gentle_piano.mp3", "melancholy_strings.mp3", "rainy_window.mp3"],
    "excited": ["action_building.mp3", "energy_rising.mp3", "triumph_near.mp3"],
    "scared": ["spooky_ambient.mp3", "dark_forest.mp3", "tension_building.mp3"],
    "determined": ["hero_march.mp3", "rising_courage.mp3", "power_up.mp3"],
    "curious": ["wonder_theme.mp3", "discovery_light.mp3", "tinkering.mp3"],
    "surprised": ["plot_twist.mp3", "reveal_sting.mp3", "gasp_moment.mp3"],
    "warm": ["friendship_theme.mp3", "cozy_evening.mp3", "heartfelt.mp3"],
    "triumphant": ["victory_fanfare.mp3", "celebration.mp3", "heroes_return.mp3"],
    "neutral": ["ambient_soft.mp3", "gentle_background.mp3"],
}

# Scene position music (overrides mood for specific story beats)
SCENE_POSITION_MUSIC = {
    "opening": ["series_intro.mp3", "episode_start.mp3"],
    "climax": ["climax_intense.mp3", "boss_battle.mp3"],
    "resolution": ["resolution_warm.mp3", "all_is_well.mp3"],
    "cliffhanger": ["cliffhanger_sting.mp3", "to_be_continued.mp3"],
}


def _select_background_music(
    scene: dict,
    scene_index: int,
    total_scenes: int,
    music_dir: str,
) -> str:
    """Select appropriate background music for a scene based on mood and position.
    
    Returns path to music file, or empty string if no suitable music found.
    """
    # Determine scene position in story arc
    position = ""
    if scene_index == 0:
        position = "opening"
    elif scene_index >= total_scenes - 1:
        position = "cliffhanger"
    elif scene_index >= total_scenes - 2:
        position = "resolution"
    elif scene_index == total_scenes // 2:
        position = "climax"

    # Try position-specific music first
    if position and position in SCENE_POSITION_MUSIC:
        for track in SCENE_POSITION_MUSIC[position]:
            path = os.path.join(music_dir, track)
            if os.path.exists(path):
                return path

    # Fall back to mood-based music
    mood = scene.get("emotion", "neutral")
    candidates = MOOD_MUSIC_MAP.get(mood, MOOD_MUSIC_MAP.get("neutral", []))

    for track in candidates:
        path = os.path.join(music_dir, track)
        if os.path.exists(path):
            return path

    # Try any available music as last resort
    if os.path.exists(music_dir):
        for f in os.listdir(music_dir):
            if f.endswith((".mp3", ".m4a", ".wav", ".ogg")):
                return os.path.join(music_dir, f)

    return ""


@app.get("/api/music-library")
def get_music_library():
    """List available background music tracks and mood mappings."""
    music_dir = os.path.join(DATA_DIR, "music")
    available_tracks = []
    if os.path.exists(music_dir):
        for f in sorted(os.listdir(music_dir)):
            if f.endswith((".mp3", ".m4a", ".wav", ".ogg")):
                path = os.path.join(music_dir, f)
                duration = _audio_duration(path) if os.path.exists(path) else 0
                available_tracks.append({
                    "filename": f,
                    "path": path,
                    "duration": duration,
                })

    return {
        "success": True,
        "totalTracks": len(available_tracks),
        "tracks": available_tracks,
        "moodMap": MOOD_MUSIC_MAP,
        "positionMap": SCENE_POSITION_MUSIC,
        "musicDir": music_dir,
        "tip": "Add .mp3/.m4a files to /data/music/ named to match the mood map keys above.",
    }


# =============================================================================
# AUDIO MIXING PIPELINE (Task 12)
# Layers: Voice (primary) + Background Music (ducked) + SFX (optional)
# Uses FFmpeg sidechaincompress for professional voice ducking.
# =============================================================================

class AudioMixRequest(BaseModel):
    """Request to mix voice, music, and SFX into final audio."""
    scenes: list[dict] = []          # Scene results from generate-audio
    musicDir: str = "/data/music"
    sfxDir: str = "/data/music/sfx"
    outputDir: str = "/data/audio"
    # Music settings
    musicVolume: float = 0.15        # Background music volume (0.0-1.0) — low for stories
    musicFadeIn: float = 2.0         # Fade music in at episode start (seconds)
    musicFadeOut: float = 3.0        # Fade music out at episode end (seconds)
    duckingEnabled: bool = True      # Duck music when voice is speaking
    duckingAmount: float = 0.3       # How much to reduce music during speech (0=silence, 1=no duck)
    # SFX settings
    sfxVolume: float = 0.4
    # Global
    crossfadeDuration: float = 1.0   # Crossfade between scene music tracks


@app.post("/api/mix-audio")
def mix_audio(req: AudioMixRequest):
    """
    Mix voice tracks with background music and optional SFX.
    
    Pipeline:
    1. For each scene, select mood-appropriate background music
    2. Loop/trim music to match scene audio duration
    3. Apply voice ducking (music gets quieter when voice plays)
    4. Concatenate all scenes with music crossfades
    5. Add episode-level fade in/out
    
    Output: /data/audio/mixed_full.m4a (voice + music combined)
    """
    os.makedirs(req.outputDir, exist_ok=True)
    music_dir = req.musicDir

    scene_mixes: list[dict] = []
    mixed_scene_paths: list[str] = []

    for idx, scene in enumerate(req.scenes):
        sn = scene.get("sceneNumber", idx + 1)
        voice_path = scene.get("combinedAudioPath", "")
        if not voice_path or not os.path.exists(voice_path):
            scene_mixes.append({"sceneNumber": sn, "success": False, "error": "No voice audio"})
            continue

        voice_dur = _audio_duration(voice_path)
        if voice_dur <= 0:
            scene_mixes.append({"sceneNumber": sn, "success": False, "error": "Zero duration voice"})
            continue

        # Select background music for this scene
        music_path = _select_background_music(scene, idx, len(req.scenes), music_dir)

        output_path = os.path.join(req.outputDir, f"scene_{sn:02d}_mixed.m4a")

        if music_path and os.path.exists(music_path):
            # Mix voice + music with ducking
            success = _mix_voice_and_music(
                voice_path=voice_path,
                music_path=music_path,
                output_path=output_path,
                target_duration=voice_dur,
                music_volume=req.musicVolume,
                ducking_enabled=req.duckingEnabled,
                ducking_amount=req.duckingAmount,
            )
        else:
            # No music available — just copy voice
            try:
                shutil.copy2(voice_path, output_path)
                success = True
            except Exception:
                success = False

        if success and os.path.exists(output_path):
            mixed_scene_paths.append(output_path)
            scene_mixes.append({
                "sceneNumber": sn,
                "success": True,
                "path": output_path,
                "duration": _audio_duration(output_path),
                "hasMusic": bool(music_path),
                "musicTrack": os.path.basename(music_path) if music_path else "",
            })
        else:
            scene_mixes.append({"sceneNumber": sn, "success": False, "error": "Mix failed"})

    # Concatenate all mixed scene audio into final track
    final_path = os.path.join(req.outputDir, "mixed_full.m4a")
    if mixed_scene_paths:
        concat_ok = _concat_audio_with_gaps(mixed_scene_paths, final_path, gap_ms=400)
    else:
        concat_ok = False

    # Apply episode-level fade in/out to final track
    if concat_ok and os.path.exists(final_path):
        final_dur = _audio_duration(final_path)
        _apply_audio_fades(final_path, req.musicFadeIn, req.musicFadeOut, final_dur)

    total_duration = _audio_duration(final_path) if concat_ok else 0

    return {
        "success": concat_ok,
        "outputPath": final_path if concat_ok else "",
        "totalDuration": total_duration,
        "scenes": scene_mixes,
        "scenesWithMusic": sum(1 for s in scene_mixes if s.get("hasMusic")),
        "totalScenes": len(req.scenes),
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _mix_voice_and_music(
    voice_path: str,
    music_path: str,
    output_path: str,
    target_duration: float,
    music_volume: float = 0.15,
    ducking_enabled: bool = True,
    ducking_amount: float = 0.3,
) -> bool:
    """Mix voice and background music with optional ducking.
    
    Ducking: music volume drops when voice is speaking, rises in pauses.
    Uses FFmpeg sidechaincompress filter for smooth professional ducking.
    """
    try:
        if ducking_enabled:
            # Professional ducking using sidechaincompress
            # Voice triggers compression on music → music ducks under speech
            filter_complex = (
                f"[1:a]aloop=loop=-1:size=2e+09,atrim=0:{target_duration},"
                f"volume={music_volume}[music];"
                f"[0:a]aformat=sample_rates=44100:channel_layouts=stereo[voice];"
                f"[music][voice]sidechaincompress="
                f"threshold=0.02:ratio=6:attack=200:release=1000:"
                f"level_sc=1:mix={1.0 - ducking_amount}[ducked];"
                f"[voice][ducked]amix=inputs=2:duration=first:"
                f"weights=1 {music_volume}[out]"
            )
        else:
            # Simple mix without ducking
            filter_complex = (
                f"[1:a]aloop=loop=-1:size=2e+09,atrim=0:{target_duration},"
                f"volume={music_volume}[music];"
                f"[0:a]aformat=sample_rates=44100:channel_layouts=stereo[voice];"
                f"[voice][music]amix=inputs=2:duration=first:"
                f"weights=1 {music_volume}[out]"
            )

        cmd = [
            "ffmpeg", "-y",
            "-i", voice_path,
            "-i", music_path,
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            output_path,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return r.returncode == 0 and os.path.exists(output_path)
    except Exception:
        return False


def _apply_audio_fades(path: str, fade_in: float, fade_out: float, duration: float) -> bool:
    """Apply fade-in and fade-out to an audio file in-place."""
    if duration <= 0:
        return False
    temp_path = path + ".faded.m4a"
    fade_out_start = max(0, duration - fade_out)
    try:
        cmd = [
            "ffmpeg", "-y", "-i", path,
            "-af", f"afade=t=in:st=0:d={fade_in},afade=t=out:st={fade_out_start}:d={fade_out}",
            "-c:a", "aac", "-b:a", "192k",
            temp_path,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode == 0 and os.path.exists(temp_path):
            os.replace(temp_path, path)
            return True
        return False
    except Exception:
        return False


# =============================================================================
# CINEMATIC TRANSITION LIBRARY (Task 14)
# Story-beat aware transitions between scenes.
# Maps emotional transitions to appropriate visual effects.
# =============================================================================

# Transition type definitions with FFmpeg xfade filter names
TRANSITION_LIBRARY = {
    # Soft transitions (default for most scenes)
    "dissolve": {"filter": "fade", "duration": 0.8, "use": "default, between similar moods"},
    "fadeblack": {"filter": "fadeblack", "duration": 1.0, "use": "time passing, scene change"},
    "fadewhite": {"filter": "fadewhite", "duration": 0.8, "use": "dream/memory, bright discovery"},

    # Directional transitions (movement/adventure scenes)
    "wipeleft": {"filter": "wipeleft", "duration": 0.6, "use": "traveling, moving forward"},
    "wiperight": {"filter": "wiperight", "duration": 0.6, "use": "flashback, going back"},
    "wipeup": {"filter": "wipeup", "duration": 0.6, "use": "ascending, flying, growing"},
    "wipedown": {"filter": "wipedown", "duration": 0.6, "use": "descending, falling, diving"},

    # Dynamic transitions (action/excitement)
    "circleopen": {"filter": "circleopen", "duration": 0.5, "use": "reveal, spotlight, discovery"},
    "circleclose": {"filter": "circleclose", "duration": 0.5, "use": "ending focus, zoom out"},
    "radial": {"filter": "radial", "duration": 0.7, "use": "magic, transformation"},

    # Dramatic transitions (tension/climax)
    "pixelize": {"filter": "pixelize", "duration": 0.4, "use": "glitch, interference, broken"},
    "diagtl": {"filter": "diagtl", "duration": 0.5, "use": "diagonal energy, power"},
}

# Emotion-pair to transition mapping
# Key format: "from_emotion->to_emotion"
EMOTION_TRANSITION_MAP = {
    # Adventure flow
    "neutral->excited": "circleopen",
    "excited->determined": "wipeleft",
    "determined->happy": "dissolve",
    "curious->surprised": "circleopen",
    "curious->excited": "wipeleft",

    # Tension building
    "neutral->scared": "fadeblack",
    "happy->scared": "fadeblack",
    "scared->determined": "radial",
    "determined->triumphant": "fadewhite",

    # Emotional beats
    "happy->sad": "fadeblack",
    "sad->determined": "dissolve",
    "sad->happy": "fadewhite",
    "surprised->excited": "wipeleft",

    # Resolution
    "excited->warm": "dissolve",
    "determined->happy": "fadewhite",
    "happy->neutral": "dissolve",
}


def _select_transition(
    prev_scene: dict,
    next_scene: dict,
    scene_index: int,
    total_scenes: int,
) -> dict:
    """Select the best transition between two scenes based on emotional flow.
    
    Returns: {"type": str, "filter": str, "duration": float}
    """
    prev_emotion = prev_scene.get("emotion", "neutral")
    next_emotion = next_scene.get("emotion", "neutral")

    # Check for specific emotion pair mapping
    pair_key = f"{prev_emotion}->{next_emotion}"
    if pair_key in EMOTION_TRANSITION_MAP:
        trans_name = EMOTION_TRANSITION_MAP[pair_key]
        trans = TRANSITION_LIBRARY[trans_name]
        return {"type": trans_name, "filter": trans["filter"], "duration": trans["duration"]}

    # Position-based defaults
    if scene_index == 0:
        return {"type": "fadeblack", "filter": "fadeblack", "duration": 1.2}  # Episode start
    if scene_index >= total_scenes - 2:
        return {"type": "fadeblack", "filter": "fadeblack", "duration": 1.5}  # Episode ending

    # Same emotion = soft dissolve
    if prev_emotion == next_emotion:
        return {"type": "dissolve", "filter": "fade", "duration": 0.6}

    # Different location = wipe
    if prev_scene.get("location", "") != next_scene.get("location", ""):
        return {"type": "wipeleft", "filter": "wipeleft", "duration": 0.6}

    # Default: dissolve
    return {"type": "dissolve", "filter": "fade", "duration": 0.8}


@app.get("/api/transitions")
def get_transitions():
    """Return available transitions and emotion-pair mappings."""
    return {
        "library": TRANSITION_LIBRARY,
        "emotionMap": EMOTION_TRANSITION_MAP,
        "tip": "Transitions are auto-selected based on scene emotion flow. Override per-scene if needed.",
    }


@app.post("/api/plan-transitions")
def plan_transitions(data: dict = {}):
    """Plan transitions for all scenes based on emotional flow.
    
    Input: {"scenes": [...]} — list of scenes with emotion/location fields.
    Returns ordered transition plan for assembly.
    """
    scenes = data.get("scenes", [])
    if len(scenes) < 2:
        return {"success": True, "transitions": [], "note": "Need at least 2 scenes"}

    transitions = []
    for i in range(len(scenes) - 1):
        trans = _select_transition(scenes[i], scenes[i + 1], i, len(scenes))
        transitions.append({
            "afterScene": scenes[i].get("sceneNumber", i + 1),
            "beforeScene": scenes[i + 1].get("sceneNumber", i + 2),
            **trans,
        })

    return {
        "success": True,
        "transitions": transitions,
        "totalTransitions": len(transitions),
    }


# =============================================================================
# CINEMATIC MOTION SYSTEM (Task 13)
# Enhanced Ken Burns with parallax, particles, and emotion-driven movement.
# Each scene type gets a specific motion style for cinematic feel.
# =============================================================================

# Motion presets per scene type (camera angle + emotion = motion pattern)
MOTION_PRESETS = {
    # Close-up emotional moments: very subtle, intimate
    "close_up:happy": {"zoom": (1.0, 1.08), "pattern": "zoom_in", "speed": "slow"},
    "close_up:sad": {"zoom": (1.05, 1.0), "pattern": "zoom_out", "speed": "very_slow"},
    "close_up:scared": {"zoom": (1.0, 1.05), "pattern": "zoom_in", "speed": "fast", "shake": True},
    "close_up:surprised": {"zoom": (1.0, 1.12), "pattern": "zoom_in", "speed": "fast"},
    "close_up:determined": {"zoom": (1.0, 1.06), "pattern": "zoom_in", "speed": "medium"},

    # Wide shots: more movement, establishing
    "wide_shot:happy": {"zoom": (1.0, 1.15), "pattern": "zoom_in_pan", "speed": "medium"},
    "wide_shot:excited": {"zoom": (1.0, 1.2), "pattern": "pan_right", "speed": "medium"},
    "wide_shot:curious": {"zoom": (1.1, 1.0), "pattern": "zoom_out", "speed": "slow"},
    "wide_shot:scared": {"zoom": (1.05, 1.15), "pattern": "pan_left", "speed": "fast"},
    "wide_shot:neutral": {"zoom": (1.0, 1.1), "pattern": "pan_right", "speed": "slow"},

    # Medium shots: moderate movement
    "medium_shot:happy": {"zoom": (1.0, 1.1), "pattern": "zoom_in", "speed": "medium"},
    "medium_shot:determined": {"zoom": (1.0, 1.12), "pattern": "zoom_in_pan", "speed": "medium"},
    "medium_shot:sad": {"zoom": (1.08, 1.0), "pattern": "zoom_out", "speed": "very_slow"},
    "medium_shot:excited": {"zoom": (1.0, 1.15), "pattern": "zoom_in_pan", "speed": "fast"},
    "medium_shot:neutral": {"zoom": (1.0, 1.08), "pattern": "zoom_in", "speed": "slow"},

    # Low angle: dramatic upward energy
    "low_angle:determined": {"zoom": (1.0, 1.15), "pattern": "zoom_in", "speed": "medium"},
    "low_angle:angry": {"zoom": (1.0, 1.18), "pattern": "zoom_in", "speed": "fast", "shake": True},
    "low_angle:excited": {"zoom": (1.0, 1.2), "pattern": "zoom_in_pan", "speed": "fast"},
}

# Speed to seconds-per-frame multiplier (affects how fast the motion plays)
MOTION_SPEEDS = {
    "very_slow": 1.5,
    "slow": 1.0,
    "medium": 0.75,
    "fast": 0.5,
}


def _get_motion_preset(camera_angle: str, emotion: str) -> dict:
    """Get the cinematic motion preset for a scene's camera + emotion combo."""
    key = f"{camera_angle}:{emotion}"
    if key in MOTION_PRESETS:
        return MOTION_PRESETS[key]

    # Try just camera angle with default emotion
    for preset_key, preset in MOTION_PRESETS.items():
        if preset_key.startswith(f"{camera_angle}:"):
            return preset

    # Ultimate fallback
    return {"zoom": (1.0, 1.1), "pattern": "zoom_in", "speed": "slow"}


def _ken_burns_with_shake(
    image_path: str, output_path: str,
    pattern: str, duration: float,
    zoom_range: tuple, width: int, height: int, fps: int,
    shake_intensity: float = 2.0,
) -> bool:
    """Enhanced Ken Burns with camera shake for tense/scared scenes.
    Uses FFmpeg crop jitter to simulate handheld camera effect.
    """
    # First generate normal Ken Burns
    temp_path = output_path + ".noshake.mp4"
    success = _ken_burns_fallback(
        image_path, temp_path, pattern, duration,
        zoom_range, width, height, fps,
    )
    if not success:
        return False

    # Apply subtle camera shake via random crop offset
    try:
        shake_expr = f"crop=w={width-4}:h={height-4}:x='2+random(1)*{shake_intensity}':y='2+random(1)*{shake_intensity}',scale={width}:{height}"
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", temp_path,
             "-vf", shake_expr,
             "-c:v", "libx264", "-preset", "fast", "-crf", "18",
             "-pix_fmt", "yuv420p", output_path],
            capture_output=True, text=True, timeout=120,
        )
        os.remove(temp_path)
        return r.returncode == 0
    except Exception:
        # Fallback: just use the non-shaken version
        if os.path.exists(temp_path):
            os.replace(temp_path, output_path)
            return True
        return False


@app.post("/api/plan-motion")
def plan_motion(data: dict = {}):
    """Plan cinematic motion for all scenes based on camera angles and emotions.
    
    SECTION 7: Enhanced motion planning with parallax and cinematic drift.
    Input: {"scenes": [...]} — scenes with cameraAngle and emotion fields.
    Returns motion plan that the assembly pipeline uses for Ken Burns/parallax/drift.
    """
    scenes = data.get("scenes", [])
    motion_plan = []

    for i, scene in enumerate(scenes):
        camera = scene.get("cameraAngle", "medium_shot")
        emotion = scene.get("emotion", "neutral")
        duration = scene.get("duration", 8.0)

        preset = _get_motion_preset(camera, emotion)

        # SECTION 7: Assign enhanced motion type based on scene context
        motion_type = "ken_burns"  # default
        parallax_direction = "right"
        drift_type = "gentle"
        
        # Wide/establishing shots → parallax (creates depth)
        if camera in ("wide_shot", "birds_eye") and emotion in ("curious", "excited", "neutral"):
            motion_type = "parallax"
            parallax_direction = random.choice(["right", "left"])
        
        # Close-up emotional moments → cinematic drift (intimate)
        elif camera == "close_up" and emotion in ("sad", "happy", "determined"):
            motion_type = "cinematic_drift"
            drift_type = "gentle" if emotion == "sad" else "breathing"
        
        # Dramatic low-angle → dramatic drift
        elif camera == "low_angle":
            motion_type = "cinematic_drift"
            drift_type = "dramatic"

        motion_plan.append({
            "sceneNumber": scene.get("sceneNumber", i + 1),
            "cameraAngle": camera,
            "emotion": emotion,
            "motionType": motion_type,
            "motionPattern": preset["pattern"],
            "zoomRange": list(preset["zoom"]),
            "speed": preset["speed"],
            "hasShake": preset.get("shake", False),
            "duration": duration,
            "parallaxDirection": parallax_direction,
            "driftType": drift_type,
        })

    return {
        "success": True,
        "motionPlan": motion_plan,
        "totalScenes": len(motion_plan),
        "motionTypes": {"ken_burns": 0, "parallax": 0, "cinematic_drift": 0},
        "presets": MOTION_PRESETS,
    }


# =============================================================================
# NODE 6 — Video Assembly (FFmpeg long-form with audio-video sync + subtitles)
# =============================================================================
# Assembles the final 5-10 minute video from per-scene clips + audio + SRT files.
#
# Key design for audio-video sync:
#   1. Each scene video clip is stretched/looped to match the ACTUAL audio duration
#      (from Node 5 ffprobe measurement, NOT estimated duration)
#   2. Per-scene SRT subtitles are burned in with time offsets matching scene position
#   3. Fade transitions on each scene clip for smooth storytelling flow
#   4. Final audio is concatenated per-scene audio (already properly paced at +0%)
#
# Assembly pipeline:
#   Step 1: Prepare each scene clip (stretch/loop to audio duration)
#   Step 2: Concatenate all scene clips into one video (no audio)
#   Step 3: Concatenate all scene audio into one audio track
#   Step 4: Burn SRT subtitles onto the combined video
#   Step 5: Merge video + audio → final output
# =============================================================================

class SubtitleStyle(BaseModel):
    fontFamily: str = "LiberationSans-Bold"
    fontSize: int = 48
    primaryColor: str = "&H00FFFFFF"    # ASS color: white
    outlineColor: str = "&H00000000"    # ASS color: black
    outlineWidth: int = 3
    shadow: int = 1
    marginV: int = 80                   # Distance from bottom
    alignment: int = 2                  # Bottom center


class AssembleVideoRequest(BaseModel):
    """Input from Collect Audio Results node."""
    # Per-scene data
    scenes: list[dict] = []
    syncMap: dict = {}
    # Full video subtitle
    fullVideoSubtitle: dict = {}
    # Metadata
    metadata: dict = {}
    # Settings
    settings: dict = {}
    # Directories
    directories: dict = {}
    # Assembly options
    outputFilename: str = "final_video.mp4"
    videoDir: str = "/data/video"
    width: int = 1920
    height: int = 1080
    fps: int = 30
    # Transitions
    fadeInDuration: float = 0.3
    fadeOutDuration: float = 0.3
    interSceneGap: float = 0.5
    # Subtitle styling
    subtitleStyle: SubtitleStyle = SubtitleStyle()
    # Keep subtitles as files by default; only burn them when explicitly requested.
    burnSubtitles: bool = False
    # Visual enhancements
    colorEnhance: bool = True
    # Async mode: returns immediately with jobId for polling progress
    asyncMode: bool = False
    # Hard stop for the full assembly worker so a stuck ffmpeg step cannot run forever.
    maxAssemblySeconds: int = 1800
    # Encoding speed: ultrafast|superfast|veryfast|faster|fast|medium (default: fast)
    encodingPreset: str = "fast"


def _ffmpeg_escape(text: str) -> str:
    """Escape special characters for FFmpeg drawtext filter."""
    if not text:
        return ""
    text = text.replace("\\", "\\\\\\\\")
    text = text.replace("'", "\u2019")
    text = text.replace(":", "\\:")
    text = text.replace("%", "%%")
    text = text.replace("[", "\\[")
    text = text.replace("]", "\\]")
    text = text.replace(";", "\\;")
    return text


def _get_media_duration(path: str) -> float:
    """Get duration of video or audio file via ffprobe."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=10,
        )
        return round(float(r.stdout.strip()), 3)
    except Exception:
        return 0.0


def _prepare_scene_clip(
    scene: dict,
    sync_data: dict,
    width: int, height: int, fps: int,
    fade_in: float, fade_out: float,
    color_enhance: bool,
    video_dir: str,
) -> dict:
    """
    Prepare one scene's video clip: stretch/loop to match audio duration.
    
    Strategy for 20-35s scenes on 6GB VRAM:
    1. If AnimateDiff clip exists: loop with forward/reverse + minterpolate
    2. If second keyframe exists: crossfade between two keyframes with Ken Burns
    3. Single image fallback: Ken Burns zoom/pan for full duration
    
    Returns {"success": bool, "clipPath": str, "duration": float, ...}
    """
    sn = scene.get("sceneNumber", 0)
    target_dur = sync_data.get("audioDuration", 0)
    if target_dur <= 0:
        target_dur = scene.get("duration", 8.0)

    # Find the video clip or images
    clip_result = scene.get("clipResult", {})
    clip_path = clip_result.get("path", "")
    lipsync_path = clip_result.get("lipSyncPath", "")
    second_kf_path = clip_result.get("secondKeyframePath", "")
    image_result = scene.get("imageResult", {})
    image_path = image_result.get("path", "")

    out_clip = os.path.join(video_dir, f"scene_{sn:02d}_prepared.mp4")

    # ── Priority: Use lip-synced clip if SadTalker produced one ──
    # Lip sync clip covers dialogue portion; combine with animated/KB for full duration
    if lipsync_path and os.path.exists(lipsync_path):
        lipsync_dur = _get_media_duration(lipsync_path)
        if lipsync_dur > 0:
            if lipsync_dur >= target_dur * 0.8:
                # Lip sync covers most of the scene — use it directly
                    vf = (f"scale={width}:{height}:flags=lanczos:force_original_aspect_ratio=increase,"
                        f"crop={width}:{height},"
                      f"fps={fps},"
                      f"fade=t=in:st=0:d={fade_in},"
                      f"fade=t=out:st={max(0, target_dur - fade_out)}:d={fade_out}")
                try:
                    r = subprocess.run(
                        ["ffmpeg", "-y", "-i", lipsync_path,
                         "-vf", vf, "-t", str(target_dur),
                         "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                         "-pix_fmt", "yuv420p", "-r", str(fps), "-an", out_clip],
                        capture_output=True, text=True, timeout=180,
                    )
                    if r.returncode == 0:
                        return {"success": True, "clipPath": out_clip,
                                "duration": target_dur, "method": "lipsync"}
                except Exception:
                    pass
            else:
                # Lip sync is shorter than scene — crossfade with Ken Burns on second keyframe
                remaining = target_dur - lipsync_dur + 1.5  # 1.5s overlap
                kb_source = second_kf_path if second_kf_path and os.path.exists(second_kf_path) else image_path
                if kb_source and os.path.exists(kb_source):
                    part_a = os.path.join(video_dir, f"scene_{sn:02d}_ls_scaled.mp4")
                    part_b = os.path.join(video_dir, f"scene_{sn:02d}_ls_kb.mp4")
                        vf_a = (f"scale={width}:{height}:flags=lanczos:force_original_aspect_ratio=increase,"
                            f"crop={width}:{height},fps={fps}")
                    total_frames_b = int((remaining + 1.5) * fps)
                    scale_w, scale_h = width * 2, height * 2
                        vf_b = (f"scale={scale_w}:{scale_h}:flags=lanczos:force_original_aspect_ratio=increase,"
                            f"crop={scale_w}:{scale_h},"
                            f"zoompan=z='min(1.0+(0.2)*on/{total_frames_b},1.2)':"
                            f"d={total_frames_b}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                            f"s={width}x{height}:fps={fps}")
                    try:
                        subprocess.run(
                            ["ffmpeg", "-y", "-i", lipsync_path,
                             "-vf", vf_a, "-t", str(lipsync_dur + 1.5),
                             "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                             "-pix_fmt", "yuv420p", "-r", str(fps), "-an", part_a],
                            capture_output=True, text=True, timeout=120,
                        )
                        subprocess.run(
                            ["ffmpeg", "-y", "-loop", "1", "-i", kb_source,
                             "-vf", vf_b, "-t", str(remaining + 1.5),
                             "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                             "-pix_fmt", "yuv420p", "-r", str(fps), "-an", part_b],
                            capture_output=True, text=True, timeout=120,
                        )
                        if os.path.exists(part_a) and os.path.exists(part_b):
                            xfade_offset = max(0, lipsync_dur - 1.5)
                            vf_xf = (f"xfade=transition=fade:duration=1.5:offset={xfade_offset},"
                                     f"fade=t=in:st=0:d={fade_in},"
                                     f"fade=t=out:st={max(0, target_dur - fade_out)}:d={fade_out}")
                            r = subprocess.run(
                                ["ffmpeg", "-y", "-i", part_a, "-i", part_b,
                                 "-filter_complex", vf_xf, "-t", str(target_dur),
                                 "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                                 "-pix_fmt", "yuv420p", "-r", str(fps), "-an", out_clip],
                                capture_output=True, text=True, timeout=180,
                            )
                            for p in [part_a, part_b]:
                                try: os.remove(p)
                                except: pass
                            if r.returncode == 0:
                                return {"success": True, "clipPath": out_clip,
                                        "duration": target_dur, "method": "lipsync_crossfade"}
                    except Exception:
                        for p in [part_a, part_b]:
                            try: os.remove(p)
                            except: pass

    if clip_path and os.path.exists(clip_path):
        # ── Strategy A: AnimateDiff clip + second keyframe crossfade ──
        # Split duration: first half = animated clip (looped), second half = Ken Burns on keyframe B
        # Crossfade between them for seamless transition
        clip_dur = _get_media_duration(clip_path)
        if clip_dur <= 0:
            clip_dur = 2.0

        if second_kf_path and os.path.exists(second_kf_path) and target_dur > 8:
            # Split: 60% animated clip (looped), 40% second keyframe (Ken Burns)
            split_a = target_dur * 0.6
            split_b = target_dur * 0.4
            crossfade_dur = 1.5  # 1.5s crossfade overlap

            # Part A: loop the AnimateDiff clip
            part_a = os.path.join(video_dir, f"scene_{sn:02d}_partA.mp4")
            loop_count = int(split_a / clip_dur) + 2
                vf_a = (f"scale={width}:{height}:flags=lanczos:force_original_aspect_ratio=increase,"
                    f"crop={width}:{height},"
                    f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir")
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-stream_loop", str(loop_count), "-i", clip_path,
                     "-vf", vf_a, "-t", str(split_a + crossfade_dur),
                     "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                     "-pix_fmt", "yuv420p", "-r", str(fps), "-an", part_a],
                    capture_output=True, text=True, timeout=300,
                )
            except Exception:
                part_a = ""

            # Part B: Ken Burns on second keyframe
            part_b = os.path.join(video_dir, f"scene_{sn:02d}_partB.mp4")
            total_frames_b = int((split_b + crossfade_dur) * fps)
            scale_w, scale_h = width * 2, height * 2
                vf_b = (f"scale={scale_w}:{scale_h}:flags=lanczos:force_original_aspect_ratio=increase,"
                    f"crop={scale_w}:{scale_h},"
                    f"zoompan=z='min(1.0+(0.2)*on/{total_frames_b},1.2)':"
                    f"d={total_frames_b}:x='(iw-iw/zoom)*on/{total_frames_b}':"
                    f"y='ih/2-(ih/zoom/2)':s={width}x{height}:fps={fps}")
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-loop", "1", "-i", second_kf_path,
                     "-vf", vf_b, "-t", str(split_b + crossfade_dur),
                     "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                     "-pix_fmt", "yuv420p", "-r", str(fps), "-an", part_b],
                    capture_output=True, text=True, timeout=300,
                )
            except Exception:
                part_b = ""

            # Crossfade A → B
            if part_a and os.path.exists(part_a) and part_b and os.path.exists(part_b):
                try:
                    xfade_offset = split_a - crossfade_dur
                    vf_xfade = f"xfade=transition=fade:duration={crossfade_dur}:offset={xfade_offset}"
                    if color_enhance:
                        vf_xfade += ",eq=saturation=1.1:contrast=1.03"
                    vf_xfade += f",fade=t=in:st=0:d={fade_in},fade=t=out:st={max(0, target_dur - fade_out)}:d={fade_out}"
                    r = subprocess.run(
                        ["ffmpeg", "-y", "-i", part_a, "-i", part_b,
                         "-filter_complex", vf_xfade, "-t", str(target_dur),
                         "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                         "-pix_fmt", "yuv420p", "-r", str(fps), "-an", out_clip],
                        capture_output=True, text=True, timeout=300,
                    )
                    if r.returncode == 0:
                        # Cleanup temp parts
                        for p in [part_a, part_b]:
                            try: os.remove(p)
                            except: pass
                        return {"success": True, "clipPath": out_clip,
                                "duration": target_dur, "method": "animated_crossfade"}
                except Exception:
                    pass
                # Cleanup on failure
                for p in [part_a, part_b]:
                    try: os.remove(p)
                    except: pass

        # ── Strategy B: Just loop the animated clip (no second keyframe) ──
        vf_parts = [
            f"scale={width}:{height}:flags=lanczos:force_original_aspect_ratio=increase",
            f"crop={width}:{height}",
        ]

        speed_ratio = clip_dur / target_dur
        if speed_ratio < 0.3:
            loop_count = int(target_dur / clip_dur) + 2
            loop_args = ["-stream_loop", str(loop_count)]
            vf_parts.append(f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir")
        elif speed_ratio < 0.8:
            setpts_factor = round(1.0 / speed_ratio, 3)
            loop_args = []
            vf_parts.append(f"setpts={setpts_factor}*PTS")
            vf_parts.append(f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir")
        else:
            loop_args = []
            vf_parts.append(f"fps={fps}")

        if color_enhance:
            vf_parts.append("eq=saturation=1.1:contrast=1.03")

        vf_parts.append(f"fade=t=in:st=0:d={fade_in}")
        vf_parts.append(f"fade=t=out:st={max(0, target_dur - fade_out)}:d={fade_out}")

        vf = ",".join(vf_parts)

        try:
            cmd = (["ffmpeg", "-y"] + loop_args + ["-i", clip_path,
                    "-vf", vf, "-t", str(target_dur),
                    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-r", str(fps), "-an", out_clip])
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if r.returncode == 0:
                return {"success": True, "clipPath": out_clip,
                        "duration": target_dur, "method": "animated"}
            # Fallback without minterpolate if it fails
            vf_simple = ",".join([
                f"scale={width}:{height}:flags=lanczos:force_original_aspect_ratio=increase",
                f"crop={width}:{height}",
                f"fps={fps}",
                f"fade=t=in:st=0:d={fade_in}",
                f"fade=t=out:st={max(0, target_dur - fade_out)}:d={fade_out}",
            ])
            loop_n = int(target_dur / clip_dur) + 2
            cmd2 = ["ffmpeg", "-y", "-stream_loop", str(loop_n), "-i", clip_path,
                    "-vf", vf_simple, "-t", str(target_dur),
                    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-r", str(fps), "-an", out_clip]
            r2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=300)
            if r2.returncode == 0:
                return {"success": True, "clipPath": out_clip,
                        "duration": target_dur, "method": "animated"}
        except Exception:
            pass

    # ── Strategy C: Two keyframes with crossfade (no animation clip) ──
    if image_path and os.path.exists(image_path) and second_kf_path and os.path.exists(second_kf_path) and target_dur > 6:
        # Split duration between two Ken Burns shots and crossfade
        half_dur = target_dur / 2
        crossfade_dur = 1.5
        total_frames_half = int((half_dur + crossfade_dur) * fps)
        scale_w, scale_h = width * 2, height * 2

        part_a = os.path.join(video_dir, f"scene_{sn:02d}_kbA.mp4")
        part_b = os.path.join(video_dir, f"scene_{sn:02d}_kbB.mp4")

        # Part A: zoom in on first keyframe
        vf_a = (f"scale={scale_w}:{scale_h}:flags=lanczos:force_original_aspect_ratio=increase,"
            f"crop={scale_w}:{scale_h},"
                f"zoompan=z='min(1.0+(0.2)*on/{total_frames_half},1.2)':"
                f"d={total_frames_half}:x='iw/2-(iw/zoom/2)':"
                f"y='ih/2-(ih/zoom/2)':s={width}x{height}:fps={fps}")
        # Part B: pan on second keyframe
        vf_b = (f"scale={scale_w}:{scale_h}:flags=lanczos:force_original_aspect_ratio=increase,"
            f"crop={scale_w}:{scale_h},"
                f"zoompan=z='1.15':d={total_frames_half}:"
                f"x='(iw-iw/zoom)*on/{total_frames_half}':"
                f"y='ih/2-(ih/zoom/2)':s={width}x{height}:fps={fps}")

        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loop", "1", "-i", image_path,
                 "-vf", vf_a, "-t", str(half_dur + crossfade_dur),
                 "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                 "-pix_fmt", "yuv420p", "-r", str(fps), "-an", part_a],
                capture_output=True, text=True, timeout=300,
            )
            subprocess.run(
                ["ffmpeg", "-y", "-loop", "1", "-i", second_kf_path,
                 "-vf", vf_b, "-t", str(half_dur + crossfade_dur),
                 "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                 "-pix_fmt", "yuv420p", "-r", str(fps), "-an", part_b],
                capture_output=True, text=True, timeout=300,
            )

            if os.path.exists(part_a) and os.path.exists(part_b):
                xfade_offset = half_dur - crossfade_dur
                vf_xfade = f"xfade=transition=fade:duration={crossfade_dur}:offset={xfade_offset}"
                if color_enhance:
                    vf_xfade += ",eq=saturation=1.1:contrast=1.03"
                vf_xfade += f",fade=t=in:st=0:d={fade_in},fade=t=out:st={max(0, target_dur - fade_out)}:d={fade_out}"
                r = subprocess.run(
                    ["ffmpeg", "-y", "-i", part_a, "-i", part_b,
                     "-filter_complex", vf_xfade, "-t", str(target_dur),
                     "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                     "-pix_fmt", "yuv420p", "-r", str(fps), "-an", out_clip],
                    capture_output=True, text=True, timeout=300,
                )
                for p in [part_a, part_b]:
                    try: os.remove(p)
                    except: pass
                if r.returncode == 0:
                    return {"success": True, "clipPath": out_clip,
                            "duration": target_dur, "method": "dual_kenburns"}
        except Exception:
            for p in [part_a, part_b]:
                try: os.remove(p)
                except: pass

    # ── Strategy D: Single image Ken Burns (original fallback) ──
    if image_path and os.path.exists(image_path):
        kb_data = scene.get("kenBurns", {})
        pattern = kb_data.get("pattern", "zoom_in")
        zoom = kb_data.get("zoomRange", [1.0, 1.25])
        z_start, z_end = (zoom[0], zoom[1]) if len(zoom) >= 2 else (1.0, 1.25)

        total_frames = int(target_dur * fps)
        # Build zoompan for this pattern
        zp_map = {
            "zoom_in": f"z='min({z_start}+({z_end}-{z_start})*on/{total_frames},{z_end})':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",
            "zoom_out": f"z='max({z_end}-({z_end}-{z_start})*on/{total_frames},{z_start})':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",
            "pan_right": f"z='{z_start}':x='(iw-iw/zoom)*on/{total_frames}':y='ih/2-(ih/zoom/2)'",
            "pan_left": f"z='{z_start}':x='(iw-iw/zoom)*(1-on/{total_frames})':y='ih/2-(ih/zoom/2)'",
            "zoom_in_pan": f"z='min({z_start}+({z_end}-{z_start})*on/{total_frames},{z_end})':x='(iw-iw/zoom)*on/{total_frames}':y='ih/2-(ih/zoom/2)'",
            "zoom_out_pan": f"z='max({z_end}-({z_end}-{z_start})*on/{total_frames},{z_start})':x='(iw-iw/zoom)*(1-on/{total_frames})':y='ih/2-(ih/zoom/2)'",
        }
        zp_expr = zp_map.get(pattern, zp_map["zoom_in"])

        # Scale up first for zoompan headroom, then zoompan to target size
        scale_w, scale_h = width * 2, height * 2
          vf = (f"scale={scale_w}:{scale_h}:flags=lanczos:force_original_aspect_ratio=increase,"
              f"crop={scale_w}:{scale_h},"
              f"zoompan={zp_expr}:d={total_frames}:s={width}x{height}:fps={fps}")

        if color_enhance:
            vf += ",eq=saturation=1.1:contrast=1.03"

        vf += f",fade=t=in:st=0:d={fade_in},fade=t=out:st={max(0, target_dur - fade_out)}:d={fade_out}"

        try:
            r = subprocess.run(
                ["ffmpeg", "-y", "-loop", "1", "-i", image_path,
                 "-vf", vf, "-t", str(target_dur),
                 "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                 "-pix_fmt", "yuv420p", "-r", str(fps), out_clip],
                capture_output=True, text=True, timeout=300,
            )
            if r.returncode == 0:
                return {"success": True, "clipPath": out_clip,
                        "duration": target_dur, "method": "ken_burns"}
        except Exception:
            pass

    # ── Last resort: colored placeholder ──
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i",
             f"color=c=#1a1a2e:s={width}x{height}:d={target_dur}:r={fps}",
             "-c:v", "libx264", "-preset", "fast", "-crf", "18",
             "-pix_fmt", "yuv420p", out_clip],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode == 0:
            return {"success": True, "clipPath": out_clip,
                    "duration": target_dur, "method": "placeholder"}
    except Exception:
        pass

    return {"success": False, "clipPath": "", "duration": 0, "method": "failed"}


def _build_ass_style(style: SubtitleStyle) -> str:
    """Build ASS subtitle style header for FFmpeg subtitles filter."""
    return (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1920\n"
        "PlayResY: 1080\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{style.fontFamily},{style.fontSize},"
        f"{style.primaryColor},&H000000FF,"
        f"{style.outlineColor},&H00000000,"
        f"-1,0,0,0,100,100,0,0,1,{style.outlineWidth},{style.shadow},"
        f"{style.alignment},20,20,{style.marginV},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )


def _srt_to_ass_events(srt_content: str) -> str:
    """Convert SRT subtitle entries to ASS Dialogue lines."""
    import re
    lines = []
    # Parse SRT blocks
    blocks = re.split(r"\n\n+", srt_content.strip())
    for block in blocks:
        parts = block.strip().split("\n")
        if len(parts) < 3:
            continue
        # Second line is timing: 00:00:01,234 --> 00:00:03,456
        timing = parts[1]
        text = " ".join(parts[2:])
        match = re.match(
            r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})",
            timing,
        )
        if not match:
            continue
        g = match.groups()
        start = f"{int(g[0])}:{g[1]}:{g[2]}.{g[3][:2]}"
        end = f"{int(g[4])}:{g[5]}:{g[6]}.{g[7][:2]}"
        # Escape special ASS characters
        text = text.replace("\\", "\\\\")
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")
    return "\n".join(lines)


@app.post("/api/assemble-video")
def assemble_video(req: AssembleVideoRequest):
    """
    Node 6: Assemble per-scene clips + audio + subtitles into final video.

    Pipeline:
      1. Prepare each scene clip (stretch/loop to match audio duration)
      2. Concatenate all scene clips → combined video (no audio)
      3. Concatenate all scene audio files → combined audio track
      4. Build ASS subtitle file from full_video.srt (word-level synced)
      5. Burn subtitles + merge audio → final output video

    Supports asyncMode=true: returns immediately with a jobId.
    Poll GET /api/job-status/{jobId} for progress and result.

    Audio-video sync:
      Each scene clip duration = audio duration from Node 5 ffprobe
      → Video and audio are always in sync per-scene
      → Subtitles come from actual TTS word timestamps, not estimates
    """
    if req.asyncMode:
        total_steps = len(req.scenes) + 4  # scenes + concat_video + concat_audio + subtitles + final_merge
        job_id = _create_job(total_steps=total_steps, description="Assembling video")
        def _run_assembly():
            try:
                _assemble_video_worker(req, job_id)
            except Exception as e:
                _fail_job(job_id, f"Unhandled error: {str(e)[:500]}")
        thread = threading.Thread(target=_run_assembly, daemon=True)
        thread.start()
        return {
            "success": True,
            "async": True,
            "jobId": job_id,
            "message": f"Assembly started in background. Poll GET /api/job-status/{job_id} for progress.",
            "totalScenes": len(req.scenes),
        }
    else:
        return _assemble_video_worker(req, job_id=None)


def _assemble_video_worker(req: AssembleVideoRequest, job_id: str = None):
    """Internal worker for video assembly. Supports job progress tracking."""
    encoding_preset = req.encodingPreset or "fast"
    os.makedirs(req.videoDir, exist_ok=True)
    w, h = req.width, req.height
    temp_files: list[str] = []
    started_at = time.time()
    max_seconds = max(60, int(req.maxAssemblySeconds or 1800))
    deadline = started_at + max_seconds

    def _remaining_seconds() -> float:
        return deadline - time.time()

    def _check_deadline(stage: str):
        if _remaining_seconds() <= 0:
            raise TimeoutError(f"Assembly timed out during {stage} after {max_seconds}s")

    def _run_ffmpeg(cmd: list[str], stage: str, timeout_cap: int):
        _check_deadline(stage)
        timeout = max(1, min(timeout_cap, int(_remaining_seconds())))
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    # ---- Step 1: Prepare scene clips (match audio duration) ----
    if job_id:
        _update_job(job_id, progress=0, current_step="Preparing scene clips...")
    prepared_clips: list[dict] = []
    for scene in req.scenes:
        _check_deadline("preparing scene clips")
        sn = scene.get("sceneNumber", len(prepared_clips) + 1)
        sync_data = req.syncMap.get(str(sn), req.syncMap.get(sn, {}))

        # If syncMap is empty, fall back to audioResult data in scene
        if not sync_data:
            ar = scene.get("audioResult", {})
            sync_data = {
                "audioDuration": ar.get("duration", scene.get("duration", 5.0)),
                "clipPath": (scene.get("clipResult") or {}).get("path", ""),
                "audioPath": ar.get("combinedAudioPath", ""),
                "subtitlePath": ar.get("subtitlePath", ""),
            }

        result = _prepare_scene_clip(
            scene, sync_data, w, h, req.fps,
            req.fadeInDuration, req.fadeOutDuration,
            req.colorEnhance, req.videoDir,
        )

        if result["success"]:
            temp_files.append(result["clipPath"])

        prepared_clips.append({
            "sceneNumber": sn,
            **result,
            "audioDuration": sync_data.get("audioDuration", 0),
            "audioPath": sync_data.get("audioPath", ""),
        })
        if job_id:
            _update_job(job_id, progress=len(prepared_clips), current_step=f"Prepared scene {sn}/{len(req.scenes)}")

    successful_clips = [c for c in prepared_clips if c.get("success")]
    if not successful_clips:
        error_msg = "No scene clips could be prepared"
        if job_id:
            _fail_job(job_id, error_msg)
        return {"success": False, "error": error_msg}

    # ---- Step 2: Concatenate video clips with crossfade transitions ----
    if job_id:
        _update_job(job_id, progress=len(req.scenes), current_step="Concatenating video clips...")
    concat_list_path = os.path.join(req.videoDir, "concat_scenes.txt")
    concat_video_path = os.path.join(req.videoDir, "concat_video.mp4")
    temp_files.extend([concat_list_path, concat_video_path])

    xfade_duration = 0.4  # crossfade between scenes

    if len(successful_clips) >= 2:
        # Use xfade filter for smooth crossfade between scenes
        inputs = []
        for clip in successful_clips:
            inputs.extend(["-i", clip["clipPath"]])

        # Build xfade filter chain: [0:v][1:v]xfade=...[v01]; [v01][2:v]xfade=...[v012]; etc.
        filter_parts = []
        prev_label = "0:v"
        for i in range(1, len(successful_clips)):
            offset = sum(
                c.get("duration", 5.0) for c in successful_clips[:i]
            ) - xfade_duration * i
            offset = max(0.1, offset)
            out_label = f"v{i}" if i < len(successful_clips) - 1 else "vout"
            filter_parts.append(
                f"[{prev_label}][{i}:v]xfade=transition=fade:duration={xfade_duration}:offset={offset:.2f}[{out_label}]"
            )
            prev_label = out_label

        filter_str = ";".join(filter_parts)

        cmd = ["ffmpeg", "-y"] + inputs + [
            "-filter_complex", filter_str,
            "-map", "[vout]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-r", str(req.fps), concat_video_path,
        ]
        r = _run_ffmpeg(cmd, "concatenating video clips", 600)

        if r.returncode != 0:
            # Fallback to simple concat if xfade fails
            print(f"[Assembly] xfade failed, falling back to simple concat: {r.stderr[:200]}")
            with open(concat_list_path, "w", encoding="utf-8") as f:
                for clip in successful_clips:
                    f.write(f"file '{clip['clipPath']}'\n")
            r = _run_ffmpeg(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                 "-i", concat_list_path, "-c", "copy", concat_video_path],
                "falling back to simple video concat", 300,
            )
            if r.returncode != 0:
                return {"success": False, "error": f"Video concat failed: {r.stderr[:400]}"}
    else:
        # Single clip — just copy
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for clip in successful_clips:
                f.write(f"file '{clip['clipPath']}'\n")
        r = _run_ffmpeg(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", concat_list_path, "-c", "copy", concat_video_path],
            "concatenating a single video clip", 300,
        )
        if r.returncode != 0:
            return {"success": False, "error": f"Video concat failed: {r.stderr[:400]}"}

    # ---- Step 3: Concatenate audio tracks ----
    if job_id:
        _update_job(job_id, progress=len(req.scenes) + 1, current_step="Concatenating audio tracks...")
    audio_paths = [c["audioPath"] for c in successful_clips
                   if c.get("audioPath") and os.path.exists(c.get("audioPath", ""))]

    concat_audio_path = os.path.join(req.videoDir, "concat_audio.m4a")
    temp_files.append(concat_audio_path)

    if len(audio_paths) > 1:
        # Use FFmpeg concat with inter-scene silence gaps
        audio_list_path = os.path.join(req.videoDir, "concat_audio.txt")
        temp_files.append(audio_list_path)

        # Generate a silence gap file
        gap_path = os.path.join(req.videoDir, "silence_gap.m4a")
        temp_files.append(gap_path)
        _run_ffmpeg(
            ["ffmpeg", "-y", "-f", "lavfi", "-i",
             f"anullsrc=r=44100:cl=stereo:d={req.interSceneGap}",
             "-c:a", "aac", "-b:a", "128k", gap_path],
            "creating inter-scene silence", 30,
        )

        with open(audio_list_path, "w", encoding="utf-8") as f:
            for i, ap in enumerate(audio_paths):
                f.write(f"file '{ap}'\n")
                if i < len(audio_paths) - 1 and os.path.exists(gap_path):
                    f.write(f"file '{gap_path}'\n")

        r = _run_ffmpeg(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", audio_list_path, "-c:a", "aac", "-b:a", "192k",
             concat_audio_path],
            "concatenating audio tracks", 300,
        )
        if r.returncode != 0:
            return {"success": False, "error": f"Audio concat failed: {r.stderr[:400]}"}
    elif len(audio_paths) == 1:
        import shutil
        shutil.copy2(audio_paths[0], concat_audio_path)
    else:
        error_msg = "No audio files available for assembly"
        if job_id:
            _fail_job(job_id, error_msg)
        return {"success": False, "error": error_msg}

    # ---- Step 4: Build ASS subtitle file (if SRT available) ----
    if job_id:
        _update_job(job_id, progress=len(req.scenes) + 2, current_step="Building subtitles...")
    ass_path = None
    if req.burnSubtitles:
        srt_info = req.fullVideoSubtitle
        srt_path = srt_info.get("path", "")

        if srt_path and os.path.exists(srt_path):
            try:
                with open(srt_path, "r", encoding="utf-8") as f:
                    srt_content = f.read()

                if srt_content.strip():
                    # Build ASS file with proper styling
                    ass_header = _build_ass_style(req.subtitleStyle)
                    ass_events = _srt_to_ass_events(srt_content)
                    ass_content = ass_header + ass_events

                    ass_path = os.path.join(req.videoDir, "subtitles.ass")
                    temp_files.append(ass_path)
                    with open(ass_path, "w", encoding="utf-8") as f:
                        f.write(ass_content)
            except Exception:
                ass_path = None

    # ---- Step 5: Final merge (video + audio + subtitles) ----
    if job_id:
        _update_job(job_id, progress=len(req.scenes) + 3, current_step="Final merge (encoding video)...")
    output_path = os.path.join(req.videoDir, req.outputFilename)

    # Build subtitle filter if ASS exists
    sub_filter = ""
    if ass_path and os.path.exists(ass_path):
        # Escape path for FFmpeg filter (forward slashes, escape colons)
        esc_ass = ass_path.replace("\\", "/").replace(":", "\\:")
        sub_filter = f"ass='{esc_ass}'"

    if sub_filter:
        cmd = [
            "ffmpeg", "-y",
            "-i", concat_video_path,
            "-i", concat_audio_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-vf", sub_filter,
            "-c:v", "libx264", "-preset", encoding_preset, "-crf", "23",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
            "-shortest", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-threads", "0",
            output_path,
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-i", concat_video_path,
            "-i", concat_audio_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", encoding_preset, "-crf", "23",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
            "-shortest", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-threads", "0",
            output_path,
        ]

    _check_deadline("final merge")
    r = _run_ffmpeg(cmd, "final merge", 3600)

    if r.returncode != 0 and sub_filter:
        # Fallback: try without subtitles
        r2 = _run_ffmpeg(
            ["ffmpeg", "-y",
             "-i", concat_video_path,
             "-i", concat_audio_path,
             "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "libx264", "-preset", encoding_preset, "-crf", "23",
             "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
             "-shortest", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart",
             "-threads", "0",
             output_path],
            "final merge without subtitles", 3600,
        )
        if r2.returncode != 0:
            error_msg = f"Final assembly failed: {r2.stderr[:400]}"
            if job_id:
                _fail_job(job_id, error_msg)
            return {"success": False, "error": error_msg}
    elif r.returncode != 0:
        error_msg = f"Final assembly failed: {r.stderr[:400]}"
        if job_id:
            _fail_job(job_id, error_msg)
        return {"success": False, "error": error_msg}

    # ---- Cleanup temp files ----
    for tf in temp_files:
        try:
            os.remove(tf)
        except Exception:
            pass

    # ---- Result ----
    elapsed = round(time.time() - started_at, 1)
    file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
    final_duration = _get_media_duration(output_path)

    result = {
        "success": True,
        "outputPath": output_path,
        "outputFilename": req.outputFilename,
        "duration": final_duration,
        "fileSize": file_size,
        "fileSizeMB": round(file_size / (1024 * 1024), 2),
        "resolution": f"{w}x{h}",
        "fps": req.fps,
        "scenesAssembled": len(successful_clips),
        "totalScenes": len(req.scenes),
        "subtitlesBurned": ass_path is not None,
        "elapsedSeconds": elapsed,
        "encodingPreset": encoding_preset,
        "clipDetails": [
            {
                "scene": c["sceneNumber"],
                "method": c.get("method", ""),
                "duration": c.get("duration", 0),
                "audioDuration": c.get("audioDuration", 0),
            }
            for c in prepared_clips
        ],
        "metadata": req.metadata,
        "assembledAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if job_id:
        _complete_job(job_id, result)
    return result


# =====================================================================
# NODE 7 — SEO & METADATA GENERATION
# Generates YouTube-optimized title, description, tags, and category
# using the same LLM cascade as Node 2 (Gemini → Groq → Ollama).
# =====================================================================


class SEORequest(BaseModel):
    """Input for SEO metadata generation."""
    # Story / episode context
    storyTitle: str = ""
    seriesName: str = "Finn, Squeaky & Misty"
    episodeNumber: int = 1
    genre: str = "fantasy_adventure"
    targetAudience: str = "kids 4-10, families"
    description: str = ""          # brief story synopsis from script
    scenes: list[dict] = []        # scene summaries (for chapter timestamps)
    characters: list[dict] = []    # character names/descriptions
    videoDuration: float = 0       # total duration in seconds

    # Channel info for branding
    channelName: str = ""
    channelTagline: str = ""

    # LLM cascade settings (same as Node 2)
    provider: str = "auto"         # auto | gemini | groq | ollama
    geminiModel: str = "gemini-2.5-flash"
    groqModel: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    ollamaModel: str = "llama3.1:8b"
    geminiApiKey: str = ""
    groqApiKey: str = ""


def _build_seo_prompt(ctx: dict) -> str:
    """Build the LLM prompt for generating YouTube SEO metadata."""

    duration_min = int(ctx.get("videoDuration", 0) // 60)
    duration_sec = int(ctx.get("videoDuration", 0) % 60)
    duration_str = f"{duration_min}:{duration_sec:02d}" if ctx.get("videoDuration") else "unknown"

    # Build scene list for chapter timestamps
    scene_lines = ""
    if ctx.get("scenes"):
        for s in ctx["scenes"]:
            sn = s.get("sceneNumber", "?")
            st = s.get("title", s.get("setting", f"Scene {sn}"))
            scene_lines += f"  Scene {sn}: {st}\n"

    char_names = ", ".join(
        c.get("name", "") for c in ctx.get("characters", []) if c.get("name")
    ) or "Captain Finn, Squeaky"

    prompt = f"""You are a YouTube SEO expert for animated children's story channels.

Generate optimized YouTube metadata for this video:

STORY DETAILS:
  Title: {ctx.get('storyTitle', 'Untitled')}
  Series: {ctx.get('seriesName', 'Finn, Squeaky & Misty')}
  Episode: #{ctx.get('episodeNumber', 1)}
  Genre: {ctx.get('genre', 'adventure')}
  Target Audience: {ctx.get('targetAudience', 'kids 4-10, families')}
  Characters: {char_names}
  Duration: {duration_str}
  Synopsis: {ctx.get('description', 'An exciting adventure episode.')}

SCENES:
{scene_lines or '  (no scene breakdown available)'}

CHANNEL:
  Name: {ctx.get('channelName', '')}
  Tagline: {ctx.get('channelTagline', '')}

Return ONLY valid JSON with these exact keys:
{{
  "title": "YouTube video title (max 100 chars, include series name, episode-relevant hook, emoji optional)",
  "description": "Full YouTube description (500-1500 chars). Include:\\n- 1-2 sentence hook\\n- Episode synopsis (3-4 sentences)\\n- Chapter timestamps if scenes available (00:00 format)\\n- Character names\\n- Call to action (subscribe, like)\\n- Relevant hashtags at end",
  "tags": ["list", "of", "15-25", "relevant", "search", "tags"],
  "category": "YouTube category (one of: Entertainment, Film & Animation, Education, People & Blogs)",
  "shortDescription": "Brief 1-2 sentence description for social sharing (max 160 chars)"
}}

RULES:
- Title must be catchy, search-friendly, and include the series name
- Tags should include: series name, character names, genre terms, "animated story", "kids cartoon", "bedtime story", age-related terms
- Description must have chapter timestamps starting at 0:00 if scenes are provided
- Keep language family-friendly and exciting
- Do NOT include any text outside the JSON object"""

    return prompt


def _build_chapter_timestamps(scenes: list[dict], sync_map: dict | None = None) -> str:
    """Build YouTube chapter timestamps from scene data."""
    if not scenes:
        return ""

    lines = []
    running_seconds = 0.0
    for s in scenes:
        sn = s.get("sceneNumber", len(lines) + 1)
        title = s.get("title", s.get("setting", f"Scene {sn}"))
        minutes = int(running_seconds // 60)
        secs = int(running_seconds % 60)
        lines.append(f"{minutes}:{secs:02d} {title}")
        # Advance by scene audio duration if available
        if sync_map and str(sn) in sync_map:
            running_seconds += float(sync_map[str(sn)])
        elif s.get("estimatedDuration"):
            running_seconds += float(s["estimatedDuration"])
        else:
            running_seconds += 30.0  # fallback estimate

    return "\n".join(lines)


def _validate_seo(seo: dict, ctx: dict) -> dict:
    """Validate and fix generated SEO metadata."""

    # Title
    title = seo.get("title", "")
    if not title or len(title) < 10:
        series = ctx.get("seriesName", "Finn, Squeaky & Misty")
        story = ctx.get("storyTitle", "New Adventure")
        title = f"{series} | {story} | Episode #{ctx.get('episodeNumber', 1)}"
    if len(title) > 100:
        title = title[:97] + "..."
    seo["title"] = title

    # Tags
    tags = seo.get("tags", [])
    if not isinstance(tags, list):
        tags = []
    # Ensure essential tags are present
    essential = [
        "animated story", "kids cartoon", "bedtime story",
        "children animation", "cartoon for kids",
    ]
    series = ctx.get("seriesName", "")
    if series:
        essential.insert(0, series.lower())
    for char in ctx.get("characters", []):
        name = char.get("name", "")
        if name:
            essential.append(name.lower())
    existing_lower = {t.lower() for t in tags}
    for tag in essential:
        if tag.lower() not in existing_lower:
            tags.append(tag)
    # Cap at 30 tags, each max 30 chars
    tags = [t[:30] for t in tags[:30]]
    seo["tags"] = tags

    # Description
    desc = seo.get("description", "")
    if not desc or len(desc) < 50:
        desc = (
            f"Watch Episode #{ctx.get('episodeNumber', 1)} of "
            f"{ctx.get('seriesName', 'Finn, Squeaky & Misty')}!\n\n"
            f"{ctx.get('description', 'Join our heroes on a brand new adventure!')}\n\n"
            f"Don't forget to like and subscribe for more animated stories!"
        )
    seo["description"] = desc

    # Category
    valid_categories = {
        "entertainment", "film & animation", "education", "people & blogs"
    }
    cat = seo.get("category", "Film & Animation")
    if cat.lower() not in valid_categories:
        cat = "Film & Animation"
    seo["category"] = cat

    # Short description
    short = seo.get("shortDescription", "")
    if not short or len(short) < 20:
        short = title[:160]
    if len(short) > 160:
        short = short[:157] + "..."
    seo["shortDescription"] = short

    return seo


@app.post("/api/generate-seo")
def generate_seo(req: SEORequest):
    """
    Node 7: Generate YouTube SEO metadata using LLM cascade.
    Produces title, description, tags, category, and chapter timestamps.
    """
    ctx = req.model_dump()
    prompt = _build_seo_prompt(ctx)

    # Resolve API keys
    gemini_key = req.geminiApiKey or GEMINI_API_KEY
    groq_key = req.groqApiKey or GROQ_API_KEY

    # Build provider order (same cascade as Node 2)
    if req.provider == "auto":
        providers = []
        if gemini_key:
            providers.append(("gemini", gemini_key))
        if groq_key:
            providers.append(("groq", groq_key))
        providers.append(("ollama", ""))
    elif req.provider == "gemini":
        providers = [("gemini", gemini_key)]
    elif req.provider == "groq":
        providers = [("groq", groq_key)]
    elif req.provider == "ollama":
        providers = [("ollama", "")]
    else:
        providers = [("ollama", "")]

    errors: list[dict] = []
    for provider_name, api_key in providers:
        if provider_name == "gemini":
            if not api_key:
                errors.append({"provider": "gemini", "error": "No API key"})
                continue
            result = _call_gemini(prompt, api_key, req.geminiModel)
        elif provider_name == "groq":
            if not api_key:
                errors.append({"provider": "groq", "error": "No API key"})
                continue
            result = _call_groq(prompt, api_key, req.groqModel)
        else:
            result = _call_ollama(prompt, req.ollamaModel)

        if not result["success"]:
            errors.append({"provider": provider_name, "error": result["error"]})
            continue

        seo = _parse_llm_json(result["text"])
        if seo is None:
            errors.append({
                "provider": provider_name,
                "error": f"Failed to parse JSON (length={len(result['text'])})",
            })
            continue

        # Validate and fix
        seo = _validate_seo(seo, ctx)

        # Build chapter timestamps separately (more reliable than LLM)
        chapters = _build_chapter_timestamps(
            req.scenes,
            ctx.get("syncMap"),
        )
        if chapters:
            seo["chapterTimestamps"] = chapters

        return {
            "success": True,
            "provider": provider_name,
            "seo": seo,
            "promptLength": len(prompt),
            "errors": errors,
            "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    # All providers failed
    raise HTTPException(
        status_code=502,
        detail={
            "message": "All LLM providers failed to generate SEO metadata",
            "errors": errors,
        },
    )


# =====================================================================
# NODE 8 — THUMBNAIL GENERATION
# Takes the base thumbnail image from Node 4 (ComfyUI) and adds
# text overlay (title, episode #, series branding) using Pillow.
# If no base image exists, generates a gradient + text fallback.
# Output: 1280x720 YouTube-ready thumbnail PNG.
# =====================================================================

THUMB_W, THUMB_H = 1280, 720


class ThumbnailRequest(BaseModel):
    """Input for thumbnail text-overlay generation."""
    # Base image from Node 4
    baseThumbnailPath: str = ""          # path to ComfyUI-generated image
    # Text to overlay
    title: str = ""                      # short punchy title (from SEO)
    episodeNumber: int = 1
    seriesName: str = "Finn, Squeaky & Misty"
    # Style
    titleFontSize: int = 72
    episodeFontSize: int = 36
    fontColor: str = "#FFFFFF"
    strokeColor: str = "#000000"
    strokeWidth: int = 4
    bannerColor: str = "#FF4444"         # red banner behind episode #
    bannerOpacity: int = 200             # 0-255
    includeTextOverlay: bool = True
    # Quality
    outputDir: str = "/data/thumbnails"
    outputFilename: str = "thumbnail_final.png"
    jpegFallback: bool = True            # also save .jpg (smaller)


def _hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 6:
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (255, 255, 255)


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Word-wrap text to fit within max_width pixels."""
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test = f"{current_line} {word}".strip()
        bbox = font.getbbox(test)
        w = bbox[2] - bbox[0]
        if w <= max_width:
            current_line = test
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines or [text]


def _draw_text_with_stroke(
    draw: ImageDraw.ImageDraw,
    position: tuple,
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple,
    stroke_fill: tuple,
    stroke_width: int,
):
    """Draw text with outline/stroke effect."""
    x, y = position
    # Draw stroke by rendering text at offsets
    for dx in range(-stroke_width, stroke_width + 1):
        for dy in range(-stroke_width, stroke_width + 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, font=font, fill=stroke_fill)
    # Draw main text
    draw.text((x, y), text, font=font, fill=fill)


def _generate_gradient_background(width: int, height: int) -> Image.Image:
    """Generate a colorful gradient background as thumbnail fallback."""
    img = Image.new("RGB", (width, height))
    for y in range(height):
        ratio = y / height
        r = int(40 + 60 * ratio)
        g = int(10 + 30 * (1 - ratio))
        b = int(80 + 100 * ratio)
        for x in range(width):
            x_ratio = x / width
            img.putpixel((x, y), (
                min(255, int(r + 40 * x_ratio)),
                min(255, int(g + 20 * x_ratio)),
                min(255, int(b - 30 * x_ratio)),
            ))
    return img


def _load_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    """Load Liberation Sans font at given size, fallback to default."""
    path = FONT_BOLD if bold else FONT_REGULAR
    try:
        return ImageFont.truetype(path, size)
    except (OSError, IOError):
        # Try common system paths
        for alt in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]:
            try:
                return ImageFont.truetype(alt, size)
            except (OSError, IOError):
                continue
    return ImageFont.load_default()


@app.post("/api/generate-thumbnail")
def generate_thumbnail(req: ThumbnailRequest):
    """
    Node 8: Generate YouTube thumbnail with text overlay.
    Uses the base image from Node 4 + adds title & episode branding.
    """
    os.makedirs(req.outputDir, exist_ok=True)
    output_path = os.path.join(req.outputDir, req.outputFilename)

    # ---- Step 1: Load or generate base image ----
    base_img = None
    if req.baseThumbnailPath and os.path.exists(req.baseThumbnailPath):
        try:
            base_img = Image.open(req.baseThumbnailPath).convert("RGB")
            base_img = base_img.resize((THUMB_W, THUMB_H), Image.LANCZOS)
        except Exception:
            base_img = None

    if base_img is None:
        base_img = _generate_gradient_background(THUMB_W, THUMB_H)
        used_fallback = True
    else:
        used_fallback = False

    base_rgba = base_img.convert("RGBA")
    draw = ImageDraw.Draw(base_rgba)
    if req.includeTextOverlay:
        # Optional overlay kept for manual use, but disabled by default in the workflow.
        overlay = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        # Gradient shadow at bottom 40%
        for y in range(int(THUMB_H * 0.6), THUMB_H):
            alpha = int(180 * ((y - THUMB_H * 0.6) / (THUMB_H * 0.4)))
            overlay_draw.line([(0, y), (THUMB_W, y)], fill=(0, 0, 0, min(alpha, 180)))
        base_rgba = Image.alpha_composite(base_rgba, overlay)

        font_color = _hex_to_rgb(req.fontColor)
        stroke_color = _hex_to_rgb(req.strokeColor)

        title_font = _load_font(req.titleFontSize, bold=True)
        title_text = req.title or f"Episode #{req.episodeNumber}"
        max_text_w = THUMB_W - 120  # 60px margin each side

        lines = _wrap_text(title_text, title_font, max_text_w)
        line_height = req.titleFontSize + 8
        total_text_h = len(lines) * line_height

        # Position title in lower third
        start_y = THUMB_H - total_text_h - 100
        for i, line in enumerate(lines):
            bbox = title_font.getbbox(line)
            text_w = bbox[2] - bbox[0]
            x = (THUMB_W - text_w) // 2
            y = start_y + i * line_height
            _draw_text_with_stroke(
                draw, (x, y), line, title_font,
                font_color, stroke_color, req.strokeWidth,
            )

        # ---- Series name (bottom right, small) ----
        if req.seriesName:
            series_font = _load_font(24, bold=False)
            s_bbox = series_font.getbbox(req.seriesName)
            s_w = s_bbox[2] - s_bbox[0]
            _draw_text_with_stroke(
                draw, (THUMB_W - s_w - 30, THUMB_H - 40),
                req.seriesName, series_font,
                (220, 220, 220), (0, 0, 0), 2,
            )

    # ---- Episode badge (always kept) ----
    episode_font = _load_font(req.episodeFontSize, bold=True)
    ep_text = f"EP. {req.episodeNumber}"
    ep_bbox = episode_font.getbbox(ep_text)
    ep_w = ep_bbox[2] - ep_bbox[0] + 30
    ep_h = ep_bbox[3] - ep_bbox[1] + 20

    banner_rgb = _hex_to_rgb(req.bannerColor)
    banner_rgba = (*banner_rgb, req.bannerOpacity)
    banner_overlay = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
    banner_draw = ImageDraw.Draw(banner_overlay)
    banner_draw.rounded_rectangle(
        [(20, 20), (20 + ep_w, 20 + ep_h)],
        radius=10,
        fill=banner_rgba,
    )
    base_rgba = Image.alpha_composite(base_rgba, banner_overlay)
    draw = ImageDraw.Draw(base_rgba)
    _draw_text_with_stroke(
        draw, (35, 25), ep_text, episode_font,
        (255, 255, 255), (0, 0, 0), 2,
    )

    # ---- Step 6: Save ----
    final_rgb = base_rgba.convert("RGB")
    final_rgb.save(output_path, "PNG", optimize=True)
    file_size = os.path.getsize(output_path)

    result = {
        "success": True,
        "path": output_path,
        "filename": req.outputFilename,
        "resolution": f"{THUMB_W}x{THUMB_H}",
        "fileSize": file_size,
        "fileSizeKB": round(file_size / 1024, 1),
        "usedFallbackBackground": used_fallback,
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Optional JPEG copy (smaller for upload)
    if req.jpegFallback:
        jpg_path = output_path.rsplit(".", 1)[0] + ".jpg"
        final_rgb.save(jpg_path, "JPEG", quality=90, optimize=True)
        result["jpegPath"] = jpg_path
        result["jpegSize"] = os.path.getsize(jpg_path)

    return result


# =====================================================================
# NODE 9 — YOUTUBE UPLOAD
# Uploads the assembled video + thumbnail + SEO metadata to YouTube
# via the YouTube Data API v3 with OAuth2.
#
# Auth flow:
#   1. First run: user visits /api/youtube-auth-url to get OAuth URL
#   2. User authorises in browser, gets redirect with code
#   3. POST /api/youtube-auth-callback with the code → saves token
#   4. Subsequent uploads use saved token (auto-refresh)
#
# Token stored in DATA_DIR/youtube_token.json (persistent via Docker volume)
# =====================================================================

import pickle  # noqa: E402
from http.client import HTTPSConnection, HTTPResponse  # noqa: E402

# YouTube API constants
YOUTUBE_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
YOUTUBE_API_SERVICE = "youtube"
YOUTUBE_API_VERSION = "v3"
CREDENTIALS_DIR = os.path.join(DATA_DIR, "credentials")
YOUTUBE_TOKEN_FILE = os.path.join(CREDENTIALS_DIR, "youtube_token.json")
YOUTUBE_CLIENT_SECRET_FILE = os.path.join(CREDENTIALS_DIR, "client_secret.json")

# Category IDs: https://developers.google.com/youtube/v3/docs/videoCategories
YOUTUBE_CATEGORIES = {
    "film & animation": "1",
    "entertainment": "24",
    "education": "27",
    "people & blogs": "22",
    "comedy": "23",
    "howto & style": "26",
}


class YouTubeUploadRequest(BaseModel):
    """Input for YouTube upload."""
    # Video
    videoPath: str                       # absolute path to MP4
    # SEO metadata
    title: str = "Untitled Episode"
    description: str = ""
    tags: list[str] = []
    category: str = "Film & Animation"   # mapped to category ID
    # Thumbnail
    thumbnailPath: str = ""              # path to thumbnail image
    # Privacy
    privacyStatus: str = "private"       # private | unlisted | public
    madeForKids: bool = True
    # Publish scheduling (ISO 8601, e.g. "2025-01-20T15:00:00Z")
    publishAt: str = ""                  # empty = no schedule (publish immediately if public)
    # Language
    defaultLanguage: str = "en"
    defaultAudioLanguage: str = "en"


# In-memory storage for OAuth flow state (web app redirect flow)
_youtube_oauth_flows = {}


def _load_youtube_credentials():
    """Load saved YouTube OAuth2 credentials from token file."""
    if not os.path.exists(YOUTUBE_TOKEN_FILE):
        return None
    try:
        with open(YOUTUBE_TOKEN_FILE, "r") as f:
            token_data = json.load(f)
        from google.oauth2.credentials import Credentials
        creds = Credentials(
            token=token_data.get("token"),
            refresh_token=token_data.get("refresh_token"),
            token_uri=token_data.get("token_uri", "https://oauth2.googleapis.com/token"),
            client_id=token_data.get("client_id"),
            client_secret=token_data.get("client_secret"),
            scopes=YOUTUBE_SCOPES,
        )
        # Refresh if expired
        if creds.expired and creds.refresh_token:
            from google.auth.transport.requests import Request
            creds.refresh(Request())
            _save_youtube_credentials(creds)
        return creds
    except Exception:
        return None


def _save_youtube_credentials(creds):
    """Save OAuth2 credentials to persistent token file."""
    token_data = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
    }
    os.makedirs(os.path.dirname(YOUTUBE_TOKEN_FILE), exist_ok=True)
    with open(YOUTUBE_TOKEN_FILE, "w") as f:
        json.dump(token_data, f, indent=2)


@app.get("/api/youtube-auth-url")
def youtube_auth_url():
    """
    Node 9 setup: Get the OAuth2 authorization URL.
    User must visit this URL in a browser to grant YouTube upload permission.
    Requires client_secret.json in /data/credentials/ (from Google Cloud Console).
    Use Web Application type OAuth client with localhost redirect URI.
    """
    if not os.path.exists(YOUTUBE_CLIENT_SECRET_FILE):
        raise HTTPException(
            status_code=400,
            detail={
                "message": "client_secret.json not found",
                "expected_path": YOUTUBE_CLIENT_SECRET_FILE,
                "instructions": (
                    "1. Go to console.cloud.google.com → APIs → Credentials\n"
                    "2. Create OAuth 2.0 Client (Web application type)\n"
                    "3. Add http://localhost:8001/api/youtube-auth-callback as Authorized redirect URI\n"
                    "4. Download client_secret.json\n"
                    "5. Place it in the data/credentials directory"
                ),
            },
        )

    from google_auth_oauthlib.flow import Flow
    redirect_uri = "http://localhost:8001/api/youtube-auth-callback"
    flow = Flow.from_client_secrets_file(
        YOUTUBE_CLIENT_SECRET_FILE,
        scopes=YOUTUBE_SCOPES,
        redirect_uri=redirect_uri,
    )
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    _youtube_oauth_flows[state] = flow
    return {
        "success": True,
        "authUrl": auth_url,
        "redirectUri": redirect_uri,
        "instructions": "Open the URL in your browser, authorize, and you will be redirected back automatically.",
    }


@app.get("/api/youtube-auth-callback")
def youtube_auth_callback(code: str, state: str = None):
    """
    Handle OAuth callback redirect from Google.
    This is called automatically when Google redirects back after authorization.
    """
    if not code:
        raise HTTPException(400, "No authorization code provided")

    if state not in _youtube_oauth_flows:
        raise HTTPException(
            400,
            "Invalid state - authorization flow not found. "
            "Please try /api/youtube-auth-url again.",
        )

    flow = _youtube_oauth_flows.pop(state)

    try:
        flow.fetch_token(code=code)
        creds = flow.credentials
        _save_youtube_credentials(creds)
        return {
            "success": True,
            "message": "YouTube authentication successful! Token saved. You can close this window.",
            "tokenPath": YOUTUBE_TOKEN_FILE,
        }
    except Exception as e:
        raise HTTPException(500, f"Token exchange failed: {str(e)}")


@app.get("/api/youtube-auth-status")
def youtube_auth_status():
    """Check if YouTube OAuth2 credentials are configured and valid."""
    creds = _load_youtube_credentials()
    has_client_secret = os.path.exists(YOUTUBE_CLIENT_SECRET_FILE)
    return {
        "authenticated": creds is not None and creds.valid,
        "hasClientSecret": has_client_secret,
        "hasToken": os.path.exists(YOUTUBE_TOKEN_FILE),
        "tokenPath": YOUTUBE_TOKEN_FILE,
        "clientSecretPath": YOUTUBE_CLIENT_SECRET_FILE,
    }


@app.post("/api/upload-youtube")
def upload_youtube(req: YouTubeUploadRequest):
    """
    Node 9: Upload video to YouTube with metadata + thumbnail.
    Requires prior OAuth2 setup via /api/youtube-auth-url + /api/youtube-auth-callback.
    """
    started_at = time.time()

    # ---- Validate inputs ----
    if not req.videoPath or not os.path.exists(req.videoPath):
        raise HTTPException(status_code=400, detail=f"Video not found: {req.videoPath}")

    # ---- Load credentials ----
    creds = _load_youtube_credentials()
    if not creds or not creds.valid:
        raise HTTPException(
            status_code=401,
            detail={
                "message": "YouTube not authenticated. Run OAuth setup first.",
                "setup_url": "/api/youtube-auth-url",
            },
        )

    # ---- Build YouTube service ----
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    youtube = build(YOUTUBE_API_SERVICE, YOUTUBE_API_VERSION, credentials=creds)

    # ---- Map category name to ID ----
    category_id = YOUTUBE_CATEGORIES.get(req.category.lower(), "1")

    # ---- Build video metadata ----
    body = {
        "snippet": {
            "title": req.title[:100],
            "description": req.description[:5000],
            "tags": req.tags[:500],
            "categoryId": category_id,
            "defaultLanguage": req.defaultLanguage,
            "defaultAudioLanguage": req.defaultAudioLanguage,
        },
        "status": {
            "privacyStatus": req.privacyStatus,
            "selfDeclaredMadeForKids": req.madeForKids,
        },
    }

    # Scheduled publishing
    if req.publishAt and req.privacyStatus == "private":
        body["status"]["publishAt"] = req.publishAt
        body["status"]["privacyStatus"] = "private"

    # ---- Upload video (resumable) ----
    media = MediaFileUpload(
        req.videoPath,
        mimetype="video/mp4",
        resumable=True,
        chunksize=10 * 1024 * 1024,  # 10MB chunks
    )

    upload_request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )

    video_id = None
    response = None
    try:
        while response is None:
            status, response = upload_request.next_chunk()
        video_id = response.get("id")
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail={
                "message": f"YouTube upload failed: {str(e)}",
                "videoPath": req.videoPath,
            },
        )

    # ---- Upload thumbnail (if available) ----
    thumbnail_uploaded = False
    if video_id and req.thumbnailPath and os.path.exists(req.thumbnailPath):
        try:
            thumb_media = MediaFileUpload(
                req.thumbnailPath,
                mimetype="image/png" if req.thumbnailPath.endswith(".png") else "image/jpeg",
            )
            youtube.thumbnails().set(
                videoId=video_id,
                media_body=thumb_media,
            ).execute()
            thumbnail_uploaded = True
        except Exception:
            # Thumbnail upload requires verified channel — non-fatal
            thumbnail_uploaded = False

    elapsed = round(time.time() - started_at, 1)
    video_url = f"https://www.youtube.com/watch?v={video_id}" if video_id else ""

    return {
        "success": True,
        "videoId": video_id,
        "videoUrl": video_url,
        "title": req.title,
        "privacyStatus": req.privacyStatus,
        "category": req.category,
        "categoryId": category_id,
        "thumbnailUploaded": thumbnail_uploaded,
        "madeForKids": req.madeForKids,
        "publishAt": req.publishAt or None,
        "elapsedSeconds": elapsed,
        "uploadedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# =============================================================================
# PLAYLIST MANAGEMENT (Task 19)
# Auto-adds uploaded videos to the series playlist.
# Creates playlist if it doesn't exist.
# =============================================================================

PLAYLIST_FILE = os.path.join(DATA_DIR, "stories", "playlists.json")


class PlaylistAddRequest(BaseModel):
    videoId: str
    playlistTitle: str = "Finn, Squeaky & Misty Adventures"
    playlistDescription: str = "Join Captain Finn, Squeaky, and Misty on magical adventures!"
    createIfMissing: bool = True


@app.post("/api/youtube-playlist-add")
def youtube_playlist_add(req: PlaylistAddRequest):
    """Add a video to the series playlist. Creates the playlist if needed."""
    creds = _load_youtube_credentials()
    if not creds or not creds.valid:
        raise HTTPException(status_code=401, detail="YouTube not authenticated")

    from googleapiclient.discovery import build
    youtube = build(YOUTUBE_API_SERVICE, YOUTUBE_API_VERSION, credentials=creds)

    # Load or find existing playlist ID
    playlist_id = _get_or_create_playlist(
        youtube, req.playlistTitle, req.playlistDescription, req.createIfMissing
    )

    if not playlist_id:
        return {"success": False, "error": "Could not find or create playlist"}

    # Add video to playlist
    try:
        youtube.playlistItems().insert(
            part="snippet",
            body={
                "snippet": {
                    "playlistId": playlist_id,
                    "resourceId": {
                        "kind": "youtube#video",
                        "videoId": req.videoId,
                    },
                },
            },
        ).execute()
    except Exception as e:
        return {"success": False, "error": f"Failed to add to playlist: {str(e)}"}

    return {
        "success": True,
        "videoId": req.videoId,
        "playlistId": playlist_id,
        "playlistTitle": req.playlistTitle,
    }


def _get_or_create_playlist(youtube, title: str, description: str, create: bool) -> str:
    """Find existing playlist by title or create a new one. Returns playlist ID."""
    # Check local cache first
    if os.path.exists(PLAYLIST_FILE):
        data = _load_json(PLAYLIST_FILE)
        cached_id = data.get("playlists", {}).get(title)
        if cached_id:
            return cached_id

    # Search channel playlists
    try:
        response = youtube.playlists().list(
            part="snippet", mine=True, maxResults=50
        ).execute()
        for item in response.get("items", []):
            if item["snippet"]["title"].lower() == title.lower():
                playlist_id = item["id"]
                _cache_playlist_id(title, playlist_id)
                return playlist_id
    except Exception:
        pass

    # Create new playlist
    if not create:
        return ""

    try:
        response = youtube.playlists().insert(
            part="snippet,status",
            body={
                "snippet": {
                    "title": title,
                    "description": description,
                },
                "status": {"privacyStatus": "public"},
            },
        ).execute()
        playlist_id = response["id"]
        _cache_playlist_id(title, playlist_id)
        return playlist_id
    except Exception:
        return ""


def _cache_playlist_id(title: str, playlist_id: str):
    """Cache playlist ID locally to avoid repeated API calls."""
    os.makedirs(os.path.dirname(PLAYLIST_FILE), exist_ok=True)
    data = {}
    if os.path.exists(PLAYLIST_FILE):
        data = _load_json(PLAYLIST_FILE)
    data.setdefault("playlists", {})[title] = playlist_id
    _save_json(PLAYLIST_FILE, data)


# =============================================================================
# SECTION 4 — QUALITY VALIDATION PIPELINE
# Pre-upload quality checks. Validates visual, audio, subtitle, and continuity.
# Returns quality score + per-category results. Blocks upload if below threshold.
# =============================================================================

class ValidateVideoRequest(BaseModel):
    """Request to validate a generated episode before upload."""
    videoPath: str = ""
    imageDir: str = "/data/images"
    audioDir: str = "/data/audio"
    clipDir: str = "/data/clips"
    subtitlePath: str = ""
    scenes: list[dict] = []
    expectedSceneCount: int = 0
    minQualityScore: float = 0.7  # 0-1, below this = fail


@app.post("/api/validate-video")
def validate_video(req: ValidateVideoRequest):
    """
    Section 4: Pre-upload quality validation pipeline.
    
    Checks:
    - Image quality (file exists, minimum size, not corrupted)
    - Audio quality (files exist, have duration)
    - Scene completeness (all expected scenes generated)
    - Video integrity (file exists, has duration, no black frames)
    - Subtitle timing (file exists, reasonable timing)
    
    Returns quality score (0-1) and per-category results.
    """
    categories = {}
    
    # --- 1. Image Validation ---
    image_score = 1.0
    image_issues = []
    scene_count = req.expectedSceneCount or len(req.scenes)
    
    for i in range(1, scene_count + 1):
        img_path = os.path.join(req.imageDir, f"scene_{i:02d}.png")
        if not os.path.exists(img_path):
            image_issues.append(f"Missing scene image: scene_{i:02d}.png")
            image_score -= (1.0 / max(scene_count, 1))
        else:
            size = os.path.getsize(img_path)
            if size < 10000:  # Less than 10KB = likely corrupted
                image_issues.append(f"scene_{i:02d}.png too small ({size} bytes) — likely corrupted")
                image_score -= (0.5 / max(scene_count, 1))
            # Check if it's a valid image
            try:
                img = Image.open(img_path)
                img.verify()
            except Exception:
                image_issues.append(f"scene_{i:02d}.png is not a valid image file")
                image_score -= (1.0 / max(scene_count, 1))

    categories["images"] = {
        "score": max(0.0, image_score),
        "issues": image_issues,
        "checked": scene_count,
    }

    # --- 2. Audio Validation ---
    audio_score = 1.0
    audio_issues = []
    audio_files_found = 0
    
    for i in range(1, scene_count + 1):
        narration_path = os.path.join(req.audioDir, f"scene_{i:02d}_narration.m4a")
        if os.path.exists(narration_path):
            audio_files_found += 1
            size = os.path.getsize(narration_path)
            if size < 1000:
                audio_issues.append(f"scene_{i:02d}_narration.m4a too small ({size} bytes)")
                audio_score -= (0.3 / max(scene_count, 1))
        else:
            audio_issues.append(f"Missing narration audio: scene_{i:02d}_narration.m4a")
            audio_score -= (0.8 / max(scene_count, 1))

    if audio_files_found == 0:
        audio_score = 0.0
        audio_issues.append("No audio files found at all")

    categories["audio"] = {
        "score": max(0.0, audio_score),
        "issues": audio_issues,
        "filesFound": audio_files_found,
    }

    # --- 3. Video Integrity ---
    video_score = 1.0
    video_issues = []
    
    if req.videoPath:
        if not os.path.exists(req.videoPath):
            video_score = 0.0
            video_issues.append("Video file does not exist")
        else:
            size = os.path.getsize(req.videoPath)
            if size < 100000:  # Less than 100KB = definitely wrong
                video_score = 0.2
                video_issues.append(f"Video file suspiciously small ({size} bytes)")
            
            # Check video duration via ffprobe
            try:
                probe = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", req.videoPath],
                    capture_output=True, text=True, timeout=30,
                )
                if probe.returncode == 0 and probe.stdout.strip():
                    duration = float(probe.stdout.strip())
                    if duration < 30:
                        video_issues.append(f"Video too short ({duration:.1f}s) — expected 2+ minutes")
                        video_score -= 0.5
                    elif duration > 600:
                        video_issues.append(f"Video too long ({duration:.1f}s) — expected under 10 minutes")
                        video_score -= 0.2
                else:
                    video_issues.append("Could not determine video duration")
                    video_score -= 0.3
            except Exception as e:
                video_issues.append(f"ffprobe failed: {str(e)[:100]}")
                video_score -= 0.3

            # Check for black frames at start
            try:
                ff = subprocess.run(
                    ["ffmpeg", "-i", req.videoPath, "-vframes", "1", "-f", "rawvideo", "-"],
                    capture_output=True, timeout=15,
                )
                if ff.returncode == 0 and ff.stdout:
                    # Check if frame is mostly black (all bytes near 0)
                    frame_bytes = ff.stdout[:10000]
                    avg_brightness = sum(frame_bytes) / len(frame_bytes)
                    if avg_brightness < 10:
                        video_issues.append("First frame appears to be black")
                        video_score -= 0.3
            except Exception:
                pass
    else:
        video_score = 0.0
        video_issues.append("No video path provided")

    categories["video"] = {
        "score": max(0.0, video_score),
        "issues": video_issues,
    }

    # --- 4. Scene Completeness ---
    completeness_score = 1.0
    completeness_issues = []
    
    clips_found = 0
    for i in range(1, scene_count + 1):
        clip_path = os.path.join(req.clipDir, f"anim_{i:02d}.mp4")
        alt_clip = os.path.join(req.clipDir, f"scene_{i:02d}.mp4")
        if os.path.exists(clip_path) or os.path.exists(alt_clip):
            clips_found += 1
        else:
            completeness_issues.append(f"Missing clip for scene {i}")
            completeness_score -= (1.0 / max(scene_count, 1))

    categories["completeness"] = {
        "score": max(0.0, completeness_score),
        "issues": completeness_issues,
        "clipsFound": clips_found,
        "expected": scene_count,
    }

    # --- 5. Subtitle Validation ---
    subtitle_score = 1.0
    subtitle_issues = []
    
    if req.subtitlePath:
        if not os.path.exists(req.subtitlePath):
            subtitle_score = 0.5
            subtitle_issues.append("Subtitle file not found (non-critical)")
        else:
            try:
                with open(req.subtitlePath, "r", encoding="utf-8") as f:
                    srt_content = f.read()
                subtitle_count = srt_content.count("-->")
                if subtitle_count < 3:
                    subtitle_issues.append(f"Very few subtitle entries ({subtitle_count})")
                    subtitle_score -= 0.3
            except Exception:
                subtitle_issues.append("Could not read subtitle file")
                subtitle_score -= 0.2

    categories["subtitles"] = {
        "score": max(0.0, subtitle_score),
        "issues": subtitle_issues,
    }

    # --- 6. Engagement Validation ---
    engagement_score = 1.0
    engagement_issues = []

    # Prefer scene-level timing if present (audioResult duration), fallback to scene duration.
    scene_durations = []
    narrations = []
    for s in req.scenes or []:
        ar = s.get("audioResult", {}) if isinstance(s, dict) else {}
        d = ar.get("duration") if isinstance(ar, dict) else None
        if not d:
            d = s.get("duration") if isinstance(s, dict) else None
        if d:
            try:
                scene_durations.append(float(d))
            except Exception:
                pass
        if isinstance(s, dict) and s.get("narration"):
            narrations.append(str(s.get("narration", "")))

    if scene_durations:
        avg_scene = sum(scene_durations) / len(scene_durations)
        if avg_scene > 11.5:
            engagement_issues.append(f"Average scene too slow ({avg_scene:.1f}s); target 6-10s")
            engagement_score -= 0.25
        if avg_scene < 4.5:
            engagement_issues.append(f"Average scene too fast ({avg_scene:.1f}s); pacing may feel rushed")
            engagement_score -= 0.1

    if narrations:
        def _norm_local(text: str) -> str:
            t = re.sub(r"[^a-z0-9\s]", " ", text.lower())
            return re.sub(r"\s+", " ", t).strip()
        unique_ratio = len({_norm_local(n) for n in narrations}) / max(1, len(narrations))
        if unique_ratio < 0.75:
            engagement_issues.append(f"Narration repetition detected (unique ratio {unique_ratio:.2f})")
            engagement_score -= 0.3

        first_norm = _norm_local(narrations[0])
        hook_tokens = ("suddenly", "mystery", "warning", "before", "secret", "storm", "alarm")
        if not any(tok in first_norm for tok in hook_tokens):
            engagement_issues.append("Opening hook is weak in first scene")
            engagement_score -= 0.15

    categories["engagement"] = {
        "score": max(0.0, engagement_score),
        "issues": engagement_issues,
    }

    # --- Calculate overall quality score ---
    weights = {
        "images": 0.25,
        "audio": 0.2,
        "video": 0.2,
        "completeness": 0.15,
        "subtitles": 0.05,
        "engagement": 0.15,
    }
    overall_score = sum(categories[cat]["score"] * w for cat, w in weights.items())
    
    passed = overall_score >= req.minQualityScore
    action = "upload" if passed else "regenerate"
    
    # Identify which scenes need regeneration
    failed_scenes = []
    for issue in image_issues + completeness_issues:
        import re as _re
        m = _re.search(r'scene[_ ](\d+)', issue)
        if m:
            sn = int(m.group(1))
            if sn not in failed_scenes:
                failed_scenes.append(sn)

    return {
        "success": True,
        "passed": passed,
        "qualityScore": round(overall_score, 3),
        "threshold": req.minQualityScore,
        "action": action,
        "failedScenes": sorted(failed_scenes),
        "categories": categories,
        "recommendation": (
            "Quality sufficient for upload" if passed
            else f"Quality below threshold ({overall_score:.2f} < {req.minQualityScore}). "
                 f"Regenerate scenes: {failed_scenes}" if failed_scenes
            else f"Quality below threshold ({overall_score:.2f} < {req.minQualityScore}). Review manually."
        ),
    }


# =============================================================================
# EPISODE CONTINUITY TRACKER (Task 20)
# Carries story state between episodes for series continuity.
# Tracks: character arcs, unresolved plot threads, world state, relationships.
# Auto-injected into the next episode's script generation prompt.
# =============================================================================


class EpisodeContinuity(BaseModel):
    """Continuity data recorded after each episode."""
    episodeNumber: int
    title: str = ""
    # Story state
    unresolvedThreads: list[str] = []      # Plot threads left open
    resolvedThreads: list[str] = []        # Plot threads closed this episode
    newDiscoveries: list[str] = []         # World-building facts established
    characterGrowth: dict = {}             # {character_id: "what they learned/changed"}
    relationships: dict = {}              # {pair: "current state"} e.g. "finn_misty": "becoming friends"
    cliffhanger: str = ""                 # The tease/hook for next episode
    # World state
    currentLocation: str = ""             # Where the characters are at episode end
    inventoryChanges: list[str] = []      # Items gained/lost
    worldState: dict = {}                 # Freeform world facts


@app.get("/api/continuity")
def get_continuity():
    """Get the current story continuity state (for injection into next episode)."""
    if not os.path.exists(CONTINUITY_FILE):
        return {
            "success": True,
            "hasHistory": False,
            "state": _default_continuity_state(),
            "note": "No episodes recorded yet. First episode starts fresh.",
        }

    data = _load_json(CONTINUITY_FILE)
    episodes = data.get("episodes", [])

    # Build cumulative state from all episodes
    state = _build_cumulative_state(episodes)

    return {
        "success": True,
        "hasHistory": len(episodes) > 0,
        "totalEpisodes": len(episodes),
        "latestEpisode": episodes[-1] if episodes else None,
        "state": state,
        "promptInjection": _build_continuity_prompt(state),
    }


@app.post("/api/continuity")
def record_continuity(ep: EpisodeContinuity):
    """Record continuity data after an episode is completed."""
    os.makedirs(os.path.dirname(CONTINUITY_FILE), exist_ok=True)

    data = {"episodes": []}
    if os.path.exists(CONTINUITY_FILE):
        data = _load_json(CONTINUITY_FILE)

    ep_dict = ep.model_dump()
    ep_dict["recordedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    data.setdefault("episodes", []).append(ep_dict)
    _save_json(CONTINUITY_FILE, data)

    return {
        "success": True,
        "episodeNumber": ep.episodeNumber,
        "unresolvedThreads": ep.unresolvedThreads,
        "cliffhanger": ep.cliffhanger,
    }


def _default_continuity_state() -> dict:
    """Default state for the first episode (no history)."""
    return {
        "activeThreads": [],
        "worldFacts": [],
        "characterStates": {},
        "relationships": {},
        "lastLocation": "The Treehouse",
        "inventory": [],
        "episodesCompleted": 0,
    }


def _build_cumulative_state(episodes: list[dict]) -> dict:
    """Build the cumulative continuity state from all episode records."""
    state = _default_continuity_state()

    all_resolved = set()
    for ep in episodes:
        # Track resolved threads
        for thread in ep.get("resolvedThreads", []):
            all_resolved.add(thread)

        # Active threads = unresolved from latest episodes, minus resolved
        for thread in ep.get("unresolvedThreads", []):
            if thread not in all_resolved and thread not in state["activeThreads"]:
                state["activeThreads"].append(thread)

        # World facts accumulate
        for fact in ep.get("newDiscoveries", []):
            if fact not in state["worldFacts"]:
                state["worldFacts"].append(fact)

        # Character growth (latest state wins)
        for cid, growth in ep.get("characterGrowth", {}).items():
            state["characterStates"][cid] = growth

        # Relationships (latest state wins)
        for pair, rel_state in ep.get("relationships", {}).items():
            state["relationships"][pair] = rel_state

        # Location from latest episode
        if ep.get("currentLocation"):
            state["lastLocation"] = ep["currentLocation"]

        # Inventory changes
        for item in ep.get("inventoryChanges", []):
            if item.startswith("-") and item[1:].strip() in state["inventory"]:
                state["inventory"].remove(item[1:].strip())
            elif not item.startswith("-"):
                state["inventory"].append(item.strip())

    # Remove resolved threads from active
    state["activeThreads"] = [t for t in state["activeThreads"] if t not in all_resolved]

    # Keep only last 5 active threads (avoid prompt bloat)
    state["activeThreads"] = state["activeThreads"][-5:]
    # Keep only last 10 world facts
    state["worldFacts"] = state["worldFacts"][-10:]

    state["episodesCompleted"] = len(episodes)
    return state


def _build_continuity_prompt(state: dict) -> str:
    """Build a prompt injection string for the next episode's script generation.
    This ensures story continuity is automatically maintained."""
    parts = []

    if state.get("episodesCompleted", 0) == 0:
        return "This is the FIRST episode. Establish the world and introduce all characters."

    parts.append(f"CONTINUITY FROM PREVIOUS {state['episodesCompleted']} EPISODE(S):")

    if state.get("activeThreads"):
        parts.append("UNRESOLVED PLOT THREADS (weave at least one into this episode):")
        for thread in state["activeThreads"]:
            parts.append(f"  - {thread}")

    if state.get("worldFacts"):
        parts.append("ESTABLISHED WORLD FACTS (do not contradict these):")
        for fact in state["worldFacts"][-5:]:
            parts.append(f"  - {fact}")

    if state.get("characterStates"):
        parts.append("CHARACTER DEVELOPMENT (build on this):")
        for cid, growth in state["characterStates"].items():
            parts.append(f"  - {cid}: {growth}")

    if state.get("relationships"):
        parts.append("RELATIONSHIPS:")
        for pair, rel in state["relationships"].items():
            parts.append(f"  - {pair}: {rel}")

    if state.get("lastLocation"):
        parts.append(f"LAST KNOWN LOCATION: {state['lastLocation']}")

    if state.get("inventory"):
        parts.append(f"ITEMS IN POSSESSION: {', '.join(state['inventory'])}")

    parts.append("")
    parts.append("USE this continuity to make the story feel connected. Reference past events naturally.")

    return "\n".join(parts)
