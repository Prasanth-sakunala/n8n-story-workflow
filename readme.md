# AI Animated Story — YouTube Automation Pipeline

Fully automated children's story episode pipeline: story generation → image creation → voice acting → video assembly → YouTube upload.

**Series:** Finn, Squeaky & Misty Adventures  
**Target:** Kids 4-10, families  
**Style:** Cinematic storybook illustration, animated movie concept art  
**Output:** 2-4 minute YouTube episodes (retention-optimized), fully automated daily

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────────────┐
│  n8n        │────▶│  FastAPI      │────▶│  External Services │
│  (port 5679)│     │  (port 8001)  │     │                   │
│  Orchestrator│     │  Generation   │     │  • Pollinations.ai │
└─────────────┘     │  Engine       │     │    (primary images)│
                    └──────────────┘     │  • ComfyUI (8188)  │
                                          │    (fallback+anim) │
                                          │  • Ollama (11434)  │
                                          │  • SadTalker (8189)│
                                          │  • Groq API        │
                                          │  • YouTube API     │
                                          └───────────────────┘
```

---

## Required Files (Don't Delete These)

```
├── docker-compose.yml          # Starts n8n + API containers
├── workflow.json                # n8n workflow (import into n8n UI)
├── sadtalker_server.py         # Optional: lip sync server (run on host with GPU)
├── api/
│   ├── Dockerfile              # API container build
│   ├── main.py                 # Core API service (6500+ lines, all endpoints)
│   ├── requirements.txt        # Python dependencies
│   └── data/
│       ├── characters.json     # Bundled character defaults
│       └── story_seeds.json    # Bundled story seed defaults
├── credentials/
│   └── client_secret.json      # YouTube OAuth2 credentials (you provide)
├── data/
│   ├── characters/
│   │   └── characters.json     # Active character registry (auto-created)
│   ├── stories/                # Generated scripts, history, continuity
│   ├── images/                 # Generated scene keyframes
│   ├── audio/                  # Generated TTS audio + mixed tracks
│   ├── clips/                  # Animated scene clips
│   ├── video/                  # Final assembled videos
│   ├── music/                  # Background music library (YOU ADD THESE)
│   ├── thumbnails/             # Generated thumbnails
│   └── credentials/            # YouTube OAuth token (auto-generated)
└── .env                        # API keys (you create this)
```

---

## Requirements

### Software (Must Be Installed)

| Software | Version | Purpose | Install |
|----------|---------|---------|---------|
| **Docker** | 20+ | Runs n8n + API containers | https://docs.docker.com/get-docker/ |
| **Docker Compose** | v2+ | Multi-container orchestration | Included with Docker Desktop |

> **Note:** ComfyUI is now **optional** — Pollinations.ai (FLUX model) is the primary automatic image generator. The imported workflow is currently set to manual image mode.

### Optional Software

| Software | Purpose | When Needed |
|----------|---------|-------------|
| **ComfyUI** | Image fallback + AnimateDiff animation | Only if Pollinations fails, or for AnimateDiff |
| **Ollama** | Local LLM fallback | Only if Gemini + Groq both fail |
| **SadTalker** | Lip sync on dialogue scenes | Only for lip-synced character animation |
| **GPU** | Image gen fallback + animation | Required for ComfyUI (6GB+ VRAM) |

### API Keys (Free Tier)

| Key | Get It From | Purpose |
|-----|-------------|---------|
| **POLLINATIONS_API_KEY** | https://enter.pollinations.ai | Primary image generation (FLUX) |
| **GEMINI_API_KEY** | https://aistudio.google.com/apikey | Script generation + SEO (primary LLM) |
| **GROQ_API_KEY** | https://console.groq.com/keys | Script fallback (Llama 4 Scout) |
| **YouTube OAuth** | https://console.cloud.google.com | Video upload (client_secret.json) |

### Image Generation Models

| Model | Provider | Purpose | Notes |
|-------|----------|---------|-------|
| **Pollinations.ai FLUX** | Pollinations.ai API | **Primary** image generator | Free tier, no GPU needed |
| **ComfyUI SD1.5** | ComfyUI (local) | Fallback if Pollinations fails | Requires GPU, optional |
| **v3_sd15_mm.ckpt** | ComfyUI (local) | AnimateDiff motion module | For animated clips (optional) |
| **ip-adapter-plus_sd15** | ComfyUI (local) | Character consistency reference | Only used with ComfyUI fallback |

### LLM Models Used

| Model | Provider | Cost | Purpose |
|-------|----------|------|---------|
| **Gemini 2.5 Flash** | Google | Free (rate-limited) | Primary script + SEO generation |
| **Llama 4 Scout 17B** | Groq | Free (rate-limited) | Fallback script generation |
| **Llama 3.1 8B** | Ollama (local) | Free (your hardware) | Last resort fallback |

---

## Setup Commands

### 1. Create .env file

```bash
# Create .env in the project root
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
PEXELS_API_KEY=            # Optional, for stock backgrounds
```

### 2. Start the stack

```bash
# Build and start everything
docker compose up -d --build

# Check logs
docker compose logs -f

# Verify health
curl http://localhost:8001/health
```

### 3. Import n8n workflow

```
1. Open n8n at http://localhost:5679
2. Go to Workflows → Import from File
3. Select workflow.json
4. Activate the workflow
```

### 4. Setup YouTube OAuth (one-time)

```bash
# 1. Place your client_secret.json in credentials/
# 2. Get auth URL
curl http://localhost:8001/api/youtube-auth-url

# 3. Open the URL in browser, authorize, copy the code
# 4. Exchange code for token
curl -X POST http://localhost:8001/api/youtube-auth-callback \
  -H "Content-Type: application/json" \
  -d '{"code": "YOUR_AUTH_CODE"}'
```

### 5. Generate character reference sheets (one-time)

```bash
# Requires ComfyUI running on port 8188
curl -X POST http://localhost:8001/api/generate-all-reference-sheets \
  -H "Content-Type: application/json" \
  -d '{}'
```

### 6. Add background music

```
Place royalty-free .mp3 files in data/music/ named by mood:
  adventure_theme.mp3
  mystery_ambient.mp3
  cheerful_morning.mp3
  gentle_piano.mp3
  hero_march.mp3
  spooky_ambient.mp3
  etc.

Check GET /api/music-library for the full mood mapping.
```

### 7. Start ComfyUI (on host)

```bash
# ComfyUI must be running on port 8188
cd /path/to/ComfyUI
python main.py --listen 0.0.0.0 --port 8188
```

### 8. (Optional) Start SadTalker server

```bash
python sadtalker_server.py
# Runs on port 8189
```

---

## Running the Workflow

### Automatic (Daily)
The n8n workflow triggers daily at 9 AM. It:
1. Fetches story ideas (Reddit + local seeds)
2. Picks the best story
3. Generates script with Gemini 2.5 Flash
4. Plans scenes (prompts, audio, motion)
5. Uses manual keyframe images from `/data/images/manual`
6. Generates TTS audio via Edge-TTS
7. Mixes audio with background music
8. Assembles video with transitions + subtitles
9. Generates thumbnail + SEO metadata
10. Uploads to YouTube + adds to playlist
11. Records episode continuity for next time

### Manual Trigger
```
In n8n UI → Open the workflow → Click "Execute Workflow"
```

### Manual Image Handoff

The imported workflow is configured for manual image mode. It keeps the generated story and scene plan, then fails clearly at `Generate All Visuals` if required image files are missing.

1. Run the workflow from `Manual Trigger`.
2. When it reaches `Generate All Visuals`, open the previous `Collect Scene Plan` node output.
3. Use `manualImageGuide.requiredFiles` to generate each scene image manually.
4. Place images in `data/images/manual/` using filenames like `scene_01.png`, `scene_02.png`, etc.
5. Optional: add alternate shots like `scene_01_b.png` and a `thumbnail.png`.
6. In n8n, retry the failed execution from `Generate All Visuals`; do not start from `Manual Trigger`, or it will pick a new story.

### API-Only (Test Individual Steps)

```bash
# Check system status
curl http://localhost:8001/health
curl http://localhost:8001/api/llm-status
curl http://localhost:8001/api/comfyui-status

# Get character consistency data
curl http://localhost:8001/api/character-consistency

# Get voice assignments
curl http://localhost:8001/api/voice-lock

# Get episode continuity
curl http://localhost:8001/api/continuity

# Generate script only
curl -X POST http://localhost:8001/api/generate-script \
  -H "Content-Type: application/json" \
  -d '{
    "storyTitle": "The Crystal Cave",
    "storyPremise": "Finn discovers a hidden cave that glows with mysterious crystals",
    "targetDuration": 5
  }'
```

---

## All API Endpoints (39 total)

### Story Engine
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | System health check |
| GET | `/api/characters` | Character registry |
| POST | `/api/characters` | Add/update character |
| POST | `/api/locations` | Add/update location |
| GET | `/api/story-seeds` | Available story seeds |
| POST | `/api/story-seeds` | Add story seed |
| POST | `/api/fetch-story-ideas` | Fetch from Reddit + local |
| POST | `/api/pick-story` | Score and select best story |
| GET | `/api/story-history` | Past episodes |
| POST | `/api/story-history` | Record completed episode |

### Character Consistency
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/character-consistency` | Frozen canonical prompts + memory |
| POST | `/api/character-consistency/validate` | Check prompt for character drift |
| POST | `/api/generate-reference-sheet` | Generate character ref sheet |
| POST | `/api/generate-all-reference-sheets` | Ref sheets for all characters |

### Script Generation
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/llm-status` | Check LLM provider availability |
| POST | `/api/generate-script` | Generate full episode script |

### Scene Planning
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/plan-scenes` | Build image/audio/motion tasks per scene |
| POST | `/api/plan-transitions` | Plan transitions between scenes |
| POST | `/api/plan-motion` | Plan cinematic motion (Ken Burns + parallax + drift) |
| GET | `/api/transitions` | Transition library |

### Visual Generation
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/comfyui-status` | Check ComfyUI + AnimateDiff |
| POST | `/api/generate-visuals` | Generate keyframes + animate |

### Audio
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/tts-voices` | Available TTS voices |
| GET | `/api/voice-lock` | Frozen voice assignments |
| POST | `/api/generate-audio` | Generate TTS + subtitles |
| POST | `/api/mix-audio` | Mix voice + music with ducking |
| GET | `/api/music-library` | Available background music |
| POST | `/api/lip-sync` | Apply SadTalker lip sync |
| GET | `/api/sadtalker-status` | Check SadTalker availability |

### Assembly & Quality
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/assemble-video` | Final video assembly |
| POST | `/api/validate-video` | **Pre-upload quality validation** |

### Publishing
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/generate-seo` | Generate YouTube metadata |
| POST | `/api/generate-thumbnail` | Generate thumbnail image |
| GET | `/api/youtube-auth-url` | OAuth2 setup URL |
| GET | `/api/youtube-auth-callback` | Exchange auth code |
| GET | `/api/youtube-auth-status` | Check YouTube auth |
| POST | `/api/upload-youtube` | Upload to YouTube |
| POST | `/api/youtube-playlist-add` | Add to series playlist |

### Continuity
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/continuity` | Get story state for next episode |
| POST | `/api/continuity` | Record episode continuity |

---

## Characters (Frozen — Never Change)

| Character | Role | Voice | Visual Identity |
|-----------|------|-------|-----------------|
| **Captain Finn** | Protagonist | en-US-GuyNeural | Brown messy hair, blue eyes, blue hoodie + star badge, brown shorts, backpack |
| **Squeaky** | Sidekick | en-US-AnaNeural | Tiny orange mouse, oversized ears, green eyes, red scarf, green backpack |
| **Misty** | Rival | en-US-JennyNeural | Purple cat, brass goggles, yellow eyes, green utility belt |
| **Narrator** | — | en-GB-RyanNeural | (no visual) |

---

## Key Design Principles

1. **Character Consistency > Everything** — Same appearance in every frame (GLOBAL_STYLE_PROMPT + canonical prompts + scene reference chaining)
2. **Story Continuity** — Episode state carries forward automatically (episode bible + continuity engine)
3. **Visual Quality** — Cinematic illustrated storybook look, never photorealistic or anime (global style lock)
4. **Voice Consistency** — Each character's voice NEVER changes (voice lock system)
5. **Production Reliability** — Retries, fallback providers, rate limit handling, quality validation gate
6. **Future Model Flexibility** — Provider abstraction layer (swap Imagen/Flux/SDXL without code changes)
7. **High Retention Storytelling** — 2-4 min episodes, fast pacing, hooks, curiosity loops, emotional arcs

---

## System Architecture (10 Sections)

| Section | System | Status |
|---------|--------|--------|
| 1 | Character Memory (canonical identity + scene reference chaining) | ✅ |
| 2 | Global Visual Style Lock (GLOBAL_STYLE_PROMPT on every image) | ✅ |
| 3 | Story Pacing (2-4 min retention mode, 5-8 min standard mode) | ✅ |
| 4 | Quality Validation Pipeline (`/api/validate-video` pre-upload gate) | ✅ |
| 5 | Episode Bible / Continuity Engine (relationships, threads, world facts) | ✅ |
| 6 | Provider Abstraction Layer (ImageProvider, ScriptProvider interfaces) | ✅ |
| 7 | Motion System (Ken Burns + parallax + cinematic drift + camera shake) | ✅ |
| 8 | Retention Engine (hooks, catchphrases, curiosity loops in script prompt) | ✅ |
| 9 | Production Stability (retries, backoff, rate limit handling, fallbacks) | ✅ |
| 10 | Final Goal: Cinematic illustrated storytelling with emotional connection | ✅ |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| ComfyUI not reachable | Not critical — Pollinations.ai is primary. For animation fallback: `python main.py --listen 0.0.0.0 --port 8188` |
| Gemini API fails | Check GEMINI_API_KEY in .env, verify at https://aistudio.google.com |
| No audio generated | Ensure container has internet access (Edge-TTS needs Microsoft servers) |
| Characters look different | Run `/api/generate-all-reference-sheets` — uses Pollinations FLUX for consistent refs |
| YouTube upload fails | Re-run OAuth: GET `/api/youtube-auth-url` and complete flow |
| n8n can't reach API | Check `docker compose logs api` — API must be healthy first |
| Low VRAM errors | Reduce imageWidth/imageHeight in plan-scenes (default 512x384) |

---

## Stop / Restart

```bash
# Stop everything
docker compose down

# Restart
docker compose up -d

# Rebuild after code changes
docker compose up -d --build

# View API logs
docker compose logs -f api

# View n8n logs
docker compose logs -f n8n
