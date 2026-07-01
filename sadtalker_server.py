"""
SadTalker Lip Sync Server
==========================
Run this on your HOST machine (needs GPU access).
Listens on port 8189 and provides lip sync via HTTP API.

SETUP:
  1. Install SadTalker:
     pip install sadtalker
     OR clone: git clone https://github.com/OpenTalker/SadTalker.git
     
  2. Install dependencies:
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     pip install flask numpy scipy face_alignment dlib
     pip install gfpgan basicsr facexlib realesrgan

  3. Download pretrained models:
     - Place in SadTalker/checkpoints/
     - Download from: https://github.com/OpenTalker/SadTalker#-2-download-trained-models
     
  4. Run this server:
     python sadtalker_server.py

The n8n story workflow will automatically call this server for lip sync.
If this server is not running, the pipeline still works (just without lip sync).
"""

import base64
import io
import os
import sys
import tempfile
import traceback

from flask import Flask, jsonify, request

app = Flask(__name__)

# ============================================
# CONFIG — adjust these paths for your setup
# ============================================
SADTALKER_DIR = os.environ.get("SADTALKER_DIR", r"C:\SadTalker")
CHECKPOINT_DIR = os.path.join(SADTALKER_DIR, "checkpoints")
RESULT_DIR = os.path.join(SADTALKER_DIR, "results")

# Add SadTalker to path
if SADTALKER_DIR not in sys.path:
    sys.path.insert(0, SADTALKER_DIR)

# Will be initialized on first request
sadtalker_instance = None


def _get_sadtalker():
    """Lazy-load SadTalker model (first call takes ~10s)."""
    global sadtalker_instance
    if sadtalker_instance is None:
        try:
            from src.gradio_demo import SadTalker
            sadtalker_instance = SadTalker(
                lazy_load=True,
                checkpoint_path=CHECKPOINT_DIR,
            )
            print("[SadTalker] Model loaded successfully")
        except ImportError:
            # Try alternative import for newer versions
            try:
                from inference import SadTalkerInference
                sadtalker_instance = SadTalkerInference(
                    checkpoint_dir=CHECKPOINT_DIR,
                )
                print("[SadTalker] Model loaded (v2 API)")
            except Exception as e:
                print(f"[SadTalker] Failed to load: {e}")
                raise
    return sadtalker_instance


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "sadtalker", "gpu": True})


@app.route("/api/animate", methods=["POST"])
def animate():
    """
    Generate lip-synced video from face image + audio.
    
    Request JSON:
        image: base64-encoded PNG image
        audio: base64-encoded WAV audio
        still_mode: bool (True = only mouth moves, best for cartoon)
        expression_scale: float (0.5-1.5)
        preprocess: str ("crop" | "resize" | "full")
    
    Response JSON:
        success: bool
        video: base64-encoded MP4
        duration: float (seconds)
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON body"}), 400

        image_b64 = data.get("image")
        audio_b64 = data.get("audio")
        still_mode = data.get("still_mode", True)
        expression_scale = data.get("expression_scale", 1.0)
        preprocess = data.get("preprocess", "crop")

        if not image_b64 or not audio_b64:
            return jsonify({"success": False, "error": "Missing image or audio"}), 400

        # Save inputs to temp files
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(base64.b64decode(image_b64))
            temp_image = f.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(base64.b64decode(audio_b64))
            temp_audio = f.name

        # Run SadTalker
        st = _get_sadtalker()
        
        # Output path
        os.makedirs(RESULT_DIR, exist_ok=True)
        
        # Call SadTalker inference
        try:
            # SadTalker API (original version)
            result_path = st.test(
                source_image=temp_image,
                driven_audio=temp_audio,
                preprocess=preprocess,
                still_mode=still_mode,
                expression_scale=expression_scale,
                result_dir=RESULT_DIR,
                enhancer=None,  # Skip GFPGAN for speed (cartoon doesn't need it)
                batch_size=2,
                size=256,  # 256px face region is sufficient for cartoon
                pose_style=0,
            )
        except TypeError:
            # Newer SadTalker API
            result_path = st.generate(
                image_path=temp_image,
                audio_path=temp_audio,
                still=still_mode,
                exp_scale=expression_scale,
                preprocess=preprocess,
            )

        # Read result video
        if result_path and os.path.exists(result_path):
            with open(result_path, "rb") as f:
                video_b64 = base64.b64encode(f.read()).decode("utf-8")

            # Cleanup
            for p in [temp_image, temp_audio, result_path]:
                try:
                    os.remove(p)
                except Exception:
                    pass

            return jsonify({
                "success": True,
                "video": video_b64,
            })
        else:
            return jsonify({"success": False, "error": "No output generated"}), 500

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        # Ensure cleanup
        for var in ["temp_image", "temp_audio"]:
            path = locals().get(var)
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass


if __name__ == "__main__":
    print("=" * 60)
    print("  SadTalker Lip Sync Server")
    print("  Listening on http://0.0.0.0:8189")
    print("  Models dir:", CHECKPOINT_DIR)
    print("=" * 60)
    print()
    print("To test: curl http://localhost:8189/health")
    print()
    app.run(host="0.0.0.0", port=8189, debug=False)
