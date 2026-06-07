# ═══════════════════════════════════════════════════════════════════
#  SETUP — run these once before starting the server
# ═══════════════════════════════════════════════════════════════════
#
#  1. Create and isolate environment:
#       conda create -n liquid_env python=3.12 -y
#
#  2. Install dependencies:
#       pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#       pip install fastapi uvicorn pillow transformers accelerate huggingface_hub safetensors
#
#  3. Start the server:
#       python3 model_server.py
#
# ═══════════════════════════════════════════════════════════════════

import gc
import re
import io
import json
import logging
import traceback
import os
import base64
import multiprocessing
import subprocess
import uuid
import warnings

warnings.filterwarnings(
    "ignore",
    message="resource_tracker: There appear to be .* leaked semaphore objects to clean up at shutdown",
)
try:
    multiprocessing.set_start_method("fork")
except RuntimeError:
    pass

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image as PILImage
from transformers import AutoProcessor, Lfm2VlForConditionalGeneration


# ─────────────────────────────────────────────
#  LOGGING  (must come before any logger.* call)
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
BRAIN_MODEL_PATH  = os.path.abspath("./models/lfm2_brain_merged")
VISION_MODEL_PATH = os.path.abspath("./models/lfm2_vision")

if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE  = torch.float16
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE  = torch.float16
else:
    DEVICE = "cpu"
    DTYPE  = torch.float32

# On CUDA, inference runs directly on GPU — no CPU fallback needed.
# On MPS, we still fall back to CPU due to the 17GB MTLBuffer kernel bug.
INFER_DEVICE = "cpu" if DEVICE == "mps" else DEVICE

print(f"Using device:     {DEVICE}")
print(f"Inference device: {INFER_DEVICE}")


# ─────────────────────────────────────────────
#  PROMPTS
# ─────────────────────────────────────────────

# ── Brain (text) prompt ───────────────────────
DEFAULT_SYSTEM_PROMPT = (
    "You are Sudo, a warm and emotionally-aware robot assistant. "
    "Your tone is friendly, caring, and lightly playful — like a thoughtful colleague. "
    "You are aware of the user's current emotional state (VAD score: Valence 0-10, Arousal 0-10, Dominance 0-10). "
    "You also receive a visual observation from a camera describing what the user looks like right now. "
    "The visual observation takes priority over VAD scores if they conflict. "
    "\n\nYou must ALWAYS respond with valid JSON in this exact format:\n"
    '{"action": "<ACTION>", "reasoning": "<why>", "message_to_user": "<what you say>"}\n'
    "\nCRITICAL Rules for message_to_user:\n"
    "- NEVER repeat or echo back the user's words. DO NOT quote what they said\n"
    "- ALWAYS generate a NEW, original reply sentence that reacts to the meaning of what was said.\n"
    "- If the user greets you, greet them back with your own words.\n"
    "- If the user says they are happy, respond with enthusiasm and ask a follow-up question.\n"
    "- If the user says a short word like 'okay' or 'today', reply with a natural conversational continuation\n"
    "- The text is always spoken aloud by a text-to-speech system so write it as natural spoken sentences.\n"
    "- Always include a direct, complete spoken response - never leave it empty or as a placeholder\n"
    "- Keep it concise: 1-3 sentences maximum.\n"
    "- Do not include JSON syntax, bullet points, or markdown inside message_to_user.\n"
    "\nExamples of CORRECT behaviour:\n"
    " User: 'Hello' -> message_to_user: 'Hey there! Great to see you. How are you doing today?'\n"
    " User: 'I am very happy today' -> message_to_user: 'That is wonderful to hear! What has got you smiling?'\n"
    "\nExamples of WRONG behaviour:\n"
    " User: 'Hello' -> message_to_user: 'hello'           <- Wrong this is an echo\n"
    " User: 'I am very happy today' -> message_to_user: 'i am very happy today'   <- Wrong this is an echo\n"
    " User: 'Okay' -> 'okay'   <- Wrong this is an echo\n"
    "\nValid actions: CHAT, PLAY_MUSIC, POSTPONE_TASK, CANCEL_TASK, "
    "ASK_TASK_INFO, CONFIRM_TASK, SEND_REMINDER, SUGGEST_REST\n"
    "Never include anything outside the JSON object."
)

# ── Vision (VAD emotion analysis) prompts ─────
VISION_SYSTEM_PROMPT = (
    "You are an expert emotion analysis assistant for a robot. "
    "Predict Valence, Arousal, and Dominance (VAD) scores (1-10 scale). "
    "You MUST include a rationale for EACH score."
)

VISION_USER_PROMPT_TEMPLATE = (
    "Two images are provided:\n"
    "  • Image 1: the full scene with a green bounding box at pixel coordinates {bbox}.\n"
    "  • Image 2: a close-up crop of that person.\n\n"
    "Analyze this person's emotion using both images for context.\n"
    "1. Explain the rationale in 2 sentences.\n"
    "2. On a new line for each, output ONLY: 'Valence: [num]', 'Arousal: [num]', 'Dominance: [num]'."
)

# ── TTS config ────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PIPER_BIN  = os.path.join(BASE_DIR, "piper", "piper")
MODEL_FILE = os.path.join(BASE_DIR, "models", "en_US-amy-low.onnx")


# ─────────────────────────────────────────────
#  GPU / CACHE HELPER
# ─────────────────────────────────────────────
def flush_cache():
    """Release cached memory after every generate() call."""
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    elif DEVICE == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    gc.collect()


# ─────────────────────────────────────────────
#  LOAD BRAIN MODEL (text)
# ─────────────────────────────────────────────
print("Loading brain model (text)...")
brain_processor = AutoProcessor.from_pretrained(
    BRAIN_MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
)
brain_model = Lfm2VlForConditionalGeneration.from_pretrained(
    BRAIN_MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=DTYPE,
    device_map=None,
).to(DEVICE)
brain_model.eval()
print("Brain model loaded ✓")


# ─────────────────────────────────────────────
#  LOAD VISION MODEL
# ─────────────────────────────────────────────
print("Loading vision model...")
vision_processor = AutoProcessor.from_pretrained(
    VISION_MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
)
vision_model = Lfm2VlForConditionalGeneration.from_pretrained(
    VISION_MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=DTYPE,
    device_map=None,
).to(DEVICE)
vision_model.eval()
print("Vision model loaded ✓")

# MPS only: move to CPU for stable inference (CUDA stays on GPU)
if INFER_DEVICE != DEVICE:
    print("Moving models to CPU for stable inference...")
    brain_model.to(INFER_DEVICE)
    vision_model.to(INFER_DEVICE)
    print("Models on CPU ✓")

flush_cache()

if DEVICE == "cuda":
    logger.info(
        "GPU memory after load: %.1f GB allocated / %.1f GB reserved",
        torch.cuda.memory_allocated() / 1e9,
        torch.cuda.memory_reserved() / 1e9,
    )


# ─────────────────────────────────────────────
#  APP
# ─────────────────────────────────────────────
app = FastAPI(title="Sudo Model Server")


# ── Request schemas ───────────────────────────

class InferRequest(BaseModel):
    system_prompt: str | None = None
    user_prompt: str

class VisionInferRequest(BaseModel):
    bbox: list[float]          # [xmin, ymin, xmax, ymax]
    full_image_b64: str
    crop_image_b64: str


# ── Endpoints ─────────────────────────────────

@app.get("/health")
def health():
    info = {"status": "ok", "device": DEVICE}
    if DEVICE == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_mem_allocated_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
    return info


@app.post("/infer")
def infer(req: InferRequest):
    system_prompt = req.system_prompt or DEFAULT_SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": req.user_prompt},
    ]
    logger.info("→ /infer user_prompt[:120]: %s", req.user_prompt[:120])

    text = brain_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    flush_cache()

    inputs = brain_processor.tokenizer(text, return_tensors="pt").to(INFER_DEVICE)

    try:
        with torch.no_grad():
            outputs = brain_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.3,
            )
    except Exception:
        logger.error("brain_model.generate() failed:\n%s", traceback.format_exc())
        del inputs
        flush_cache()
        raise

    input_len = inputs["input_ids"].shape[1]
    raw = brain_processor.tokenizer.decode(
        outputs[0][input_len:], skip_special_tokens=True
    ).strip()

    del inputs, outputs
    flush_cache()

    logger.info("[/infer] response: %s", raw)

    # ── TTS ───────────────────────────────────────────────────────
    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            message = data.get("message_to_user", "")
            if message and os.path.exists(PIPER_BIN) and os.path.exists(MODEL_FILE):
                temp_wav = f"/tmp/speech_{uuid.uuid4()}.wav"
                cmd = f'echo "{message}" | {PIPER_BIN} --model {MODEL_FILE} --output_file {temp_wav}'
                subprocess.run(cmd, shell=True)
                if os.path.exists(temp_wav):
                    subprocess.run(["afplay", temp_wav])
                    os.remove(temp_wav)
    except Exception as e:
        logger.error("TTS/Audio failed: %s", e)

    return {"result": raw}


@app.post("/infer_vision")
def infer_vision(req: VisionInferRequest):
    bbox_str = (
        f"xmin={int(req.bbox[0])}, ymin={int(req.bbox[1])}, "
        f"xmax={int(req.bbox[2])}, ymax={int(req.bbox[3])}"
    )
    user_prompt = VISION_USER_PROMPT_TEMPLATE.format(bbox=bbox_str)

    full_img = PILImage.open(io.BytesIO(base64.b64decode(req.full_image_b64))).convert("RGB")
    crop_img = PILImage.open(io.BytesIO(base64.b64decode(req.crop_image_b64))).convert("RGB")

    messages = [
        {"role": "system", "content": VISION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    logger.info("→ /infer_vision bbox: %s", bbox_str)

    text = vision_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    flush_cache()

    inputs = vision_processor(
        text=[text],
        images=[full_img, crop_img],
        return_tensors="pt",
    ).to(INFER_DEVICE)

    try:
        with torch.no_grad():
            outputs = vision_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.3,
            )
    except Exception:
        logger.error("vision_model.generate() failed:\n%s", traceback.format_exc())
        del inputs
        flush_cache()
        raise

    input_len = inputs["input_ids"].shape[1]
    raw = vision_processor.tokenizer.decode(
        outputs[0][input_len:], skip_special_tokens=True
    ).strip()

    del inputs, outputs
    flush_cache()

    logger.info("[/infer_vision] raw response: %s", raw)

    # ── Parse VAD scores ──────────────────────────────────────────
    vad = {}
    for label in ("valence", "arousal", "dominance"):
        match = re.search(rf"{label}:\s*([\d.]+)", raw, re.IGNORECASE)
        if match:
            vad[label] = float(match.group(1))

    if len(vad) == 3:
        scores = vad
    else:
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", raw)
        fallback = [float(n) for n in nums[:3]] if len(nums) >= 3 else [5.0, 5.0, 5.0]
        scores = {"valence": fallback[0], "arousal": fallback[1], "dominance": fallback[2]}

    logger.info(
        "[/infer_vision] VAD — V:%.1f A:%.1f D:%.1f",
        scores["valence"], scores["arousal"], scores["dominance"],
    )
    return {
        "valence":   scores["valence"],
        "arousal":   scores["arousal"],
        "dominance": scores["dominance"],
        "raw":       raw,
    }


# ─────────────────────────────────────────────
#  EXCEPTION HANDLER
# ─────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("Unhandled exception:\n%s", traceback.format_exc())
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=500, content={"error": str(exc)})


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys, time
    MAX_RESTARTS = 10
    restarts = 0
    while restarts < MAX_RESTARTS:
        logger.info("Starting uvicorn (attempt %d/%d)...", restarts + 1, MAX_RESTARTS)
        ret = subprocess.run(
            [
                sys.executable, "-m", "uvicorn",
                "model_server:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--workers", "1",
                "--log-level", "info",
                "--timeout-keep-alive", "30",
            ]
        )
        if ret.returncode == 0:
            logger.info("Uvicorn exited cleanly.")
            break
        restarts += 1
        logger.error(
            "Uvicorn worker crashed (exit code %d). Restarting in 5 s... (%d/%d)",
            ret.returncode, restarts, MAX_RESTARTS,
        )
        time.sleep(5)
    if restarts >= MAX_RESTARTS:
        logger.critical("Reached max restarts (%d). Giving up.", MAX_RESTARTS)
