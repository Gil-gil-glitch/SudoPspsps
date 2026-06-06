# ═══════════════════════════════════════════════════════════════════
#  SETUP — run these once on Mac before starting the server
# ═══════════════════════════════════════════════════════════════════
#
#  1. Create and isolate environment:
#       conda create -n liquid_env python=3.12 -y
#
#  2. Install dependencies:
#       /opt/anaconda3/envs/liquid_env/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
#       /opt/anaconda3/envs/liquid_env/bin/pip install fastapi uvicorn pillow transformers accelerate huggingface_hub safetensors
#
#  3. Start the server:
#       /opt/anaconda3/envs/liquid_env/bin/python3 model_server.py
#
# ═══════════════════════════════════════════════════════════════════

import gc
import io
import logging
import traceback
import os
import base64
import multiprocessing
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
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image as PILImage
from transformers import AutoProcessor, Lfm2VlForConditionalGeneration


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
BRAIN_MODEL_PATH  = os.path.abspath("./models/lfm2_brain")
VISION_MODEL_PATH = os.path.abspath("./models/lfm2_vision")

if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE  = torch.float16
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE  = torch.float16
else:
    DEVICE = "cpu"
    DTYPE  = torch.float32

print(f"Using device: {DEVICE}")

# MPS has a kernel bug with this model architecture where generation
# triggers a tensor shape overflow producing a nonsensical 17 GB
# MTLBuffer allocation that kills the process.  The fix is to move
# the model to CPU for the generate() call, then move it back to MPS
# so the next request starts with GPU-resident weights.
#
# The context manager below handles that transparently.
INFER_DEVICE = "cpu" if DEVICE == "mps" else DEVICE
print(f"Inference device: {INFER_DEVICE}")


from contextlib import contextmanager

@contextmanager
def on_infer_device(model):
    """Temporarily move *model* to INFER_DEVICE, yield, then move it back.

    On CUDA or CPU (INFER_DEVICE == DEVICE) this is a no-op so there is
    no overhead.  On MPS it shifts the weights to CPU before generate()
    and back to MPS afterwards, keeping them warm for the next request.
    """
    if INFER_DEVICE == DEVICE:
        yield model
    else:
        model.to(INFER_DEVICE)
        flush_mps_cache()
        try:
            yield model
        finally:
            model.to(DEVICE)
            flush_mps_cache()


# ─────────────────────────────────────────────
#  SYSTEM PROMPT  (stored server-side)
#
#  FIX #1: The system prompt now lives here on the server.
#  The client (cat_brain_planner.py) no longer needs to transmit
#  it on every request, reducing payload size and eliminating the
#  risk of a future prompt change causing a tokenisation spike.
# ─────────────────────────────────────────────
DEFAULT_SYSTEM_PROMPT = (
    "You are Sudo, a warm and emotionally-aware robot assistant. "
    "Your tone is friendly, caring, and lightly playful — like a thoughtful colleague. "
    "You are aware of the user's current emotional state (VAD score: Valence 0-10, Arousal 0-10, Dominance 0-10). "
    "You also receive a visual observation from a camera describing what the user looks like right now. "
    "The visual observation takes priority over VAD scores if they conflict. "
    "\n\nYou must ALWAYS respond with valid JSON in this exact format:\n"
    '{"action": "<ACTION>", "reasoning": "<why>", "message_to_user": "<what you say>"}\n'
    "\nValid actions: CHAT, PLAY_MUSIC, POSTPONE_TASK, CANCEL_TASK, "
    "ASK_TASK_INFO, CONFIRM_TASK, SEND_REMINDER, SUGGEST_REST\n"
    "Never include anything outside the JSON object."
)


# ─────────────────────────────────────────────
#  MPS MEMORY HELPER
#
#  FIX #2: MPS does not release cached buffers between inference
#  calls the way CUDA does, so memory fragments over time until a
#  single large contiguous allocation fails (the 17 GB MTLBuffer
#  error you saw).  Calling this after every generation reclaims
#  that cached memory before it accumulates.
# ─────────────────────────────────────────────
def flush_mps_cache():
    if DEVICE == "mps":
        # synchronise and empty the MPS allocator cache
        torch.mps.synchronize()
        torch.mps.empty_cache()
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()
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
).to(DEVICE)
vision_model.eval()
print("Vision model loaded ✓")

# Warm up the MPS allocator so the first real request doesn't pay
# the cold-start cost and immediately stress the allocator.
flush_mps_cache()


# ─────────────────────────────────────────────
#  APP
# ─────────────────────────────────────────────
app = FastAPI(title="Sudo Model Server")


# ── Request schemas ───────────────────────────

class InferRequest(BaseModel):
    # FIX #1 (continued): system_prompt is now optional.
    # If omitted the server uses DEFAULT_SYSTEM_PROMPT above.
    # The client can still override it when needed (e.g. for the
    # intervention-gate prompt that uses a different JSON schema).
    system_prompt: str | None = None
    user_prompt: str

class VisionInferRequest(BaseModel):
    system_prompt: str | None = None
    user_prompt: str
    full_image_b64: str
    crop_image_b64: str


# ── Endpoints ─────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


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

    flush_mps_cache()

    inputs = brain_processor.tokenizer(text, return_tensors="pt").to(INFER_DEVICE)

    try:
        with on_infer_device(brain_model), torch.no_grad():
            outputs = brain_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                repetition_penalty=1.1,
            )
    except Exception:
        logger.error("brain_model.generate() failed:\n%s", traceback.format_exc())
        del inputs
        flush_mps_cache()
        raise

    input_len = inputs["input_ids"].shape[1]
    raw = brain_processor.tokenizer.decode(
        outputs[0][input_len:], skip_special_tokens=True
    ).strip()

    del inputs, outputs
    flush_mps_cache()

    logger.info("[/infer] response: %s", raw)
    return {"result": raw}


@app.post("/infer_vision")
def infer_vision(req: VisionInferRequest):
    system_prompt = req.system_prompt or DEFAULT_SYSTEM_PROMPT

    full_img = PILImage.open(io.BytesIO(base64.b64decode(req.full_image_b64))).convert("RGB")
    crop_img = PILImage.open(io.BytesIO(base64.b64decode(req.crop_image_b64))).convert("RGB")

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": req.user_prompt},
            ],
        },
    ]
    text = vision_processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    flush_mps_cache()

    inputs = vision_processor(
        images=[full_img, crop_img],
        text=[text],
        return_tensors="pt",
    ).to(INFER_DEVICE)

    try:
        with on_infer_device(vision_model), torch.no_grad():
            out = vision_model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
            )
    except Exception:
        logger.error("vision_model.generate() failed:\n%s", traceback.format_exc())
        del inputs
        flush_mps_cache()
        raise

    response = vision_processor.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    del inputs, out
    flush_mps_cache()

    return {"result": response}


# ─────────────────────────────────────────────
#  LOGGING + ENTRY POINT
#  Surface MPS/Metal assertion failures that uvicorn
#  would otherwise swallow silently
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("Unhandled exception:\n%s", traceback.format_exc())
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=500, content={"error": str(exc)})


if __name__ == "__main__":
    # FIX #3 (revised): workers=2 caused a double model-load OOM because
    # uvicorn forks *after* the module-level model loading runs, so each
    # worker tried to put a full copy of both models onto MPS (9+ GB each).
    #
    # Solution: single worker + a restart loop.  If a Metal assertion kills
    # the process, the loop relaunches it within ~10 s rather than leaving
    # the server permanently down.  Models are loaded once, in the parent,
    # before uvicorn starts — the child worker inherits them via fork.
    import subprocess, sys, time
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
