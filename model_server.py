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

"""
2026-06-07 01:36:17,602 [INFO] → /infer user_prompt[:120]: User said: 'Hello, I am happy.'. Visual observation from camera: 'Neutral'. Emotion scores — valence: 5.0/10, arousal: 5
[transformers] Ignoring clean_up_tokenization_spaces=True for BPE tokenizer TokenizersBackend. The clean_up_tokenization post-processing step is designed for WordPiece tokenizers and is destructive for BPE (it strips spaces before punctuation). Set clean_up_tokenization_spaces=False to suppress this warning, or set clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output=True to force cleanup anyway.
2026-06-07 01:36:47,076 [INFO] [/infer] response: {"action": "CHAT", "reasoning": "prioritized over scoring when the camera shows neutral expression", "message_to_user": "'Hello' → 'hey there!'"}
INFO:     10.42.0.1:59524 - "POST /infer HTTP/1.1" 200 OK
2026-06-07 01:37:09,888 [INFO] → /infer user_prompt[:120]: User said: 'I'd like to eat curry.'. Visual observation from camera: 'Neutral'. Emotion scores — valence: 5.0/10, arousa
2026-06-07 01:37:39,926 [INFO] → /infer user_prompt[:120]: User said: 'You'. Visual observation from camera: 'Neutral'. Emotion scores — valence: 5.0/10, arousal: 5.0/10, dominanc
2026-06-07 01:37:40,786 [INFO] [/infer] response: {"action": "CHAT", "reasoning": "The camera observed neutral expression, high arousal level, and low domination level when compared to the given emotions, which aligns with typical behavior at these values.", "message_to_user": "Curry sounds delicious! When’s dinner? I’m craving some spice."}
2026-06-07 01:38:08,519 [INFO] [/infer] response: {"action": "CHAT", "reasoning": "camera observation aligns with high valence and low intensity, prompting a soft, empathetic response.", "message_to_user": "Nice to meet you too!"}
INFO:     10.42.0.1:33966 - "POST /infer HTTP/1.1" 200 OK
2026-06-07 01:38:08,527 [INFO] → /infer user_prompt[:120]: User said: 'Dinner is at 3am.'. Visual observation from camera: 'Neutral'. Emotion scores — valence: 5.0/10, arousal: 5.
2026-06-07 01:38:38,762 [INFO] [/infer] response: {"action": "CHAT", "reasoning": "prioritized_over_vad_scores_and_camera_observation_priority_shaped_tone", "message_to_user": "Hmm...? I didn't think about dinner until after 2pm."}
"""

"""
(base) brandonpratamakwee@Brandons-Air liquidAI % /opt/anaconda3/envs/liquid_env/bin/python3 model_server.py 
Using device: mps
Inference device: cpu
Loading brain model (text)...
Traceback (most recent call last):
  File "/Users/brandonpratamakwee/Desktop/liquidAI/model_server.py", line 187, in <module>
    brain_processor = AutoProcessor.from_pretrained(
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/envs/liquid_env/lib/python3.12/site-packages/transformers/models/auto/processing_auto.py", line 441, in from_pretrained
    return processor_class.from_pretrained(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/envs/liquid_env/lib/python3.12/site-packages/transformers/processing_utils.py", line 1691, in from_pretrained
    args = cls._get_arguments_from_pretrained(pretrained_model_name_or_path, processor_dict, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/envs/liquid_env/lib/python3.12/site-packages/transformers/processing_utils.py", line 1820, in _get_arguments_from_pretrained
    sub_processor = auto_processor_class.from_pretrained(
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/envs/liquid_env/lib/python3.12/site-packages/transformers/models/auto/image_processing_auto.py", line 575, in from_pretrained
    raise initial_exception
  File "/opt/anaconda3/envs/liquid_env/lib/python3.12/site-packages/transformers/models/auto/image_processing_auto.py", line 562, in from_pretrained
    config_dict, _ = ImageProcessingMixin.get_image_processor_dict(
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/envs/liquid_env/lib/python3.12/site-packages/transformers/image_processing_base.py", line 334, in get_image_processor_dict
    raise OSError(
OSError: Can't load image processor for '/Users/brandonpratamakwee/Desktop/liquidAI/models/lfm2_brain_merged'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '/Users/brandonpratamakwee/Desktop/liquidAI/models/lfm2_brain_merged' is the correct path to a directory containing a preprocessor_config.json file
(base) brandonpratamakwee@Brandons-Air liquidAI % 
"""


"""
(base) brandonpratamakwee@Brandons-Air liquidAI % /opt/anaconda3/envs/liquid_env/bin/python3 model_server.py
Using device: mps
Inference device: cpu
Loading brain model (text)...
Loading weights: 100%|██████████████████████████████████████████████████████████████████████| 589/589 [00:06<00:00, 92.08it/s]
Brain model loaded ✓
Loading vision model...
Loading weights: 100%|██████████████████████████████████████████████████████████████████████| 589/589 [00:07<00:00, 76.52it/s]
Vision model loaded ✓
Moving models to CPU for stable inference...
Models on CPU ✓
2026-06-07 00:28:29,368 [INFO] Starting uvicorn (attempt 1/10)...
Using device: mps
Inference device: cpu
Loading brain model (text)...
Loading weights: 100%|██████████████████████████████████████████████████████████████████████| 589/589 [00:06<00:00, 96.61it/s]
Brain model loaded ✓
Loading vision model...
Loading weights: 100%|██████████████████████████████████████████████████████████████████████| 589/589 [00:08<00:00, 67.00it/s]
Vision model loaded ✓
Moving models to CPU for stable inference...
Models on CPU ✓
INFO:     Started server process [5119]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     10.42.0.1:41892 - "GET /health HTTP/1.1" 200 OK
2026-06-07 00:30:22,077 [INFO] → /infer user_prompt[:120]: User said: 'Hello, I'm very happy-'. Visual observation from camera: 'Neutral'. Emotion scores — valence: 5.0/10, arousa
2026-06-07 00:30:52,110 [INFO] → /infer user_prompt[:120]: User said: 'It's a day.'. Visual observation from camera: 'Neutral'. Emotion scores — valence: 5.0/10, arousal: 5.0/10, 
[transformers] Ignoring clean_up_tokenization_spaces=True for BPE tokenizer TokenizersBackend. The clean_up_tokenization post-processing step is designed for WordPiece tokenizers and is destructive for BPE (it strips spaces before punctuation). Set clean_up_tokenization_spaces=False to suppress this warning, or set clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output=True to force cleanup anyway.
2026-06-07 00:30:53,721 [INFO] [/infer] response: {"action": "chat", "reasoning": "the image shows neutral expression but high vad, low arousal.", "message_to_user": "hey there! great to see you. how are you doing today?"}
2026-06-07 00:31:22,634 [INFO] [/infer] response: {"action": "CHAT", "reasoning": "visual observation prioritized over vad scores when conflicting.", "message_to_user": "hey there! great to see you. how are you doing today?"}
2026-06-07 00:31:28,050 [INFO] → /infer user_prompt[:120]: User said: 'Because the robot works.'. Visual observation from camera: 'Neutral'. Emotion scores — valence: 5.0/10, arou
2026-06-07 00:31:56,294 [INFO] [/infer] response: {"action": "CHAT", "reasoning": "camera visuals prioritize neutrality above scoring", "message_to_user": "hey"}
INFO:     10.42.0.1:48482 - "POST /infer HTTP/1.1" 200 OK
2026-06-07 00:31:56,302 [INFO] → /infer user_prompt[:120]: User said: 'Yeah.'. Visual observation from camera: 'Neutral'. Emotion scores — valence: 5.0/10, arousal: 5.0/10, domina
2026-06-07 00:32:26,316 [INFO] → /infer user_prompt[:120]: User said: 'You'. Visual observation from camera: 'Neutral'. Emotion scores — valence: 5.0/10, arousal: 5.0/10, dominanc
2026-06-07 00:32:33,364 [INFO] [/infer] response: {"action": "CHAT", "reasoning": "visual observations take precedence over vad scores when conflicting.", "message_to_user": "neutral"}
2026-06-07 00:32:56,347 [INFO] → /infer user_prompt[:120]: User said: 'I like'. Visual observation from camera: 'Neutral'. Emotion scores — valence: 5.0/10, arousal: 5.0/10, domin
2026-06-07 00:33:04,040 [INFO] [/infer] response: {"action": "chat", "reasoning": "the camera notices neutral expression, calmness, no tension, so chat is appropriate.", "message_to_user": "hi"}
2026-06-07 00:33:26,388 [INFO] → /infer user_prompt[:120]: User said: 'You'. Visual observation from camera: 'Neutral'. Emotion scores — valence: 5.0/10, arousal: 5.0/10, dominanc
2026-06-07 00:33:34,379 [INFO] [/infer] response: {"action": "CHAT", "reasoning": "visual observations take precedence when conflicting with vad values.", "message_to_user": "hey there! great to see you. how are you doing today?"}
2026-06-07 00:33:55,647 [INFO] [/infer] response: {"action": "chat"}
INFO:     10.42.0.1:36944 - "POST /infer HTTP/1.1" 200 OK

"""
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
BRAIN_MODEL_PATH  = os.path.abspath("./models/lfm2_brain_merged")
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
# MTLBuffer allocation that kills the process.
#
# Round-tripping the model MPS→CPU→MPS between requests caused a
# second crash: internal KV-cache buffers created during one generate()
# call stayed on MPS, so the next call found a device mismatch.
#
# Simplest stable solution: load onto MPS for fast weight loading,
# then move permanently to CPU before the first request.  All
# generate() calls run on CPU where the kernels are fully stable.
# For short conversational replies on Apple Silicon M-series this is
# ~3-8 s per response, which is fine for a robot assistant.
INFER_DEVICE = "cpu" if DEVICE == "mps" else DEVICE
print(f"Inference device: {INFER_DEVICE}")


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
    "\nCRITICAL Rules for message_to_user:\n"
    "- NEVER repeat or echo back the user's words. DO NOT quote what they said\n"
    "- ALWAYS generate a NEWm original reply sentence that reacts to the meaning of what was said. \n"
    "- If the user greets you, greet them back with your own words.\n"
    "- If the user says they are happy, respond with enthusiasm and ask a follow-up question.\n"
    "- If the user says a short word like 'okay' or 'today', reply with a natural conversational continuation\n"
    "- The text is always spoken aloud by a text-to-speech system so write it as natural spoken sentences.\n"
    "- Always include a direct, complete spoken response - never leave it empty or as a placeholder\n" 
    "- Keep it concise: 1-3 sentences maximum. \n"
    "- Do not include JSON syntax, bullet points, or markdown inside message_to_user. \n"
    "\nExamples of CORRECT behaviour: \n"
    " User: 'Hello' -> message_to_user: 'Hey there! Great to see you. How are you doing today?' \n"
    " User: 'I am very happy today' -> message_to_user: 'That is wonderful to hear! What has got you smiling?'\n"
    "\nExamples of WRONG behaviour: \n"
    " User: 'Hello' -> message_to_user: 'hello'           <- Wrong this is an echo\n"
    " User: 'I am very happy today' -> message_to_user: 'i am very happy today'   <- Wrong this is an echo\n"
    " User: 'Okay' -> 'okay'   <- Wrong this is an echo\n"
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

# If MPS is the load device, move models permanently to CPU now.
# This must happen after both models are fully loaded so MPS fast
# weight loading is still used, but before any generate() call.
if INFER_DEVICE != DEVICE:
    print("Moving models to CPU for stable inference...")
    brain_model.to(INFER_DEVICE)
    vision_model.to(INFER_DEVICE)
    print("Models on CPU ✓")

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
        with torch.no_grad():
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
