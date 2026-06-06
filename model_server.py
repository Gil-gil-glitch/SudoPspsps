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

import io
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


# ─────────────────────────────────────────────
#  APP
# ─────────────────────────────────────────────
app = FastAPI(title="Sudo Model Server")


# ── Request schemas ───────────────────────────

class InferRequest(BaseModel):
    system_prompt: str
    user_prompt: str

class VisionInferRequest(BaseModel):
    system_prompt: str
    user_prompt: str
    full_image_b64: str
    crop_image_b64: str


# ── Endpoints ─────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.post("/infer")
def infer(req: InferRequest):
    messages = [
        {"role": "system", "content": req.system_prompt},
        {"role": "user",   "content": req.user_prompt},
    ]
    print(messages)
    text = brain_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = brain_processor.tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = brain_model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            repetition_penalty=1.1,
        )
    input_len = inputs["input_ids"].shape[1]
    raw = brain_processor.tokenizer.decode(
        outputs[0][input_len:], skip_special_tokens=True
    ).strip()
    print(f"[/infer] response: {raw}")
    return {"result": raw}



@app.post("/infer_vision")
def infer_vision(req: VisionInferRequest):
    full_img = PILImage.open(io.BytesIO(base64.b64decode(req.full_image_b64))).convert("RGB")
    crop_img = PILImage.open(io.BytesIO(base64.b64decode(req.crop_image_b64))).convert("RGB")

    messages = [
        {"role": "system", "content": req.system_prompt},
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
    inputs = vision_processor(
        images=[full_img, crop_img],
        text=[text],
        return_tensors="pt",
    ).to(DEVICE)
    with torch.no_grad():
        out = vision_model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
        )
    response = vision_processor.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()
    return {"result": response}


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
