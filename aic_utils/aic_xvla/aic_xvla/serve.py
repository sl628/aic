"""Phase-aware HTTP inference server with multi-adapter routing.

Loads a shared base model + 4 phase-specific LoRA adapters + an image-based
phase classifier. On /act: classify phase → switch adapter → generate actions.

Run (inside X-VLA conda env):
    export XVLA_REPO=~/workspace/X-VLA
    PYTHONPATH=$XVLA_REPO CUDA_VISIBLE_DEVICES=0 \
    python -m aic_xvla.serve \
        --base-model 2toINF/X-VLA-Pt \
        --checkpoints /ckpt/phase_0,/ckpt/phase_1,/ckpt/phase_2,/ckpt/phase_3 \
        --classifier /home/yifeng/aic_xvla_data/phase_classifier_v2.pt
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import torchvision.transforms as T

from aic_xvla.eval import DEFAULT_INSTRUCTION, AICXVLAPolicy, _load_model, _state_to_proprio
from aic_xvla.phase_classifier_model import PhaseClassifierNet, PHASES, IMAGE_MEAN, IMAGE_STD

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

log = logging.getLogger("aic_xvla.serve")

_IMAGE_TF = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(IMAGE_MEAN, IMAGE_STD),
])


class ActRequest(BaseModel):
    state: list[float]  # 26D
    images: list[str]  # 3 base64-encoded JPEGs (left, center, right)
    instruction: str = DEFAULT_INSTRUCTION
    steps: int = 10


def _decode_image(b64: str) -> Image.Image:
    import base64
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _load_phase_classifier(path: str, device: str) -> PhaseClassifierNet:
    model = PhaseClassifierNet()
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def _load_multi_adapter_model(
    base_id: str,
    checkpoint_paths: list[str],
    device: str,
    action_encodings: list[str],
) -> tuple[torch.nn.Module, list[str]]:
    """Load base model + 4 LoRA adapters, return model + resolved action encodings."""
    base, processor = _load_model(base_id, device)  # base XVLA on device

    # Load first adapter the standard way.
    ckpt0 = checkpoint_paths[0]
    model = PeftModel.from_pretrained(base, ckpt0, adapter_name="phase_0")
    for i, ckpt in enumerate(checkpoint_paths[1:], 1):
        model.load_adapter(ckpt, adapter_name=f"phase_{i}")
    model.to(device).eval()
    return model, action_encodings


def build_app(
    model: torch.nn.Module | PeftModel,
    processor,
    classifier: PhaseClassifierNet,
    action_encodings: list[str],
    device: str,
    default_steps: int = 10,
) -> FastAPI:
    app = FastAPI(title="aic-xvla-phase")

    @app.get("/healthz")
    @app.get("/health")
    def healthz():
        return {"ok": True, "device": device}

    @app.post("/act")
    def act(req: ActRequest):
        try:
            if len(req.state) != 26:
                return JSONResponse({"error": "state must be 26D"}, status_code=400)
            if len(req.images) != 3:
                return JSONResponse({"error": "need 3 images"}, status_code=400)

            state = np.asarray(req.state, dtype=np.float64)
            proprio = _state_to_proprio(state)

            # Decode images and run classifier.
            views = []
            for b64 in req.images:
                pil = _decode_image(b64)
                views.append(_IMAGE_TF(pil))
            img_tensor = torch.stack(views, dim=0).unsqueeze(0).to(device)  # (1, V, 3, H, W)

            with torch.no_grad():
                logits = classifier(img_tensor)
                phase_idx = logits.argmax(dim=1).item()
            phase_name = PHASES[phase_idx]

            log.info("classified phase=%d (%s)", phase_idx, phase_name)

            # Switch to the corresponding adapter.
            if hasattr(model, "set_adapter"):
                model.set_adapter(f"phase_{phase_idx}")

            # Build the action encoding from the resolved value for this checkpoint.
            enc = action_encodings[phase_idx] if phase_idx < len(action_encodings) else "delta"

            # Use the base XVLA model's generate_actions.
            image_input = img_tensor
            image_mask = torch.ones(1, 3, dtype=torch.bool, device=device)
            proprio_t = torch.from_numpy(proprio).unsqueeze(0).to(device)  # (1, 10)
            lang = processor.encode_language([req.instruction])
            lang = {k: v.to(device) for k, v in lang.items()}

            base_model = model.get_base_model() if hasattr(model, "get_base_model") else model

            with torch.no_grad():
                actions_20d = base_model.generate_actions(
                    image_input=image_input,
                    image_mask=image_mask,
                    proprio=proprio_t,
                    domain_id=torch.tensor([0], device=device),
                    steps=req.steps,
                    **lang,
                )

            act20 = actions_20d[0].cpu().float().numpy()
            actions_7d = AICXVLAPolicy._xvla_to_aic_actions(
                act20, proprio[:3], enc
            )

            return {
                "actions": actions_7d.tolist(),
                "horizon": int(actions_7d.shape[0]),
                "phase": phase_name,
                "phase_idx": phase_idx,
            }

        except Exception as e:
            log.exception("act failed")
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/classify")
    def classify(req: ActRequest):
        """Standalone phase classification (no action generation)."""
        try:
            views = []
            for b64 in req.images:
                pil = _decode_image(b64)
                views.append(_IMAGE_TF(pil))
            img_tensor = torch.stack(views, dim=0).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = classifier(img_tensor)
                phase_idx = logits.argmax(dim=1).item()
            return {"phase": PHASES[phase_idx], "phase_idx": phase_idx}
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    return app


def _resolve_action_encoding(checkpoint: str) -> str:
    """Read aic_xvla_meta.json sidecar for action_encoding; fall back to 'delta'."""
    for d in (checkpoint, os.path.dirname(checkpoint.rstrip("/"))):
        sidecar = os.path.join(d, "aic_xvla_meta.json")
        if os.path.isfile(sidecar):
            with open(sidecar) as f:
                enc = json.load(f).get("action_encoding")
            if enc:
                return enc.lower()
    return "delta"


def main() -> None:
    p = argparse.ArgumentParser(description="Phase-aware aic-xvla HTTP server")
    p.add_argument("--base-model", default="2toINF/X-VLA-Pt")
    p.add_argument(
        "--checkpoints",
        required=True,
        help="Comma-separated paths to 4 phase-specific LoRA checkpoints",
    )
    p.add_argument("--classifier", required=True, help="Path to phase_classifier_v2.pt")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8010)
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=10)
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    ckpt_paths = [p.strip() for p in args.checkpoints.split(",")]
    if len(ckpt_paths) != 4:
        raise SystemExit(f"Exactly 4 checkpoints required (phases 0-3), got {len(ckpt_paths)}")

    log.info("Loading phase classifier from %s ...", args.classifier)
    classifier = _load_phase_classifier(args.classifier, device)

    # Resolve action encoding for each checkpoint.
    encodings = [_resolve_action_encoding(c) for c in ckpt_paths]
    log.info("Action encodings: %s", encodings)

    log.info("Loading multi-adapter model (base=%s) ...", args.base_model)
    model, processor = _load_multi_adapter_model(args.base_model, ckpt_paths, device, encodings)
    log.info("Model loaded on %s", device)

    app = build_app(model, processor, classifier, encodings, device, args.steps)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
