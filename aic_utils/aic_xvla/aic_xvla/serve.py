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

from aic_xvla.eval import DEFAULT_INSTRUCTION, AICXVLAPolicy, _load_model
from aic_xvla.handler import _state_to_proprio
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
    return model, processor, action_encodings


def build_app(
    model: torch.nn.Module | PeftModel,
    processor,
    classifier: PhaseClassifierNet,
    action_encodings: list[str],
    device: str,
    default_steps: int = 10,
    phase_hysteresis: int = 5,
    phase_confidence: float = 0.0,
) -> FastAPI:
    app = FastAPI(title="aic-xvla-phase")

    # Phase switching state: monotonic + hysteresis.
    # Only increases (0→1→2→3). Requires N consecutive preds of higher phase.
    _state = {"current_phase": 0, "consecutive": 0}

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
            # Pad 10D proprio to 20D (ee6d action space expects 20D).
            if proprio.shape[-1] < 20:
                proprio = np.concatenate(
                    [proprio, np.zeros(20 - proprio.shape[-1], dtype=np.float32)]
                )

            # Decode images and run classifier.
            views = []
            for b64 in req.images:
                pil = _decode_image(b64)
                views.append(_IMAGE_TF(pil))
            img_tensor = torch.stack(views, dim=0).unsqueeze(0).to(device)  # (1, V, 3, H, W)

            with torch.no_grad():
                logits = classifier(img_tensor)
                probs = torch.softmax(logits, dim=1)[0]  # [P0, P1, P2, P3]

            # Phase switching: monotonic + normalized confidence + progressive threshold.
            #
            # Only consider phases > current (never go back). Zero out passed
            # phases and renormalize, so the model just chooses among remaining
            # phases. Confidence threshold drops for later transitions (where
            # visual differences are subtler).
            s = _state
            cur = s["current_phase"]
            remaining_probs = probs.clone()
            remaining_probs[:cur + 1] = 0.0
            remaining_sum = remaining_probs.sum().item()

            next_prob = 0.0
            # Require at least 1% remaining probability to avoid amplifying noise.
            if remaining_sum > 0.01 and cur < 3:
                normalized = remaining_probs / remaining_sum
                next_phase = normalized.argmax().item()
                next_prob = normalized[next_phase].item()

                # Progressive threshold: easier to transition as we go.
                # P0→P1 needs 0.5, P1→P2 needs 0.4, P2→P3 needs 0.3.
                thresholds = [0.5, 0.4, 0.3]
                conf_thresh = thresholds[cur] if cur < len(thresholds) else phase_confidence

                if next_prob >= conf_thresh:
                    s["consecutive"] += 1
                    if s["consecutive"] >= phase_hysteresis:
                        s["current_phase"] = next_phase
                        s["consecutive"] = 0
                        log.info("SWITCH phase → %d (%s) norm_prob=%.3f thresh=%.2f",
                                 next_phase, PHASES[next_phase], next_prob, conf_thresh)
                else:
                    s["consecutive"] = 0
            else:
                s["consecutive"] = 0

            phase_idx = s["current_phase"]
            phase_name = PHASES[phase_idx]

            log.info("phase=%d (%s) raw=[P0=%.2f P1=%.2f P2=%.2f P3=%.2f] norm_next=%.2f consec=%d",
                     phase_idx, phase_name,
                     probs[0].item(), probs[1].item(), probs[2].item(), probs[3].item(),
                     next_prob if remaining_sum > 0 else 0.0,
                     s["consecutive"])

            # Switch to the corresponding adapter (only if changed).
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

            with torch.no_grad():
                actions_20d = model.generate_actions(
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
    p.add_argument(
        "--action-encoding", default=None,
        help="Override action encoding for all checkpoints (delta|absolute). "
             "If unset, auto-detect from aic_xvla_meta.json sidecar.",
    )
    p.add_argument("--phase-hysteresis", type=int, default=5,
        help="Min consecutive predictions of a higher phase before switching")
    p.add_argument("--phase-confidence", type=float, default=0.0,
        help="Min softmax probability to consider a phase switch (0=argmax)")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    ckpt_paths = [p.strip() for p in args.checkpoints.split(",")]
    if len(ckpt_paths) != 4:
        raise SystemExit(f"Exactly 4 checkpoints required (phases 0-3), got {len(ckpt_paths)}")

    log.info("Loading phase classifier from %s ...", args.classifier)
    classifier = _load_phase_classifier(args.classifier, device)

    # Resolve action encoding for each checkpoint.
    if args.action_encoding:
        encodings = [args.action_encoding] * 4
    else:
        encodings = [_resolve_action_encoding(c) for c in ckpt_paths]
    log.info("Action encodings: %s", encodings)

    log.info("Loading multi-adapter model (base=%s) ...", args.base_model)
    model, processor, encodings = _load_multi_adapter_model(args.base_model, ckpt_paths, device, encodings)
    log.info("Model loaded on %s", device)

    app = build_app(model, processor, classifier, encodings, device, args.steps,
                    phase_hysteresis=args.phase_hysteresis,
                    phase_confidence=args.phase_confidence)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
