"""HTTP inference server for a fine-tuned X-VLA checkpoint on aic.

Wraps `AICXVLAPolicy` with a FastAPI `/act` endpoint. The aic-specific
encoding (state[26]→proprio[10] rot6d, action 20D→7D quat with delta-pos
decode) lives here, so the ROS client only sees: 26D state in, [T, 7] in
(pos + quat_xyzw) out.

Run inside the X-VLA conda env, with X-VLA repo on PYTHONPATH:

    PYTHONPATH=~/workspace/X-VLA python -m aic_xvla.serve \\
        --checkpoint /home/yifeng/aic_xvla_overfit/ckpt-3000 \\
        --host 0.0.0.0 --port 8010
"""

from __future__ import annotations

import argparse
import io
import logging

import numpy as np
import uvicorn
from aic_xvla.eval import DEFAULT_INSTRUCTION, AICXVLAPolicy
from aic_xvla.handler import _state_to_proprio
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel


class ActRequest(BaseModel):
    state: list[float]  # 26D
    images: list[str]  # 3 base64-encoded JPEGs/PNGs (left, center, right)
    instruction: str = DEFAULT_INSTRUCTION
    steps: int = 10


def _decode_image(b64: str) -> Image.Image:
    import base64

    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def build_app(policy: AICXVLAPolicy) -> FastAPI:
    app = FastAPI(title="aic-xvla")
    log = logging.getLogger("aic_xvla.serve")

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "device": str(policy.device)}

    @app.post("/act")
    def act(req: ActRequest):
        try:
            if len(req.state) != 26:
                return JSONResponse(
                    {"error": f"state must be 26D, got {len(req.state)}"},
                    status_code=400,
                )
            if len(req.images) != 3:
                return JSONResponse(
                    {"error": f"need 3 images, got {len(req.images)}"}, status_code=400
                )
            state = np.asarray(req.state, dtype=np.float64)
            proprio = _state_to_proprio(state)
            images = [_decode_image(b) for b in req.images]
            actions = policy.predict(images, proprio, req.instruction)
            return {"actions": actions.tolist(), "horizon": int(actions.shape[0])}
        except Exception:
            log.exception("act failed")
            return JSONResponse(
                {"error": "act failed; see server log"}, status_code=500
            )

    return app


def main() -> None:
    p = argparse.ArgumentParser(description="aic-xvla HTTP inference server")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8010)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--steps", type=int, default=10, help="diffusion steps for generate_actions"
    )
    args = p.parse_args()

    policy = AICXVLAPolicy(args.checkpoint, device=args.device, steps=args.steps)
    print(
        f"loaded {args.checkpoint} on {args.device}; serving on {args.host}:{args.port}"
    )
    uvicorn.run(build_app(policy), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
