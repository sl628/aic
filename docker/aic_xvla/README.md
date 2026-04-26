# aic_xvla submission image

One-container submission for the AI for Industry Challenge wrapping the
fine-tuned X-VLA LoRA checkpoint validated at total score 63.527 locally.

## Architecture

Single OCI image, two processes inside:
- **inference server** (conda XVLA env, python 3.10 + torch + transformers + peft):
  loads the LoRA adapter on top of `2toINF/X-VLA-Pt`, exposes FastAPI on
  `localhost:8010` with `/health` and `/act`.
- **policy bridge** (pixi env, ROS Kilted): runs `ros2 run aic_model aic_model`
  with `policy=aic_xvla.ros.RunXVLA`, which HTTP-posts each observation to the
  local server and returns 7D pose targets.

The entrypoint backgrounds the server, waits for `/health`, then exec's the
ROS node — preserving the standard Zenoh wiring exactly as
`docker/aic_model/Dockerfile` does.

## Build prerequisites

1. Conda XVLA env on host (used only by `build_assets.sh` to populate the HF cache).
2. Local LoRA ckpt at `/home/yifeng/aic_xvla_overfit_abs/ckpt-3000` and sidecar.
3. X-VLA repo at `/home/yifeng/workspace/X-VLA` (any commit is fine, archived as-is).
4. Docker with nvidia-container-toolkit (`docker info | grep nvidia` should list a runtime).

## Build

```bash
cd /home/yifeng/workspace/aic/.claude/worktree-xvla-submission

# 1. Pre-fetch big artifacts (~3.5 GB) into docker/aic_xvla/{hf_cache,aic_xvla_ckpt,X-VLA-src.tar}
./docker/aic_xvla/build_assets.sh

# 2. Build the image
docker compose -f docker/docker-compose.yaml build model
```

Image size target: ~12 GB.

## Local verification (mandatory before push)

```bash
# Standard local eval — no ACL
docker compose -f docker/docker-compose.yaml up

# Watch for "All Trials Processed!" line in eval logs.
# Then save the score:
cp ~/aic_results/scoring.yaml /home/yifeng/aic_xvla_overfit_abs/scoring_dockerized.yaml
```

Pass criteria: `total: ` value within ±5 of 63.5.

To test with ACL on (matches portal environment), uncomment `AIC_ENABLE_ACL: true`
on both `eval` and `model` services in `docker-compose.yaml`, then re-run.

## Push

```bash
export AWS_PROFILE=cableholder
aws ecr get-login-password --region us-east-1 \
    | docker login --username AWS --password-stdin 973918476471.dkr.ecr.us-east-1.amazonaws.com

TAG="xvla-$(date +%Y%m%d)-1"
docker tag aic-xvla:v1 973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/cableholder:$TAG
docker push 973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/cableholder:$TAG
echo "URI for portal: 973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/cableholder:$TAG"
```

Then paste the URI into the submission portal under Qualification phase.

## Files

- `Dockerfile` — multi-layer build (ROS Kilted base + pixi + conda XVLA + assets)
- `xvla_requirements.txt` — pip deps for the conda XVLA env (cu121 wheels)
- `entrypoint.sh` — server background-launch + Zenoh wiring + aic_model exec
- `build_assets.sh` — pre-fetches HF cache, copies LoRA ckpt, archives X-VLA src
- `.gitignore` — excludes baked assets from git (kept only for docker context)
