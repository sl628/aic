# aic_xvla phase-aware submission image

One-container submission wrapping the phase classifier + 4 phase-specific X-VLA
LoRA adapters. The inference server classifies the current phase (approach,
coarse-align, fine-align, insert) and routes to the corresponding adapter.

## Architecture

Single OCI image, two processes inside:
- **inference server** (conda XVLA env, python 3.10): loads the phase classifier
  (ResNet18) + 4 LoRA adapters. On `/act` runs classifier → selects adapter →
  generates actions. Exposes FastAPI on `localhost:8010`.
- **policy bridge** (pixi env, ROS Kilted): runs `ros2 run aic_model aic_model`
  with `policy=aic_xvla.ros.RunXVLA`, HTTP-posts observations to the server.

## Build

```bash
cd /home/yifeng/aic/.claude/worktrees/yf_phase-inference

# 1. Pre-fetch big artifacts (~3.5 GB) from Hugging Face
./docker/aic_xvla/build_assets.sh

# 2. Build the image
docker compose -f docker/docker-compose.yaml build model
```

## Local verification

```bash
docker compose -f docker/docker-compose.yaml up
cp ~/aic_results/scoring.yaml /tmp/scoring_phase.yaml
```

## Push

```bash
export AWS_PROFILE=cableholder
aws ecr get-login-password --region us-east-1 \
    | docker login --username AWS --password-stdin 973918476471.dkr.ecr.us-east-1.amazonaws.com
TAG="xvla-phase-$(date +%Y%m%d)-1"
docker tag aic-xvla:v1 973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/cableholder:$TAG
docker push 973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/cableholder:$TAG
```
