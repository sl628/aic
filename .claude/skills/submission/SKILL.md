---
name: submission
description: Use this skill when the user mentions submitting to the AI for Industry Challenge (AIC), pushing a Docker image to ECR, registering on the submission portal, the team CableHolder, or asks "how do I submit my model". Captures the validated end-to-end workflow for `sl628/aic` plus 7 known failure modes encountered on 2026-04-26 during the first X-VLA submission.
---

# AIC submission workflow (CableHolder team)

Submitting a policy to the AI for Industry Challenge: build a Docker image with the policy + weights baked in, verify locally that it scores like the dev run, push to AWS ECR, register the image URI on the portal. Strict 1-per-day limit and immutable tags — local verification is mandatory.

## Canonical references in this repo

- `docs/submission.md` — official build → push → register flow
- `docs/custom_dockerfile.md` — required Zenoh wiring + env vars (read this if you build a non-aic_model Dockerfile)
- `docs/access_control.md` — Zenoh ACL details; portal runs with auth on
- `docker/aic_xvla/README.md` — X-VLA-specific submission image (validated 63.5 baseline)
- `docker/aic_model/Dockerfile` — official example dockerfile to base custom ones on
- `docker/docker-compose.yaml` — local eval orchestration (eval + model containers)

## Hard constraints (from `docs/submission.md`)

- **1 submission per team per day.** Failed submissions count. Do NOT push without local verification.
- **ECR tags are immutable.** Use `xvla-$(date +%Y%m%d)-N` format and bump N for retries.
- **Image must use `rmw_zenoh_cpp`** and connect to the Zenoh router given by `AIC_MODEL_ROUTER_ADDR` (portal) or `AIC_ROUTER_ADDR` (local compose).
- **Auth**: user=`model`, password=`AIC_MODEL_PASSWD`. Write `model:$AIC_MODEL_PASSWD` to `/credentials.txt` and set `transport/auth/usrpwd/{user,password,dictionary_file}` in `ZENOH_CONFIG_OVERRIDE`.
- **Network is locked at eval.** All weights/models must be baked into the image. Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` in entrypoint.
- **No additional CLI args provided to entrypoint** at portal — supply defaults via `CMD`.

## Project-specific facts

| field | value |
| :--- | :--- |
| Team name | CableHolder |
| Team slug | cableholder |
| AWS profile to use | `cableholder` (set up via `aws configure --profile cableholder`) |
| AWS region | `us-east-1` |
| ECR registry | `973918476471.dkr.ecr.us-east-1.amazonaws.com` |
| ECR repo URI prefix | `973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/cableholder` |
| AWS Secret Access Key | NOT in this skill or any file. Lives in `~/.aws/credentials`. |
| X-VLA validated baseline score | 63.527 (3 trials, pose mode + replan=15, ckpt-3000) |

For credential/auth details and rotation policy, see memory file `~/.claude/projects/-home-yifeng-workspace-aic/memory/reference_aic_submission.md`.

## Canonical 8-step workflow (X-VLA submission)

Run from worktree root (currently `/home/yifeng/workspace/aic/.claude/worktree-xvla-submission`):

```bash
# 1. Pre-fetch baked assets (HF cache, LoRA ckpt, X-VLA src). Defaults to host
#    XVLA python at ~/miniconda3/envs/XVLA/bin/python. ~5 min if HF cache cached.
./docker/aic_xvla/build_assets.sh

# 2. Sanity: AWS profile loads + ECR repo is accessible.
export AWS_PROFILE=cableholder
aws sts get-caller-identity
aws ecr describe-repositories --region us-east-1 --repository-names aic-team/cableholder

# 3. Build the model image. ~30 min first time; ~5 min on rebuild unless apt
#    layer changed. After apt-layer changes use `build --no-cache model`.
docker compose -f docker/docker-compose.yaml build model 2>&1 | tee /tmp/build.log

# 4. Local verify, ACL OFF (~12 min). Brings up BOTH containers (eval + model).
#    Watch model logs for "X-VLA server ready", then eval logs for
#    "All Trials Processed!" with Total Score within ±5 of 63.5.
docker compose -f docker/docker-compose.yaml up
cp ~/aic_results/scoring.yaml /home/yifeng/aic_xvla_overfit_abs/scoring_dockerized.yaml
docker compose -f docker/docker-compose.yaml down

# 5. Local verify, ACL ON (~12 min). MOST IMPORTANT TEST — portal runs with ACL.
#    Uncomment `AIC_ENABLE_ACL: true` on BOTH eval and model services in compose.
#    Re-run, score should still be ~63. Revert compose after.

# 6. (Optional) Test the AIC_MODEL_ROUTER_ADDR fallback path:
#    edit compose model.environment to use AIC_MODEL_ROUTER_ADDR instead of
#    AIC_ROUTER_ADDR; rerun to prove portal's env-var name path works.

# 7. Push to ECR (~10 min upload). DO NOT proceed unless steps 4 and 5 PASSED.
aws ecr get-login-password --region us-east-1 \
    | docker login --username AWS --password-stdin 973918476471.dkr.ecr.us-east-1.amazonaws.com
TAG="xvla-$(date +%Y%m%d)-1"
docker tag aic-xvla:v1 973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/cableholder:$TAG
docker push 973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/cableholder:$TAG
echo "URI for portal: 973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/cableholder:$TAG"

# 8. Manual: portal → AI for Industry Challenge → Submit → Qualification phase
#    → paste URI → Submit. Monitor on My Submissions page. ~5–15 min to Finished.
```

## Pre-push checklist

Block on ALL of these:

- [ ] `~/aic_results/scoring.yaml` total within ±5 of validated baseline (63.5 for X-VLA)
- [ ] ACL-on local run also passed (step 5)
- [ ] `docker/aic_xvla/{hf_cache,aic_xvla_ckpt,X-VLA-src.tar}` exist (step 1 ran)
- [ ] `aws sts get-caller-identity` returns CableHolder identity (step 2 ran)
- [ ] `docker image inspect aic-xvla:v1 --format '{{.Size}}'` size noted (no documented portal limit; X-VLA image is ~22 GB and was accepted)
- [ ] No secrets baked into image (`docker run --rm aic-xvla:v1 env | grep -i aws` is empty)

## Known failure modes (2026-04-26 debugging session)

If you see these errors during build or `up`, here is what they mean:

| symptom | root cause | fix |
| :--- | :--- | :--- |
| `pixi install --locked` → `lock-file not up-to-date with the workspace` | `pixi.toml`/lock drift on a transitive ROS pkg (e.g. `ros-kilted-aic-teleoperation`) | Use `pixi install --frozen` instead — matches CI (`.github/workflows/pixi.yml:37`). NEVER edit `pixi.lock` by hand. |
| `conda create` → `To accept these channels' Terms of Service, run...` | Recent miniconda needs ToS for default Anaconda channels | Use `conda create -c conda-forge --override-channels` |
| `transformers 4.51.3 depends on huggingface-hub<1.0 and >=0.30.0` | Pinned `huggingface_hub` too low | Pin all requirements from the host's working XVLA env: `pip list --format=freeze` and copy versions verbatim |
| `ModuleNotFoundError: No module named 'pyarrow'` (or h5py, json_numpy) | X-VLA's `datasets/utils.py` imports these but they're not in our requirements | Add `pyarrow`, `h5py`, `json_numpy` to `xvla_requirements.txt` |
| `ImportError: libGL.so.1: cannot open shared object file` | `mmengine`/`timm` transitively pull in `opencv-python` (non-headless), shadowing `opencv-python-headless`. Non-headless opencv needs system libGL. | apt install `libgl1 libglib2.0-0` in the Dockerfile |
| Dockerfile changes don't take effect (apt layer caches stale) | BuildKit cached the apt layer despite Dockerfile edit | `docker compose build --no-cache model` (forces full rebuild ~30 min) |
| `eval` container logs `No node with name 'aic_model' found. Retrying...` indefinitely | `model` container's entrypoint crashed before the ROS node started | Check `docker compose logs model` for the actual stack trace; common causes are the ones above. |
| Build context error `aic_xvla_ckpt: not found` or `X-VLA-src.tar: not found` | Forgot to run `build_assets.sh` first | Run `./docker/aic_xvla/build_assets.sh`; assets land in `docker/aic_xvla/` (gitignored). |

## DO NOT

- DO NOT push to ECR if local score isn't within ±5 of baseline.
- DO NOT modify `pixi.lock` by hand. Pixi resolver only.
- DO NOT bake AWS credentials into the image. Image runs at portal — secrets would be exfiltrated.
- DO NOT skip the ACL-on local test (step 5). Portal runs with ACL; if it fails locally with ACL, it WILL fail at portal.
- DO NOT use the same ECR tag twice (immutable). Bump the suffix.
- DO NOT push to upstream `intrinsic-dev/aic`. We push to `sl628/aic` only.
- DO NOT post the AWS Secret Access Key in chat or commit it to any file. It lives only in `~/.aws/credentials`.

## When this skill applies

Trigger phrases that should make you load this skill:

- "submit to aic", "submit to the challenge", "submit my model"
- "ECR push", "ecr login", "docker login ECR"
- "leaderboard", "qualification phase", "submission portal"
- "team CableHolder", "cableholder slug"
- "what's wrong with my docker compose up" / "model container won't start" (X-VLA-specific)
- "how do I run the eval locally"

When applied, default to:
1. Confirming where the user is in the 8-step workflow.
2. Surfacing the relevant failure-mode row from the table if they pasted an error.
3. Reminding about the daily limit + ACL local test before any `docker push`.
