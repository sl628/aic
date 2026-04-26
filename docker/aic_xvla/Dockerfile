# aic_xvla submission image: ROS Kilted policy bridge + X-VLA inference server
# baked into one container. Two processes share localhost:8010 over HTTP.
#
# Build context must be the repo root (see docker-compose.yaml).

FROM docker.io/library/ros:kilted-ros-core AS build

# OS deps for building/installing python + curl + cv2 runtime libs
# (libgl1, libglib2.0-0 needed because pypi opencv-python gets pulled in as
# a transitive of mmengine/timm and shadows opencv-python-headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates bzip2 build-essential \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --- pixi (ROS-side env, identical to docker/aic_model/Dockerfile) ---
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:$PATH"

# --- miniconda for X-VLA inference env ---
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh \
    && bash /tmp/mc.sh -b -p /opt/conda \
    && rm /tmp/mc.sh
ENV PATH="/opt/conda/bin:$PATH"

# --- aic ROS workspace (mirrors docker/aic_model/Dockerfile) ---
COPY aic_example_policies /ws_aic/src/aic/aic_example_policies
COPY aic_model /ws_aic/src/aic/aic_model
COPY aic_interfaces /ws_aic/src/aic/aic_interfaces
COPY aic_utils /ws_aic/src/aic/aic_utils
COPY pixi.toml pixi.lock /ws_aic/src/aic/
COPY pixi_env_setup.sh /ws_aic/src/aic/

SHELL ["/bin/bash", "-c"]
# Match CI (.github/workflows/pixi.yml uses frozen=true). The committed
# pixi.lock occasionally drifts from pixi.toml (transitive ros pkg metadata
# changes); --frozen installs straight from the lockfile without re-checking.
RUN --mount=type=cache,target=/root/.cache/rattler/cache --mount=type=cache,target=/ws_aic/src/aic/.pixi/build \
    cd /ws_aic/src/aic && pixi install --frozen

# --- conda XVLA env: torch + transformers + peft + fastapi + ... ---
# Use conda-forge only to avoid Anaconda default-channel ToS prompts.
COPY docker/aic_xvla/xvla_requirements.txt /tmp/xvla_requirements.txt
RUN /opt/conda/bin/conda create -n XVLA -c conda-forge --override-channels python=3.10 -y \
    && /opt/conda/envs/XVLA/bin/pip install --no-cache-dir -r /tmp/xvla_requirements.txt

# --- X-VLA upstream repo (pinned to commit 6bc2513 used to train ckpt-3000) ---
COPY docker/aic_xvla/X-VLA-src.tar /tmp/X-VLA-src.tar
RUN mkdir -p /opt/X-VLA && tar -xf /tmp/X-VLA-src.tar -C /opt/X-VLA && rm /tmp/X-VLA-src.tar

# --- baked-in HF cache for 2toINF/X-VLA-Pt (3.3 GB, no runtime download) ---
COPY docker/aic_xvla/hf_cache /opt/hf_cache

# --- baked-in LoRA adapter (the validated ckpt-3000) + sidecar ---
COPY docker/aic_xvla/aic_xvla_ckpt /opt/aic_xvla_ckpt

WORKDIR /ws_aic/src/aic

# entrypoint launches FastAPI server in background, then aic_model ROS node
COPY --chmod=755 docker/aic_xvla/entrypoint.sh /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--ros-args", "-p", "policy:=aic_xvla.ros.RunXVLA", "-p", "use_sim_time:=true"]
