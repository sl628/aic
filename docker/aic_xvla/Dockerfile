# aic_xvla submission image: ROS Kilted policy bridge + X-VLA inference server
# baked into one container. Two processes share localhost:8010 over HTTP.
#
# Build context must be the repo root (see docker-compose.yaml).
# Multi-stage: `base` stage is stable, `final` adds bulky baked assets.
# Build with: docker build -t aic-xvla:v1 --target final .

FROM docker.io/library/ros:kilted-ros-core AS base

# OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates bzip2 build-essential \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --- pixi ---
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:$PATH"

# --- miniconda ---
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh \
    && bash /tmp/mc.sh -b -p /opt/conda \
    && rm /tmp/mc.sh
ENV PATH="/opt/conda/bin:$PATH"

# --- aic ROS workspace ---
COPY aic_example_policies /ws_aic/src/aic/aic_example_policies
COPY aic_model /ws_aic/src/aic/aic_model
COPY aic_interfaces /ws_aic/src/aic/aic_interfaces
COPY aic_utils /ws_aic/src/aic/aic_utils
COPY pixi.toml pixi.lock /ws_aic/src/aic/
COPY pixi_env_setup.sh /ws_aic/src/aic/

SHELL ["/bin/bash", "-c"]
RUN --mount=type=cache,target=/root/.cache/rattler/cache --mount=type=cache,target=/ws_aic/src/aic/.pixi/build \
    cd /ws_aic/src/aic && pixi install --frozen

# --- conda XVLA env ---
COPY docker/aic_xvla/xvla_requirements.txt /tmp/xvla_requirements.txt
RUN /opt/conda/bin/conda create -n XVLA -c conda-forge --override-channels python=3.10 -y \
    && /opt/conda/envs/XVLA/bin/pip install --no-cache-dir -r /tmp/xvla_requirements.txt

# --- X-VLA upstream repo ---
COPY docker/aic_xvla/X-VLA-src.tar /tmp/X-VLA-src.tar
RUN mkdir -p /opt/X-VLA && tar -xf /tmp/X-VLA-src.tar -C /opt/X-VLA && rm /tmp/X-VLA-src.tar

# ===== FINAL STAGE =====
FROM base AS final

# --- baked-in HF cache (3.3 GB) ---
COPY docker/aic_xvla/hf_cache /opt/hf_cache

# --- baked-in phase LoRA adapters + phase classifier ---
COPY docker/aic_xvla/aic_xvla_ckpt /opt/aic_xvla_ckpt

WORKDIR /ws_aic/src/aic

COPY --chmod=755 docker/aic_xvla/entrypoint.sh /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--ros-args", "-p", "policy:=aic_xvla.ros.RunXVLA", "-p", "use_sim_time:=true"]
