# FR3 ISAC pinned GPU environment (the project plan.md section 7).
#
# Targets:
#   base : Ubuntu 24.04 + Python venv + PyTorch(cu128) + core scientific stack
#   dev  : base + test/lint/dev tools (used for P0 development and CI)
#   run  : base + baked project package only (used for production jobs)
#
# The CUDA *driver*, libcuda and OptiX (needed by Sionna RT / Mitsuba) are
# injected from the host by the NVIDIA Container Toolkit, hence
# NVIDIA_DRIVER_CAPABILITIES=all. No kernel driver is installed in the image.
ARG BASE_IMAGE=nvidia/cuda:12.8.1-base-ubuntu24.04

FROM ${BASE_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

# llvm: CPU fallback backend of Dr.Jit (used only if no GPU is visible)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip python3-dev \
        libatomic1 llvm git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04 ships Python 3.12 (Sionna 2.2 needs >= 3.11)
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# PyTorch with CUDA 12.8 (H100 / sm_90). If the host driver is too old for
# CUDA 12.8, rebuild with --build-arg TORCH_INDEX=.../whl/cu126
ARG TORCH_VERSION=2.9.1
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch==${TORCH_VERSION} --index-url ${TORCH_INDEX}

# Core pinned scientific stack (Sionna 2.2.0 + data/provenance stack)
COPY docker/requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock

WORKDIR /workspace

# The bind-mounted workspace is owned by the host user; allow git to read it.
RUN git config --system --add safe.directory /workspace

# --------------------------------------------------------------------------- #
FROM base AS dev
ARG BUILD_GIT_SHA=unknown
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*
COPY docker/requirements-dev.lock /tmp/requirements-dev.lock
RUN pip install --no-cache-dir -r /tmp/requirements-dev.lock
COPY pyproject.toml /workspace/pyproject.toml
COPY src /workspace/src
RUN pip install --no-cache-dir -e /workspace
RUN echo "${BUILD_GIT_SHA}" > /etc/fr3isac_build_sha
LABEL org.fr3isac.stage="dev" org.fr3isac.git_sha="${BUILD_GIT_SHA}"
CMD ["fr3isac", "--help"]

# --------------------------------------------------------------------------- #
FROM base AS run
ARG BUILD_GIT_SHA=unknown
COPY pyproject.toml /workspace/pyproject.toml
COPY src /workspace/src
RUN pip install --no-cache-dir /workspace
RUN echo "${BUILD_GIT_SHA}" > /etc/fr3isac_build_sha
LABEL org.fr3isac.stage="run" org.fr3isac.git_sha="${BUILD_GIT_SHA}"
CMD ["fr3isac", "--help"]
