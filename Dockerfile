# MDITRE Docker Image
# Multi-stage build for Python MDITRE with optional R support
# Based on Ubuntu 24.04 LTS with CUDA support

FROM nvidia/cuda:12.4.0-runtime-ubuntu24.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY Python/requirements.txt /workspace/requirements.txt
COPY Python/requirements-dev.txt /workspace/requirements-dev.txt

# Install Python dependencies
RUN pip3 install --upgrade pip setuptools wheel && \
    pip3 install -r requirements.txt && \
    pip3 install -r requirements-dev.txt

# Copy Python package
COPY Python/ /workspace/Python/

# Install MDITRE package in development mode
RUN cd /workspace/Python && pip3 install -e .

# Set Python path
ENV PYTHONPATH=/workspace/Python:$PYTHONPATH

# Default command
CMD ["/bin/bash"]


# ============================================================================
# Stage 2: R Support (optional)
# ============================================================================
FROM base AS with-r

# Install R and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base=4.5.2-1.2404.0 \
    r-base-dev=4.5.2-1.2404.0 \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    libgit2-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy R installation script
COPY R/install_dependencies.R /workspace/R/install_dependencies.R
COPY R/ /workspace/R/

# Install R packages
RUN Rscript /workspace/R/install_dependencies.R

# Set R library path
ENV R_LIBS_USER=/usr/local/lib/R/site-library

CMD ["/bin/bash"]
