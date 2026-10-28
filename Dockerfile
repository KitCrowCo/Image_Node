# Use the official AMD ROCm PyTorch image (6.2 is currently stable for Strix Point)
FROM rocm/pytorch:rocm7.2.4_ubuntu22.04_py3.10_pytorch_release_2.7.1
#FROM rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0
#FROM rocm/pytorch:latest # Careful any changes will trigger massive redownloads

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Update and install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install requirements
# We use the index-url matching the ROCm version in the image (rocm6.2)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi \
    uvicorn \
    transformers==4.51.3 \
    accelerate>=1.1.0 \
    gguf>=0.10.0 \
    peft>=0.11.0 \
    git+https://github.com/huggingface/diffusers.git@v0.39.0
#pinned diffuser to @v0.39.0 to prevent changes

# Copy your application code
COPY . .

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir sentencepiece protobuf

EXPOSE 8000

# The environment variable is set at runtime (in compose), but you can also set it here if you want it baked in.
ENV HSA_OVERRIDE_GFX_VERSION=11.0.0

CMD ["python3", "main.py"]
