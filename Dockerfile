# PathogenIQ — Biosurveillance Intelligence Platform
# Build: docker build -t pathogeniq .
# Run dashboard: docker run -p 8765:8765 -v ./reports:/app/reports pathogeniq
# Run watcher:   docker run -v ./data:/data -v ./reports:/app/reports pathogeniq watch /data

FROM python:3.12-slim

LABEL org.opencontainers.image.title="PathogenIQ"
LABEL org.opencontainers.image.description="AI-powered pathogen biosurveillance"
LABEL org.opencontainers.image.version="0.1.0"

# Install system build dependencies (needed for scipy/numpy compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer — only rebuilds when requirements change)
COPY requirements.txt pyproject.toml ./

# Install core deps (skip torch — add separately if VQ-VAE training needed)
RUN pip install --no-cache-dir \
        numpy scipy pandas scikit-learn biopython pyyaml click rich \
        networkx statsmodels requests tqdm fastapi "uvicorn[standard]"

# Copy source code
COPY pathogeniq/ ./pathogeniq/
COPY cli.py ./
COPY configs/ ./configs/

# Install the package in editable mode (registers `pathogeniq` CLI entry point)
RUN pip install --no-cache-dir -e . --no-deps

# Create non-root user for runtime security
RUN useradd -m -u 1000 pathogeniq \
    && mkdir -p /home/pathogeniq/.pathogeniq \
    && chown -R pathogeniq:pathogeniq /app /home/pathogeniq

USER pathogeniq

# Persist SQLite history DB and reports across container restarts
VOLUME ["/home/pathogeniq/.pathogeniq", "/app/reports"]

EXPOSE 8765

# Default: start the web dashboard
CMD ["pathogeniq", "dashboard", "--host", "0.0.0.0", "--port", "8765"]
