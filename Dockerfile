# PathogenIQ — Biosurveillance Intelligence Platform
# Build:          docker build -t pathogeniq .
# Run dashboard:  docker run -p 8765:8765 -v ./reports:/app/reports pathogeniq
# Run watcher:    docker run -v ./data:/data -v ./reports:/app/reports pathogeniq watch /data
# Full stack:     docker compose up

FROM python:3.12-slim

LABEL org.opencontainers.image.title="PathogenIQ"
LABEL org.opencontainers.image.description="AI-powered pathogen biosurveillance"
LABEL org.opencontainers.image.version="0.1.0"

# System build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (cached layer)
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Note: graph-tool (SBM community detection) cannot be installed via pip.
# Without it, PathogenIQ skips SBM and uses degree-based community assignment.
# To enable SBM: use the conda-based image or mount graph-tool from host.
# See: https://graph-tool.skewed.de/installation

# Copy source and install package
COPY pathogeniq/ ./pathogeniq/
COPY cli.py run.py temporal_replay.py download_datasets.sh ./
COPY configs/ ./configs/

RUN pip install --no-cache-dir -e . --no-deps

# Non-root runtime user
RUN useradd -m -u 1000 pathogeniq \
    && mkdir -p /home/pathogeniq/.pathogeniq /app/reports \
    && chown -R pathogeniq:pathogeniq /app /home/pathogeniq

USER pathogeniq

# Persist SQLite history DB and reports across restarts
VOLUME ["/home/pathogeniq/.pathogeniq", "/app/reports"]

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -sf http://localhost:8765/api/summary || exit 1

CMD ["pathogeniq", "dashboard", "--host", "0.0.0.0", "--port", "8765"]
