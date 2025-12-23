# syntax=docker/dockerfile:1

# ============================================
# NSE Stock Analyzer - Production Dockerfile
# ============================================

FROM python:3.12-slim AS base

# Prevent Python from buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ============================================
# Builder stage - install dependencies
# ============================================
FROM base AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# ============================================
# Production stage
# ============================================
FROM base AS production

WORKDIR /app

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Copy wheels from builder
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Copy application files
COPY Home.py .
COPY symbols.csv .
COPY README.md .

# Change ownership to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Streamlit configuration
ENV STREAMLIT_BROWSER_GATHERUSAGESTATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_ENABLECORS=true \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Launch the app
CMD ["streamlit", "run", "Home.py"]
