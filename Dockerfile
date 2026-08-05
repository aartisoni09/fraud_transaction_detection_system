# ── FraudGuard — Multi-target Dockerfile ──────────────────────
# Build targets:
#   docker build --target api -t fraudguard-api .
#   docker build --target ui  -t fraudguard-ui  .

# ── Base stage ─────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# ── API target ─────────────────────────────────────────────────
FROM base AS api

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

# ── UI target ──────────────────────────────────────────────────
FROM base AS ui

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true"]
