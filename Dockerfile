# ---- Stage 1: install dependencies and train the model --------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

COPY src/ src/
# The model is trained once, at build time, with a fixed seed: every container
# started from this image serves exactly the same model artefact.
RUN PYTHONPATH=/install/lib/python3.11/site-packages:src \
    python -m medcalib_api.model > artifacts-report.json \
 && mkdir -p /app/artifacts_out && mv artifacts/model.joblib artifacts-report.json /app/artifacts_out/

# ---- Stage 2: minimal runtime image ----------------------------------------
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="medcalib-api" \
      org.opencontainers.image.description="Calibrated early-Alzheimer's risk estimation (research demo)" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MEDCALIB_MODEL_PATH=/app/artifacts/model.joblib

WORKDIR /app
COPY --from=builder /install /usr/local
COPY src/ src/
COPY --from=builder /app/artifacts_out/ artifacts/

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)"

CMD ["uvicorn", "medcalib_api.app:app", "--host", "0.0.0.0", "--port", "8000"]
