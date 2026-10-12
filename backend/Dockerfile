FROM python:3.13.9-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PYTHONPATH=/app/backend
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.lock /app/backend/requirements.lock
# Install the CPU wheel first: the common lock accepts its 2.8.0+cpu version.
# ARM64 PyTorch wheels are already CPU-only and come from PyPI.
RUN if [ "$(uname -m)" = "x86_64" ]; then \
      pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0; \
    else pip install torch==2.8.0; fi \
    && pip install -r backend/requirements.lock \
    && pip check
COPY backend /app/backend
COPY data /app/data
COPY artifacts /app/artifacts
COPY scripts /app/scripts
RUN useradd --create-home --uid 10001 neurostock && mkdir -p /app/runtime && chown -R neurostock:neurostock /app
USER neurostock
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
