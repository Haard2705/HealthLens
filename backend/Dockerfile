# HealthLens API image.
#
# Build from the REPO ROOT so the model artifacts and the ml package are in the context:
#     docker build -f backend/Dockerfile -t healthlens-api .
#
# The artifacts are gitignored, so build on a machine that has run the pipeline
# (`python -m ml.src.train`). A build without them still produces a working image: the
# prediction endpoints answer 503 and everything else keeps working.
#
# Run:
#     docker run -p 8000:8000 -e ENV=production \
#       -e JWT_SECRET="$(openssl rand -hex 32)" -e DEMO_PASSWORD=... healthlens-api
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Dependencies first: this layer is cached unless requirements.txt changes.
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# The API and the pipeline package it imports (registry, features, explain).
COPY backend backend
COPY ml/src ml/src

# Model artifacts and the cleaned survey file. .dockerignore drops the training-only
# model files, so this is ~85 MB rather than ~451 MB.
COPY ml/models ml/models
COPY data/processed data/processed

# Run as a non-root user.
RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

# The app refuses to start in production without a real JWT_SECRET and DEMO_PASSWORD.
# Hosts that inject a port use it; otherwise 8000.
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
