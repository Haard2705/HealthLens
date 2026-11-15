# Backend image. Build from the repo root: docker build -f backend/Dockerfile -t healthlens-api .
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY backend backend
COPY ml/src ml/src
EXPOSE 8000
# Set ENV=production, JWT_SECRET and DEMO_PASSWORD at runtime; the app refuses to start otherwise.
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
