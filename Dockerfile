# Stage 1: Builder — install dependencies
FROM python:3.11-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY agentic_chunker.py .
COPY llm_client.py .

RUN mkdir -p /app/logs

# ---

# Stage 2: Final — runtime
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code from builder
COPY --from=builder /app/app.py .
COPY --from=builder /app/agentic_chunker.py .
COPY --from=builder /app/llm_client.py .
COPY --from=builder /app/logs /app/logs

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup -m -d /home/appuser -s /sbin/nologin appuser
RUN chown -R appuser:appgroup /app /home/appuser && \
    chmod -R 755 /app && \
    chmod -R u+w /app/logs

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
