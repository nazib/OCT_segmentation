# syntax=docker/dockerfile:1
#
# Multi-stage build for the OCT segmentation inference server.
#
# Stage 1 ("builder") needs network access to resolve and download wheels.
# Stage 2 ("runtime") copies only the resolved dependencies, app code and
# model weights, so the final image never needs network access to start or
# to serve requests.

FROM python:3.10-slim AS builder

WORKDIR /build

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip

COPY requirements-serving.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --user --no-warn-script-location -r requirements-serving.txt

FROM python:3.10-slim AS runtime

# Non-root, unprivileged user the app runs as.
RUN groupadd --gid 10001 app && \
    useradd --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app

COPY --from=builder /root/.local /home/app/.local
COPY app.py batcher.py hubconf.py ./
COPY unet ./unet
COPY utils ./utils
COPY weights ./weights

RUN chown -R app:app /app /home/app/.local

USER app
ENV PATH=/home/app/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    MODEL_PATH=/app/weights/final_model.pth

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,os,sys; sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/healthz', timeout=2).status == 200 else 1)"

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
