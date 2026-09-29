"""OCT segmentation inference server.

Exposes a single `/segment` endpoint backed by a trained U-Net. Concurrent
requests are coalesced by an `AsyncBatcher` into a single model forward pass
(see batcher.py), plus Kubernetes-style liveness/readiness probes and
Prometheus metrics for observability.
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List, Tuple

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from batcher import AsyncBatcher
from unet import UNet
from utils.data_loading import BasicDataset
from utils.utils import save_image_overlay

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("oct_segmentation")

MODEL_PATH = os.getenv("MODEL_PATH", "weights/final_model.pth")
N_CHANNELS = int(os.getenv("N_CHANNELS", "3"))
N_CLASSES = int(os.getenv("N_CLASSES", "2"))
INFERENCE_SIZE = int(os.getenv("INFERENCE_SIZE", "256"))
BATCH_MAX_SIZE = int(os.getenv("BATCH_MAX_SIZE", "8"))
BATCH_MAX_WAIT_MS = float(os.getenv("BATCH_MAX_WAIT_MS", "10"))

REQUESTS_TOTAL = Counter(
    "segment_requests_total", "Total /segment requests by outcome", ["status"]
)
REQUEST_LATENCY = Histogram(
    "segment_request_latency_seconds", "End-to-end /segment request latency"
)
BATCH_SIZE = Histogram(
    "inference_batch_size",
    "Number of requests coalesced into a single forward pass",
    buckets=(1, 2, 4, 8, 16, 32, 64),
)
BATCH_LATENCY = Histogram(
    "inference_batch_latency_seconds", "Model forward pass latency per batch"
)
QUEUE_DEPTH = Gauge(
    "inference_queue_depth", "Requests currently waiting to be batched"
)

# Mutable server state, populated at startup by the lifespan handler below.
state = {
    "model": None,
    "device": torch.device("cpu"),
    "batcher": None,
    "ready": False,
}

PreprocessedItem = Tuple[torch.Tensor, Image.Image]


def _load_model() -> torch.nn.Module:
    model = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, bilinear=False)
    state_dict = torch.load(MODEL_PATH, map_location=state["device"])
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _preprocess(img: Image.Image) -> torch.Tensor:
    # Every image in a batch must share the same spatial dims to be stacked
    # into one tensor, so we resize to a fixed size and map the predicted
    # mask back onto the original resolution afterwards.
    resized = img.resize((INFERENCE_SIZE, INFERENCE_SIZE), resample=Image.BICUBIC)
    array = BasicDataset.preprocess(resized, scale=1.0, is_mask=False)
    return torch.from_numpy(array).float()


def _run_batch_sync(items: List[PreprocessedItem]) -> List[Image.Image]:
    """Runs one forward pass for a coalesced batch. Executed off the event loop."""
    tensors = torch.stack([tensor for tensor, _ in items]).to(state["device"])
    with torch.no_grad():
        logits = state["model"](tensors)
    masks = torch.argmax(logits, dim=1).cpu().numpy()

    outputs = []
    for (_, original_img), mask in zip(items, masks):
        mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(
            original_img.size, resample=Image.NEAREST
        )
        overlay = save_image_overlay(
            np.array(original_img.convert("RGB")), np.array(mask_img)
        )
        outputs.append(overlay)
    return outputs


async def _batch_fn(items: List[PreprocessedItem]) -> List[Image.Image]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _run_batch_sync, items)


def _on_batch(batch_size: int, latency_seconds: float) -> None:
    BATCH_SIZE.observe(batch_size)
    BATCH_LATENCY.observe(latency_seconds)


async def _report_queue_depth() -> None:
    while True:
        batcher = state["batcher"]
        if batcher is not None:
            QUEUE_DEPTH.set(batcher.queue_depth())
        await asyncio.sleep(0.5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    queue_depth_task = asyncio.create_task(_report_queue_depth())
    try:
        state["model"] = _load_model()
        state["batcher"] = AsyncBatcher(
            batch_fn=_batch_fn,
            max_batch_size=BATCH_MAX_SIZE,
            max_wait_seconds=BATCH_MAX_WAIT_MS / 1000,
            on_batch=_on_batch,
        )
        state["batcher"].start()
        state["ready"] = True
        logger.info("Model loaded from %s (inference size=%d, max batch=%d, max wait=%dms)",
                    MODEL_PATH, INFERENCE_SIZE, BATCH_MAX_SIZE, BATCH_MAX_WAIT_MS)
    except Exception:
        logger.exception("Failed to load model from %s", MODEL_PATH)
        state["ready"] = False

    yield

    queue_depth_task.cancel()
    if state["batcher"] is not None:
        await state["batcher"].stop()


app = FastAPI(title="OCT Segmentation Inference Server", lifespan=lifespan)


@app.get("/healthz")
async def liveness():
    """Liveness probe: process is up and serving HTTP. Never depends on the model."""
    return {"status": "alive"}


@app.get("/readyz")
async def readiness():
    """Readiness probe: model is loaded and the batcher is accepting work."""
    batcher = state["batcher"]
    if not state["ready"] or batcher is None or not batcher.running:
        raise HTTPException(status_code=503, detail="model not ready")
    return {"status": "ready"}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/segment")
async def segment(image: UploadFile = File(...)):
    request_start = time.monotonic()
    batcher = state["batcher"]
    if not state["ready"] or batcher is None:
        REQUESTS_TOTAL.labels(status="unavailable").inc()
        raise HTTPException(status_code=503, detail="model not ready")

    try:
        raw = await image.read()
        pil_img = Image.open(io.BytesIO(raw))
        pil_img.load()
    except Exception as exc:
        REQUESTS_TOTAL.labels(status="bad_request").inc()
        raise HTTPException(status_code=400, detail=f"invalid image: {exc}") from exc

    tensor = _preprocess(pil_img)
    try:
        overlay = await batcher.submit((tensor, pil_img))
    except Exception:
        logger.exception("Inference failed")
        REQUESTS_TOTAL.labels(status="error").inc()
        raise HTTPException(status_code=500, detail="inference failed")

    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    REQUESTS_TOTAL.labels(status="ok").inc()
    REQUEST_LATENCY.observe(time.monotonic() - request_start)

    return Response(content=buf.getvalue(), media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
