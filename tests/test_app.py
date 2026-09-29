import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import app


@pytest.fixture()
def client():
    with TestClient(app) as c:  # runs FastAPI lifespan startup/shutdown
        yield c


def _sample_image_bytes(size=(40, 30)):
    img = Image.new("RGB", size, color=(120, 80, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_liveness_does_not_require_model(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "alive"}


def test_readiness_once_model_loaded(client):
    resp = client.get("/readyz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready"}


def test_segment_returns_png_matching_input_size(client):
    resp = client.post(
        "/segment", files={"image": ("test.png", _sample_image_bytes(), "image/png")}
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"

    out = Image.open(io.BytesIO(resp.content))
    assert out.size == (40, 30)


def test_segment_rejects_invalid_image(client):
    resp = client.post(
        "/segment", files={"image": ("bad.png", io.BytesIO(b"not an image"), "image/png")}
    )
    assert resp.status_code == 400


def test_metrics_exposes_prometheus_series(client):
    client.post("/segment", files={"image": ("test.png", _sample_image_bytes(), "image/png")})

    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "segment_requests_total" in resp.text
    assert "inference_batch_size" in resp.text
    assert "inference_queue_depth" in resp.text
