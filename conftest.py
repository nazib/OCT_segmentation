"""Test-time environment setup, executed before `app` module import.

Creates a tiny, randomly-initialised U-Net checkpoint so the inference
server can start up in CI without a real trained model, and configures a
small inference size so CPU-only test runs stay fast.
"""
import os
import tempfile

import torch

from unet import UNet

os.environ.setdefault("N_CHANNELS", "3")
os.environ.setdefault("N_CLASSES", "2")
os.environ.setdefault("INFERENCE_SIZE", "32")
os.environ.setdefault("BATCH_MAX_SIZE", "4")
os.environ.setdefault("BATCH_MAX_WAIT_MS", "25")

if "MODEL_PATH" not in os.environ:
    _weights_dir = tempfile.mkdtemp(prefix="oct-test-weights-")
    _weights_path = os.path.join(_weights_dir, "model.pth")
    _model = UNet(
        n_channels=int(os.environ["N_CHANNELS"]),
        n_classes=int(os.environ["N_CLASSES"]),
    )
    torch.save(_model.state_dict(), _weights_path)
    os.environ["MODEL_PATH"] = _weights_path
