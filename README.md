
- [Quick start](#quick-start)
  - [Without Docker](#without-docker)
  - [With Docker](#with-docker)
  - [Training](#training)
  - [Prediction](#prediction)
  - [Inference Server](#inference-server)
- [Kubernetes](#kubernetes)
- [CI](#ci)
- [Weights \& Biases](#weights--biases)

## About

This project trains a [U-Net](https://arxiv.org/abs/1505.04597) model to segment retinal blood
vessels in optical coherence tomography (OCT) / fundus images, and serves the trained model
behind an async FastAPI inference server (`app.py`) so it can be integrated into other systems.
Given an input image, the server returns a PNG with the predicted vessel mask overlaid on the
original image. Concurrent requests are coalesced by an async micro-batcher (`batcher.py`) into
a single model forward pass, and the server exposes Kubernetes-style liveness/readiness probes
and Prometheus metrics.

## Quick start

### Without Docker

1. Model is trained and tested on CPU only machines. You can easily change the it to train on cuda/mps by changing --device argument

2. Install miniconda 

3. Create conda environment using environment.yaml file 
  ```
  conda env create -f /tmp/environment.yaml
  conda activate oct
  ```

4. Download the data from (https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction) 
```
python train.py --dataset-dir $PATH_TO_DATASET --load-dir $MODEL_NAME --classes $NUMBEROFCLASS --device $DEVICE
```

### With Docker

1. [Install Docker 19.03 or later:](https://docs.docker.com/get-docker/)
```bash
curl https://get.docker.com | sh && sudo systemctl --now enable docker
```
2. Before building, [download a trained model](https://drive.google.com/drive/u/0/folders/1i_kX8HDWj2sMAC6Lw03dLKc988R9jbC7)
   and put it at `weights/final_model.pth`. The build bakes the weights into the image so the
   container never needs network access at runtime.
3. Build the image:
   ```bash
   docker build -t oct-segmentation .
   ```
   This is a multi-stage build: dependencies are resolved in a `builder` stage (the only stage
   that needs network access), and the final `runtime` stage contains just the app, the model
   weights and a minimal Python runtime, running as a non-root user (uid `10001`).
4. Run it:
   ```bash
   docker run --rm -p 8080:8080 oct-segmentation
   ```
   The container starts the same FastAPI inference server described in
   [Inference Server](#inference-server) below, and needs no network access to run.

### Training

```console
> python train.py -h
usage: train.py [-h] [--dataset-dir][--load-dir] [--epochs E] [--batch-size B] [--learning-rate LR]
                [--load LOAD] [--scale SCALE] [--validation VAL] [--amp]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  --dataset-dir         dataset directory 
  --model-dir           model save directory
  --epochs E, -e E      Number of epochs
  --batch-size B, -b B  Batch size
  --learning-rate LR, -l LR
                        Learning rate
  --load LOAD, -f LOAD  Load model from a .pth file
  --scale SCALE, -s SCALE
                        Downscaling factor of the images
  --validation VAL, -v VAL
                        Percent of the data that is used as validation (0-100)
  --amp                 Use mixed precision
```

By default, the `scale` is 0.5, so if you wish to obtain better results (but use more memory), set it to 1.

Automatic mixed precision is also available with the `--amp` flag. [Mixed precision](https://arxiv.org/abs/1710.03740) allows the model to use less memory and to be faster on recent GPUs by using FP16 arithmetic. Enabling AMP is recommended.


### Prediction

After training your model and saving it to --model-dir, you can easily test the output masks on your images via the CLI.

To predict, give the images path,model path and the output path:

`python predict.py -i '/test' -o /output -m /model_dir/final_model.pth`

To predict.py will load all the images in the input directory and will save the prediction overlayed on the input images to the output path.
ß
```console
> python predict.py -h
usage: predict.py [-h] [--model FILE] --input INPUT [INPUT ...] 
                  [--output INPUT [INPUT ...]] [--viz] [--no-save]
                  [--mask-threshold MASK_THRESHOLD] [--scale SCALE]

Predict masks from input images

optional arguments:
  -h, --help            show this help message and exit
  --model FILE, -m FILE
                        Specify the file in which the model is stored
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        Filenames of input images
  --output INPUT [INPUT ...], -o INPUT [INPUT ...]
                        Filenames of output images
  --viz, -v             Visualize the images as they are processed
  --no-save, -n         Do not save the output masks
  --mask-threshold MASK_THRESHOLD, -t MASK_THRESHOLD
                        Minimum probability value to consider a mask pixel white
  --scale SCALE, -s SCALE
                        Scale factor for the input images
```
You can specify which model file to use with `--model MODEL.pth`.

### Inference Server

`app.py` serves the trained model over HTTP so it can be integrated into other systems,
instead of running predictions from the CLI. It's built on FastAPI + Uvicorn (not the old
Flask/gevent server) so it can batch concurrent requests asynchronously.

**Run without Docker:**

1. Install serving dependencies: `pip install -r requirements-serving.txt`
2. Place a trained model at `weights/final_model.pth` (see [Download Model](#with-docker) above).
3. Start the server:
   ```bash
   python app.py
   ```
   The server listens on port `8080` by default; set the `PORT` environment variable to
   change it. It runs on CPU.

**Run with Docker:** building and running the image (see [With Docker](#with-docker) above)
starts this same server automatically.

**Endpoints:**

| Method | Path       | Description                                                                  |
|--------|------------|-------------------------------------------------------------------------------|
| GET    | `/healthz` | Liveness probe — `200` once the process is up, independent of model state.    |
| GET    | `/readyz`  | Readiness probe — `200` once the model is loaded and the batcher is running, `503` otherwise. |
| GET    | `/metrics` | Prometheus metrics in text exposition format.                                 |
| POST   | `/segment` | Accepts an image, returns a PNG with the predicted mask overlaid on it.       |

**Example request:**

```bash
curl -X POST http://localhost:8080/segment \
  -F "image=@/path/to/input_image.tif" \
  -o segmentation.png
```

The response body is a `segmentation.png` image (`Content-Type: image/png`) containing the
input image with the predicted vessel mask drawn on top of it.

**Micro-batching:** concurrent `/segment` requests are coalesced by an async micro-batcher
(`batcher.py`) into a single model forward pass, instead of running one forward pass per
request. Tunable via environment variables:

| Variable            | Default                     | Meaning                                                              |
|----------------------|------------------------------|------------------------------------------------------------------------|
| `MODEL_PATH`          | `weights/final_model.pth`   | Path to the trained model checkpoint.                                 |
| `INFERENCE_SIZE`      | `256`                        | Images are resized to `INFERENCE_SIZE x INFERENCE_SIZE` before batching (so images of any size can be stacked into one tensor), then the mask is resized back to the original resolution. |
| `BATCH_MAX_SIZE`      | `8`                           | Maximum number of requests coalesced into one forward pass.           |
| `BATCH_MAX_WAIT_MS`   | `10`                          | Maximum time the batcher waits for more requests before running a partial batch. |

**Metrics** exposed at `/metrics` (Prometheus text format) include `segment_requests_total`
(by outcome), `segment_request_latency_seconds`, `inference_batch_size`,
`inference_batch_latency_seconds`, and `inference_queue_depth`.

## Kubernetes

Manifests live in [`k8s/`](k8s) and are managed with [Kustomize](https://kustomize.io/):

```bash
kubectl apply -k k8s
```

This deploys `oct-segmentation` as a `Deployment` (2 replicas, running as non-root uid `10001`
with a read-only root filesystem) fronted by a `ClusterIP` `Service`, with:

- **Liveness/readiness probes** wired to `/healthz` and `/readyz`, plus a `startupProbe` on
  `/readyz` that gives the model up to 150s to load before the readiness probe takes over.
- **Prometheus scraping** via `prometheus.io/*` annotations on the pod and service. If your
  cluster runs the [Prometheus Operator](https://prometheus-operator.dev/), uncomment
  `servicemonitor.yaml` in `k8s/kustomization.yaml` instead.

Update `image:` in `k8s/deployment.yaml` (or `k8s/kustomization.yaml`'s `images:` block) to
point at wherever you push the image built from the [Dockerfile](Dockerfile).

## CI

- **GitHub Actions** ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)): lints with
  `ruff`, runs the `pytest` suite, builds the Docker image, and validates the Kubernetes
  manifests against upstream schemas with `kubeconform`.
- **GitLab CI** ([`.gitlab-ci.yml`](.gitlab-ci.yml)): mirrors the same four stages
  (lint, test, build, validate-manifests).

## Weights & Biases

The training progress can be visualized in real-time using [Weights & Biases](https://wandb.ai/).  Loss curves, validation curves, weights and gradient histograms, as well as predicted masks are logged to the platform.

When launching a training, a link will be printed in the console. Click on it to go to your dashboard. If you have an existing W&B account, you can link it
 by setting the `WANDB_API_KEY` environment variable. If not, it will create an anonymous run which is automatically deleted after 7 days.

---
Original paper by Olaf Ronneberger, Philipp Fischer, Thomas Brox:

[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

![network architecture](https://i.imgur.com/jeDVpqF.png)
