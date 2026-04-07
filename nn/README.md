# Task 2 — Drone Localisation Neural Network

Prototype NN-based system that takes an FPV drone image, uses the bird's-eye
warp from Task 1 as an intermediary, and localises the drone on a 500 km²
satellite map (500 tiles of 1 km² each). Deployable on Raspberry Pi 5.

---

## System Overview

```
FPV drone image
      │
      ▼
 Task 1: C++ warp
      │
      ▼
800×800 bird's-eye patch (1 px = 1 m, drone at centre)
      │
      ├──────────────────────────────────────────┐
      ▼                                          │
 Stage 1: Coarse retrieval                       │
 Encoder (MobileNetV3-Small)                     │
      │                                          │
      ▼                                          │
 128-dim L2-norm embedding                       │
      │                                          │
      ▼                                          │
 FAISS flat-IP index query                       │
 (pre-encoded satellite patches)                 │
      │                                          │
      ▼                                          │
 Top-K candidate tiles + rough (x,y)             │
      │                                          │
      ▼                                          │
 Stage 2: Fine localisation ◄────────────────────┘
 Normalised cross-correlation (NCC)
      │
      ▼
 Sub-metre (x,y) offset within tile → GPS coordinates
```

---

## Architecture

### Encoder

| Component | Detail |
|-----------|--------|
| Backbone | MobileNetV3-Small (ImageNet pretrained) |
| Output of backbone | 576-dim feature map after adaptive avg pool |
| Projection head | Linear(576→256) → BN → ReLU → Linear(256→128) |
| Final normalisation | L2-norm → cosine sim == dot product |
| Input | 224×224 RGB, ImageNet mean/std |
| Parameters | ~2.5 M (backbone) + ~150 k (head) |

The same encoder weights are used for both the drone branch and the satellite
branch (Siamese / shared-weight architecture). This halves the number of
parameters and forces the model to learn a joint embedding space.

### Loss — InfoNCE (NT-Xent)

```
L = -1/N Σ_i log( exp(q_i·k_i / τ) / Σ_j exp(q_i·k_j / τ) )
```

- **τ (temperature)** = 0.07 (default). Lower → sharper distribution → harder task.
- **Positive pair** = (bird's-eye patch at GPS position X, satellite crop at X).
- **Negatives** = all other satellite crops in the batch (in-batch negatives).
- Loss is computed symmetrically (q→k and k→q) and averaged.

Larger batches provide more negatives and improve representation quality.
Recommended batch size ≥ 64 for real training.

### Augmentation

| Branch | Augmentation |
|--------|-------------|
| Drone (query) | RandomResizedCrop (scale 0.85–1.0), RandomAffine shear ±5° (pitch proxy), ColorJitter (brightness/contrast/saturation), RandomHorizontalFlip |
| Satellite (key) | RandomResizedCrop (scale 0.9–1.0), RandomRotation ±180°, mild ColorJitter |

---

## Offline Index Build

```bash
python inference.py build \
    --tiles-dir /data/satellite_tiles \
    --index-out index.faiss \
    --checkpoint checkpoints/final.pt
```

Each 1 km² tile (800×800 px) is sliced into overlapping 800×800 patches with
stride 200 px, encoded, and stored in a FAISS `IndexFlatIP` (exact inner
product). For 500 tiles the index contains roughly:

```
patches per tile ≈ ((800 - 800) / 200 + 1)² = 1 (full tile only, no sub-patches needed
                                                    since patch == tile size)
```

In practice tiles can be larger; use `--stride` to control density vs. index size.

**Estimated index size:** 500 tiles × 128 floats × 4 bytes ≈ 0.25 MB (trivially fits RPi5 RAM).
With sub-tile patches (e.g. 4×4 grid per tile) → ~4 MB, still fine.

---

## Online Inference (per frame)

1. Bird's-eye patch → encoder → 128-dim embedding (~30 ms on RPi5 INT8).
2. FAISS IP search → top-5 candidate tiles + patch offsets (~5 ms).
3. NCC (`cv2.matchTemplate`) within best tile → sub-metre (x, y) (~15 ms).
4. **Total target: ~65 ms @ 15 fps on RPi5.**

```bash
python inference.py query \
    --image bird_eye.png \
    --index index.faiss \
    --checkpoint checkpoints/final.pt
```

---

## RPi5 Deployment

```bash
# Export to ONNX
python inference.py export \
    --checkpoint checkpoints/final.pt \
    --out encoder.onnx

# Run with onnxruntime (pre-installed on RPi5 via pip)
pip install onnxruntime faiss-cpu opencv-python
```

The ONNX model is dynamically INT8-quantised; for best accuracy on RPi5 use
static quantisation with `onnxruntime.quantization` and a calibration dataset.

---

## Training

```bash
# Install dependencies
pip install -r requirements.txt

# Overfit sanity check (single synthetic pair, ~30 s)
python train.py --overfit

# Full synthetic training run
python train.py --epochs 20 --batch-size 32

# Resume from checkpoint
python train.py --checkpoint checkpoints/epoch_015.pt
```

---

## Performance Tradeoffs

| Dimension | Choice | Tradeoff |
|-----------|--------|----------|
| Backbone | MobileNetV3-Small | Fast (RPi5 friendly) but lower accuracy than ResNet/EfficientNet |
| Embedding dim | 128 | Small index, fast FAISS search; lower capacity than 256/512 |
| FAISS index | FlatIP (exact) | Accurate but O(N) scan; swap for IVF/HNSW if N > 100 k patches |
| NCC fine alignment | CPU, OpenCV | Simple, no GPU needed; struggles with rotated patches |
| Stage 2 | NCC | No learned parameters; replace with a learned matcher (e.g. LoFTR) for higher accuracy |

---

## Pros & Cons

**Pros**
- Lightweight: runs on RPi5 without GPU.
- No GPS required at inference time.
- Offline index build decoupled from online query.
- Two-stage design cleanly separates coarse (learning) and fine (geometry) problems.
- Contrastive loss doesn't require class labels — scales to large unlabelled datasets.

**Cons**
- Requires representative satellite + drone training pairs (data collection is hard).
- NCC stage fails under large rotation (drone yaw uncertainty); needs compass or
  multi-angle patch augmentation.
- Single-scale encoder may miss texture-poor areas (fields, open water).
- FAISS FlatIP is exact but O(N); large maps need IVF indexing.

---

## Alternatives Considered

| Alternative | Why not chosen |
|-------------|---------------|
| SuperGlue / LoFTR (keypoint matcher) | Too heavy for RPi5 |
| NetVLAD place recognition | Designed for street-level imagery, not top-down |
| Direct regression (CNN → lat/lon) | Requires huge labelled dataset; poor generalisation |
| Particle filter localisation | Complex to implement; needs motion model |
| EfficientNet backbone | Slightly heavier; MobileNetV3 better RPi5 latency |

---

## File Structure

```
nn/
├── models/
│   ├── __init__.py
│   └── encoder.py        # MobileNetV3 + 128-dim embedding head
├── data/
│   ├── __init__.py
│   └── dataset.py        # SyntheticPairDataset + RealPairDataset stub
├── losses/
│   ├── __init__.py
│   └── infonce.py        # InfoNCE contrastive loss
├── train.py              # Training loop + overfit test
├── inference.py          # FAISS index build + query + ONNX export
├── requirements.txt
└── README.md
```
