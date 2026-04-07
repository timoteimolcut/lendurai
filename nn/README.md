# Task 2 — Drone Localisation Neural Network

Prototype NN-based system that takes an FPV drone image, uses the bird's-eye
warp from Task 1 as an intermediary, and localises the drone on a satellite map.
Deployable on Raspberry Pi 5.

---

## Real Data (this repo)

| File | Size | Role |
|------|------|------|
| `images/src/processed/image_fpv_drone.png` | 490×216 px | Raw FPV camera frame |
| `images/src/processed/satellite_map.png` | 570×570 px | Satellite reference map |
| `images/dst/dst/birds_eye_map_*.png` | 800×800 px | Task 1 output — bird's-eye warp |

The bird's-eye images are 800×800 but only the top portion contains real ground
pixels — the rest is black. At 60 m / -25° pitch, approximately the top 412 rows
are non-black. `inference.py query` automatically crops the non-black bounding
box before encoding so the encoder sees only real ground texture.

The satellite map (570×570 px) is smaller than the default 800 px patch size
assumed in the original design. The updated sliding window automatically clamps
to `min(patch_size, H, W)`, so it still produces meaningful patches.

**Recommended pitch for best ground coverage:** ≥ -40° at ≥ 150 m altitude.
Shallower angles (e.g. -25° / 60 m) produce a narrow trapezoid with large black
borders, which reduces encoder quality.

---

## System Overview

```
FPV drone image  (490×216)
      │
      ▼
 Task 1: C++ geometric warp
      │
      ▼
800×800 bird's-eye patch — large black border outside warped ground region
      │
      ▼ (auto crop non-black → ~800×412 at -40°/150m)
      │
      ├──────────────────────────────────────────┐
      ▼                                          │
 Stage 1: Coarse retrieval                       │
 Encoder (MobileNetV3-Small, 224×224 input)      │
      │                                          │
      ▼                                          │
 128-dim L2-norm embedding                       │
      │                                          │
      ▼                                          │
 FAISS IndexFlatIP query                         │
 (pre-encoded 256×256 satellite patches)         │
      │                                          │
      ▼                                          │
 Top-K candidates (tile, patch offset)           │
      │                                          │
      ▼                                          │
 Stage 2: Fine localisation ◄────────────────────┘
 NCC (cv2.matchTemplate TM_CCOEFF_NORMED)
 query resized to fit satellite if needed
      │
      ▼
 Drone (x, y) in satellite pixel coordinates
```

---

## Architecture

### Encoder

| Component | Detail |
|-----------|--------|
| Backbone | MobileNetV3-Small (ImageNet pretrained) |
| Backbone output | 576-dim after adaptive avg pool |
| Projection head | Linear(576→256) → BN → ReLU → Linear(256→128) |
| Final normalisation | L2-norm — cosine sim == dot product in FAISS |
| Input size | 224×224 RGB, ImageNet mean/std |
| Parameters | ~2.5 M backbone + ~150 k head |

Same weights for both the drone branch and the satellite branch (Siamese
architecture). Forces both views into the same embedding space.

### Loss — InfoNCE (NT-Xent)

```
L = -1/N Σ_i log( exp(q_i·k_i / τ) / Σ_j exp(q_i·k_j / τ) )
```

- **τ** = 0.07 (default). Lower → sharper distribution → harder contrastive task.
- **Positive pair**: (bird's-eye patch at location X, satellite crop at X).
- **Negatives**: all other satellite crops in the same batch.
- Loss computed symmetrically (q→k and k→q) and averaged.
- Larger batches (≥ 64) improve representation quality by providing more negatives.

### Augmentation

| Branch | Transforms |
|--------|-----------|
| Drone (query) | RandomResizedCrop (0.85–1.0), RandomAffine shear ±5° (pitch proxy), ColorJitter, RandomHorizontalFlip |
| Satellite (key) | RandomResizedCrop (0.9–1.0), RandomRotation ±180°, mild ColorJitter |

---

## Offline Index Build

```bash
# Using the real satellite map (570×570):
python inference.py build \
    --tiles-dir /path/to/satellite/tiles/ \
    --index-out index.faiss \
    --patch-size 256 \
    --stride 100

# Example output for satellite_map.png (570×570):
# [1/1] satellite_map.png (570×570px, eff_patch=256px): 16 patches
# Index saved → index.faiss  (16 vectors, D=128)
```

> **Important:** put only satellite images in `--tiles-dir`. If FPV or bird's-eye
> images are in the same folder they will be indexed too and pollute the search.

The sliding window is adaptive:
- If the image is smaller than `--patch-size`, the effective patch is clamped to
  `min(patch_size, H, W)`, so at least one patch is always extracted.
- Stride is similarly clamped to never exceed the effective patch size.

**Index size estimate:**

| Map size | patch=256, stride=100 | patch=800 (full tile) |
|----------|-----------------------|----------------------|
| 570×570 (this repo) | 16 patches × 128×4 B = 8 KB | 1 patch |
| 1000×1000 km² (500 tiles) | ~500 k patches → ~250 MB | ~2 MB |

For > 100 k patches replace `IndexFlatIP` with `IndexIVFFlat` or `IndexHNSWFlat`.

---

## Online Inference (per frame)

```bash
python inference.py query \
    --image ../images/dst/dst/birds_eye_map_20260407_113340_150_-40_0_0_110_80.png \
    --index index.faiss \
    --tiles-dir ../images/src/processed/ \
    --top-k 5
```

**Steps:**

1. Load 800×800 bird's-eye PNG.
2. Auto-crop non-black bounding box (removes the empty black border from the warp).
3. Encode cropped patch → 128-dim embedding.
4. FAISS IP search → top-K candidate patches.
5. Load best candidate tile; resize query to fit within tile if needed.
6. NCC (`cv2.matchTemplate`) within ±100 px search window.
7. Report drone position in tile pixel coordinates.

**Measured latency on CPU (no trained weights, ImageNet init):**

| Step | Time |
|------|------|
| Encode (CPU, PyTorch FP32) | ~50 ms |
| FAISS search (16 vectors) | < 1 ms |
| NCC fine alignment | ~15 ms |
| **Total** | **~65–75 ms** |

On RPi5 with ONNX INT8 encoder the encode step targets ~30 ms.

> **Note on match scores:** with ImageNet-pretrained weights and no fine-tuning,
> cosine similarity scores will be low (< 0.1). This is expected — the encoder
> has not learned the drone↔satellite domain. Scores improve significantly after
> contrastive training on real paired data.

---

## Training

```bash
# Install dependencies
pip install -r requirements.txt

# Overfit sanity check (single synthetic pair — passes in ~4 steps on GPU)
python train.py --overfit

# Full synthetic training
python train.py --epochs 20 --batch-size 32

# Custom hyperparameters
python train.py --epochs 50 --batch-size 64 --lr 1e-4

# Resume from checkpoint
python train.py --checkpoint checkpoints/epoch_015.pt
```

Checkpoints are saved every 5 epochs to `checkpoints/epoch_XXX.pt` and at the
end as `checkpoints/final.pt`.

---

## RPi5 Deployment

```bash
# Export trained encoder to ONNX
python inference.py export \
    --checkpoint checkpoints/final.pt \
    --out encoder.onnx

# RPi5: install runtime dependencies
pip install onnxruntime faiss-cpu opencv-python
```

Dynamic INT8 quantisation is applied at export. For production accuracy use
static quantisation with `onnxruntime.quantization` and a calibration set of
real drone images.

---

## Performance Tradeoffs

| Dimension | Choice | Tradeoff |
|-----------|--------|----------|
| Backbone | MobileNetV3-Small | RPi5-friendly; lower accuracy than ResNet/EfficientNet |
| Embedding dim | 128 | Small index, fast search; lower capacity than 256/512 |
| Patch size | 256 px (default) | Good coverage on small maps; increase for larger tiles |
| FAISS index | FlatIP (exact) | Accurate but O(N); swap for IVF/HNSW above 100 k patches |
| NCC | CPU, OpenCV | No parameters, no GPU; fails under large yaw uncertainty |

---

## Pros & Cons

**Pros**
- No GPS required at inference.
- Lightweight: runs on RPi5 without GPU.
- Offline index build decoupled from online query.
- Contrastive loss doesn't require class labels — scales to large unlabelled datasets.
- Adaptive sliding window handles satellite maps of any size.
- Auto non-black crop improves encoder quality on shallow-pitch bird's-eye images.

**Cons**
- NCC fails under large drone yaw; needs compass or rotational augmentation.
- Single-scale encoder may miss texture-poor areas (open fields, water).
- No trained weights — scores are near-random until fine-tuned on paired data.
- FPV and satellite images must be kept in separate folders for index building.

---

## Alternatives Considered

| Alternative | Why not chosen |
|-------------|---------------|
| SuperGlue / LoFTR | Too heavy for RPi5 |
| NetVLAD | Designed for street-level, not top-down imagery |
| Direct regression (CNN → lat/lon) | Requires large labelled dataset; poor generalisation |
| Particle filter | Complex motion model needed |
| EfficientNet backbone | Slightly heavier; MobileNetV3 better on RPi5 |

---

## File Structure

```
nn/
├── models/
│   ├── __init__.py
│   └── encoder.py        # MobileNetV3-Small + 128-dim L2-norm head
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
