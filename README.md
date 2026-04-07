# LendurAI — GPS-Free FPV Drone Localisation

Two-stage system that localises a drone on a 500 km² satellite map using only the onboard
camera — no GPS required. Designed to run on a Raspberry Pi 5.

```
FPV image  →  [Task 1: C++ geometric warp]  →  800×800 bird's-eye patch
                                                        │
                                    ┌───────────────────┴────────────────────┐
                                    ▼                                        │
                         Stage 1: FAISS coarse retrieval                     │
                         (MobileNetV3 encoder + embedding index)             │
                                    │                                        │
                                    ▼                                        │
                         Stage 2: NCC fine localisation ◄────────────────────┘
                                    │
                                    ▼
                         Drone (x, y) in metres on the satellite map
```

---

## Repository Structure

```
lendurai/
├── cv/                     # Task 1 — C++ bird's-eye warp
│   ├── main.cpp            # Production warp pipeline
│   ├── CMakeLists.txt
│   └── old/                # Archived prototypes (manual calib, trackbars, pinhole ref)
├── nn/                     # Task 2 — PyTorch localisation system
│   ├── models/
│   │   └── encoder.py      # MobileNetV3-Small + 128-dim embedding head
│   ├── data/
│   │   └── dataset.py      # SyntheticPairDataset + RealPairDataset stub
│   ├── losses/
│   │   └── infonce.py      # InfoNCE contrastive loss
│   ├── train.py            # Training loop + overfit sanity test
│   ├── inference.py        # FAISS index build / query / ONNX export
│   ├── requirements.txt
│   └── README.md           # Detailed NN architecture docs
├── scripts/
│   └── sweep.sh            # Batch parameter sweep for the C++ binary
├── images/
│   ├── src/processed/      # Input FPV images
│   └── dst/sweep/          # Bird's-eye warp outputs
└── .gitignore
```

---

## Task 1 — C++ Bird's-Eye Warp

### What it does

Takes an FPV drone image and reprojects it to a top-down 800×800 px map at
**1 px = 1 m**, with the drone at the centre (400, 400). Accepts known flight
parameters or can auto-estimate pitch from the image horizon.

### Build

```bash
cd cv
mkdir build && cd build
cmake ..
make -j$(nproc)
```

Requires: CMake ≥ 3.16, OpenCV, C++17 compiler.

### Usage

```bash
./birdseye <alt_m> <pitch_deg|AUTO> <roll_deg> <yaw_deg> <hfov_deg> <vfov_deg>
```

| Argument | Description |
|----------|-------------|
| `alt_m` | Flight altitude in metres (must be > 0) |
| `pitch_deg` | Camera pitch — negative = nose down. Pass `AUTO` to estimate from image |
| `roll_deg` | Roll in degrees — positive = right wing down |
| `yaw_deg` | Yaw in degrees — positive = nose right |
| `hfov_deg` | Horizontal field of view (0–180°) |
| `vfov_deg` | Vertical field of view (0–180°) |

```bash
# Fixed pitch
./birdseye 60 -25 0 0 110 80

# Auto-estimate pitch from horizon
./birdseye 60 AUTO 0 0 110 80

# High altitude, steep pitch
./birdseye 200 -70 0 0 110 80
```

Output is saved to `images/dst/sweep/birds_eye_map_<timestamp>_<params>.png` and
displayed in an OpenCV window.

> **Note:** Input image path is currently hardcoded to
> `images/src/processed/image_fpv_drone.png`. Change the `IMAGE_PATH` define in
> `main.cpp` or pass it as an argument (future improvement).

### Geometry engine

Camera convention: optical axis = +Z, image right = +X, image down = +Y.
World frame: forward = +Y, right = +X, up = +Z.

```
Camera ray  →  [pitch → roll → yaw rotation]  →  World ray
                                                        │
                                             t = -altitude / ray_z
                                                        │
                                                   Ground hit (x, y)
```

Rays at or above the horizon (`ray_z > -1e-6`) are clamped to prevent the
homography from blowing up. The image is cropped to the ground strip before
warping.

### Parameter sweep

```bash
cd scripts
./sweep.sh      # requires ../cv/build/birdseye to exist
```

Runs the binary over a grid of altitudes (`200, 300, 400 m`) and pitches
(`-60` to `-85°`) and saves all outputs to `images/dst/sweep/`.

---

## Task 2 — NN Localisation System

### Architecture overview

Two stages:

**Stage 1 — Coarse retrieval** finds which satellite tile(s) the drone is
flying over.

**Stage 2 — Fine localisation** finds the exact sub-metre position within the
best tile.

```
Bird's-eye patch (800×800)
        │
        ▼
  Encoder (MobileNetV3-Small)
  Shared weights — Siamese
  576 → 256 → 128 dim, L2-norm
        │
        ▼
  FAISS IndexFlatIP query
  (pre-encoded satellite patches)
        │
        ▼
  Top-K candidates (tile, patch offset)
        │
        ▼
  NCC template matching within best tile
  cv2.matchTemplate TM_CCOEFF_NORMED
  ±200 px search window, grayscale
        │
        ▼
  Drone (x, y) in tile pixels = metres
```

### Real data in this repo

| File | Size | Role |
|------|------|------|
| `images/src/processed/image_fpv_drone.png` | 490×216 px | Raw FPV input |
| `images/src/processed/satellite_map.png` | 570×570 px | Reference satellite map |
| `images/dst/dst/birds_eye_map_*.png` | 800×800 px | Task 1 output (bird's-eye warp) |

The bird's-eye output is 800×800 but contains a large black border — at 60 m /
-25° pitch, only the top ~412 rows have real ground pixels. The query pipeline
auto-crops this black border before encoding. For better ground coverage use
steeper pitch angles (≥ -40° at ≥ 150 m).

The satellite map (570×570) is smaller than the 800 px patch size originally
assumed. The updated sliding window automatically adapts: `eff_patch =
min(patch_size, H, W)`, so at least one patch is always extracted regardless of
map size.

> Keep only satellite images in `--tiles-dir` when building the index.
> If FPV or bird's-eye images share the same folder they will be indexed too.

### Setup

```bash
cd nn
pip install -r requirements.txt
# or via conda:
conda install -c conda-forge opencv
conda install -c pytorch faiss-cpu
```

### Training

```bash
# Overfit sanity check (single synthetic pair — passes in ~4 steps on GPU)
python train.py --overfit

# Full training on synthetic data
python train.py --epochs 20 --batch-size 32

# Custom hyperparameters
python train.py --epochs 50 --batch-size 64 --lr 1e-4 --temperature 0.1

# Resume from checkpoint
python train.py --checkpoint checkpoints/epoch_015.pt

# Force device
python train.py --device cuda
```

Checkpoints are saved every 5 epochs to `checkpoints/epoch_XXX.pt` and at the
end as `checkpoints/final.pt`.

**Loss — InfoNCE (symmetric NT-Xent):**

```
L = -1/N Σ log( exp(q·k / τ) / Σ exp(q·k_neg / τ) )
```

Positive pairs: `(bird's-eye patch at location X, satellite crop at location X)`.
Negatives: all other crops in the batch. Temperature τ = 0.07 (default).

**Augmentation:**

| Branch | Transforms |
|--------|-----------|
| Drone (query) | RandomResizedCrop (0.85–1.0), RandomAffine shear ±5° (pitch proxy), ColorJitter, RandomHorizontalFlip |
| Satellite (key) | RandomResizedCrop (0.9–1.0), RandomRotation ±180°, mild ColorJitter |

### Inference

#### 1. Build the FAISS index (offline, once per map update)

```bash
# With the real satellite_map.png (570×570):
python inference.py build \
    --tiles-dir ../images/src/processed/ \
    --index-out index.faiss \
    --patch-size 256 --stride 100

# Output:
# [1/1] satellite_map.png (570×570px, eff_patch=256px): 16 patches
# Index saved → index.faiss  (16 vectors, D=128)
```

The `--patch-size` is automatically clamped to the image dimensions, so this
works for satellite maps of any size. Use `--patch-size 800 --stride 200` for
full 1 km² tiles.

#### 2. Query (online, per frame)

```bash
python inference.py query \
    --image ../images/dst/dst/birds_eye_map_20260407_113340_150_-40_0_0_110_80.png \
    --index index.faiss \
    --tiles-dir ../images/src/processed/ \
    --top-k 5
```

Omit `--tiles-dir` to skip NCC and get coarse results only.

**Real output (ImageNet weights, no fine-tuning):**
```
Non-black crop: 800×800 → 800×412 px

Query: birds_eye_map_...  (coarse: 57.1 ms)

Rank    Score  Tile             Patch offset (x,y) px
1      0.0593  satellite_map    (100, 200)
...

Fine localisation (NCC, 15.6 ms):
  Tile       : satellite_map
  Match TL   : (0, 277) px in tile
  Drone pos  : (400, 483) px from tile origin
  NCC score  : 0.3469
  Total time : 72.7 ms
```

Cosine scores are low (< 0.1) because the encoder has not been fine-tuned on
drone/satellite imagery. They improve significantly with contrastive training.

#### 3. Export to ONNX (RPi5 deployment)

```bash
python inference.py export \
    --checkpoint checkpoints/final.pt \
    --out encoder.onnx
```

The resulting `encoder.onnx` runs on `onnxruntime` (pre-installable on RPi5
via `pip install onnxruntime`). Dynamic INT8 quantisation is applied.

**Measured latency (CPU, PyTorch FP32, no fine-tuning):** ~73 ms total.
**Target latency on RPi5 (ONNX INT8):**

| Stage | Time |
|-------|------|
| Encoder (ONNX INT8) | ~30 ms |
| FAISS search | < 1 ms |
| NCC fine alignment | ~15 ms |
| **Total** | **~50–65 ms** |

---

## Dependencies

### C++
- CMake ≥ 3.16
- OpenCV (any recent version)
- C++17

### Python
| Package | Purpose |
|---------|---------|
| `torch` + `torchvision` | Model training and inference |
| `faiss-cpu` | Approximate nearest-neighbour retrieval |
| `opencv-python` | Image I/O, NCC template matching |
| `onnx` + `onnxruntime` | Export and deploy on RPi5 |
| `numpy`, `Pillow` | Array ops, image transforms |

---

## Known Limitations

- **Image path hardcoded** — `IMAGE_PATH` in `main.cpp` must be updated manually.
- **No real training data** — `SyntheticPairDataset` generates random pairs.
  Localisation quality will be limited until real drone + satellite pairs are
  collected. See `nn/data/dataset.py` (`RealPairDataset`) for the expected data
  layout.
- **NCC fails under large yaw** — if the drone heading differs significantly
  from north, the NCC search window may miss the correct position. A compass
  reading would constrain the search.
- **Single-scale encoder** — may struggle in texture-poor areas (open water,
  bare fields).
