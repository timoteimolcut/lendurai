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

### Setup

```bash
cd nn
pip install -r requirements.txt
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
python inference.py build \
    --tiles-dir /data/satellite_tiles \
    --index-out index.faiss \
    --checkpoint checkpoints/final.pt
```

Slides an 800×800 window (stride 200 px) over every tile, encodes each patch,
and stores embeddings in a FAISS `IndexFlatIP`. Metadata (tile name, patch
offset) is saved alongside as `index.npz`.

#### 2. Query (online, per frame)

```bash
python inference.py query \
    --image bird_eye.png \
    --index index.faiss \
    --checkpoint checkpoints/final.pt \
    --tiles-dir /data/satellite_tiles \
    --top-k 5
```

Omit `--tiles-dir` to skip NCC and get coarse results only.

**Example output:**
```
Query: bird_eye.png  (coarse: 28.4 ms)

Rank   Score  Tile                            Patch offset (x,y) px = m
----------------------------------------------------------------------
1     0.8934  tile_042                        (200, 400)
2     0.8102  tile_041                        (400, 200)
...

Fine localisation (NCC, 14.1 ms):
  Tile      : tile_042
  Patch TL  : (213, 387) px
  Drone pos : (613, 787) px = (613 m, 787 m) from tile origin
  NCC score : 0.7821  (1.0 = perfect match)
  Total     : 42.5 ms
```

#### 3. Export to ONNX (RPi5 deployment)

```bash
python inference.py export \
    --checkpoint checkpoints/final.pt \
    --out encoder.onnx
```

The resulting `encoder.onnx` runs on `onnxruntime` (pre-installable on RPi5
via `pip install onnxruntime`). Dynamic INT8 quantisation is applied.

**Latency budget on RPi5 5:**

| Stage | Time |
|-------|------|
| Encoder (ONNX INT8) | ~30 ms |
| FAISS search | ~5 ms |
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
