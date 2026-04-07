# LendurAI — GPS-Free FPV Drone Localisation

Two-stage system that localises a drone on a satellite map using only the onboard
camera — no GPS required. Deployable on Raspberry Pi 5.

```
FPV image  →  [Task 1: C++ geometric warp]  →  800×800 bird's-eye patch (1px=1m)
                                                        │
                                    ┌───────────────────┴───────────────────┐
                                    ▼                                       │
                         Stage 1: FAISS coarse retrieval                    │
                         MobileNetV3 encoder → 128-dim embedding            │
                                    │                                       │
                                    ▼                                       │
                         Stage 2: NCC fine localisation ◄───────────────────┘
                                    │
                                    ▼
                         Drone (x, y) in satellite pixel coordinates
                                    │
                                    ▼
                         Footprint overlay on satellite map  (visualize)
```

---

## Repository Structure

```
lendurai/
├── cv/                         # Task 1 — C++ bird's-eye warp
│   ├── main.cpp                # Production warp pipeline
│   ├── CMakeLists.txt
│   └── old/                    # Archived prototypes (manual calib, trackbars, pinhole ref)
├── nn/                         # Task 2 — PyTorch localisation system
│   ├── models/encoder.py       # MobileNetV3-Small + 128-dim L2-norm head
│   ├── data/dataset.py         # SyntheticPairDataset + RealPairDataset stub
│   ├── losses/infonce.py       # InfoNCE contrastive loss
│   ├── train.py                # Training loop + overfit sanity test
│   ├── inference.py            # FAISS index build / query / visualize / ONNX export
│   ├── requirements.txt
│   └── README.md               # Detailed NN architecture docs
├── scripts/
│   └── sweep.sh                # Batch parameter sweep for the C++ binary
├── images/
│   ├── src/processed/          # Input images (FPV + satellite)
│   └── dst/dst/                # Bird's-eye warp outputs
└── .gitignore
```

---

## Real Data in this Repo

| File | Size | Role |
|------|------|------|
| `images/src/processed/image_fpv_drone.png` | 490×216 px | Raw FPV camera frame |
| `images/src/processed/satellite_map.png` | 570×570 px | Satellite reference map |
| `images/dst/dst/birds_eye_map_*.png` | 800×800 px | Task 1 output (bird's-eye warp) |

The bird's-eye output is 800×800 but contains a large black border — only the
warped ground region has real pixels (trapezoid shape, roughly the top 400 rows).
All pipeline steps auto-crop this border before processing.

---

## Task 1 — C++ Bird's-Eye Warp

### What it does

Reprojects an FPV drone image to a top-down 800×800 canvas at **1 px = 1 m**,
drone at centre (400, 400). Accepts known flight parameters or can auto-estimate
pitch from the image horizon using Sobel edge energy.

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
./build/birdseye <alt_m> <pitch_deg|AUTO> <roll_deg> <yaw_deg> <hfov_deg> <vfov_deg>
```

| Argument | Description |
|----------|-------------|
| `alt_m` | Altitude in metres — must be > 0 |
| `pitch_deg` | Camera pitch, negative = nose down. Use `AUTO` to estimate from horizon |
| `roll_deg` | Roll in degrees, positive = right wing down |
| `yaw_deg` | Yaw in degrees, positive = nose right |
| `hfov_deg` | Horizontal FOV in degrees (0–180) |
| `vfov_deg` | Vertical FOV in degrees (0–180) |

```bash
# Good coverage — steep pitch, high altitude
./build/birdseye 150 -40 0 0 110 80

# Shallow pitch — large black border in output
./build/birdseye 60 -25 0 0 110 80

# Auto-estimate pitch from horizon
./build/birdseye 150 AUTO 0 0 110 80
```

Output is saved to `images/dst/sweep/birds_eye_map_<timestamp>_<params>.png`
and displayed in an OpenCV window.

> **Note:** Input path is hardcoded to `images/src/processed/image_fpv_drone.png`
> via `IMAGE_PATH` in `main.cpp`.

### Geometry engine

Camera frame: optical axis = +Z, right = +X, down = +Y.
World frame: forward = +Y, right = +X, up = +Z.
Rotation order: pitch → roll → yaw.

```
Pixel  →  normalised ray (camera)
       →  [pitch → roll → yaw]  →  world ray
       →  ground intersection: t = -altitude / ray_z
       →  ground point (x_m, y_m)
       →  canvas pixel (400 + x_m, 400 - y_m)
```

### Parameter sweep

```bash
cd scripts && ./sweep.sh   # requires cv/build/birdseye
```

Runs the binary over altitudes `[200, 300, 400 m]` × pitches `[-60° … -85°]`
and saves all 18 outputs to `images/dst/sweep/`.

---

## Task 2 — NN Localisation System

### Architecture

```
Bird's-eye patch (800×800, auto-cropped to non-black bbox)
        │
        ▼
  Encoder — MobileNetV3-Small (ImageNet pretrained, fine-tunable)
  Shared weights for drone and satellite branches (Siamese)
  576-dim backbone → Linear(256) → BN → ReLU → Linear(128) → L2-norm
        │
        ▼
  128-dim L2-normalised embedding
        │
        ▼
  FAISS IndexFlatIP — exact inner product (= cosine, embeddings are L2-normed)
  Offline: slide window over satellite tiles, encode each patch, store
  Online:  encode query → search top-K candidates
        │
        ▼
  NCC fine localisation — cv2.matchTemplate (TM_CCOEFF_NORMED, grayscale)
  Query resized to fit satellite; ±100 px search around coarse hit
        │
        ▼
  Drone (x, y) in satellite pixel coordinates
        │
        ▼
  visualize — footprint overlay on satellite map saved as PNG
```

### Setup

```bash
# Conda (recommended — matches the pytorch-cuda env used during development)
conda install -c conda-forge opencv
conda install -c pytorch faiss-cpu

# Or pip
cd nn && pip install -r requirements.txt
```

### Training

```bash
cd nn

# Overfit sanity check — single synthetic pair, proves gradients flow
# Expected: loss < 0.05 within ~4 steps on GPU
python train.py --overfit

# Full synthetic training (no real data needed)
python train.py --epochs 20 --batch-size 32

# Custom hyperparameters
python train.py --epochs 50 --batch-size 64 --lr 1e-4 --temperature 0.1

# Resume from checkpoint
python train.py --checkpoint checkpoints/epoch_015.pt
```

Checkpoints saved every 5 epochs → `checkpoints/epoch_XXX.pt`, and at end →
`checkpoints/final.pt`.

**Loss — InfoNCE (symmetric NT-Xent):**
```
L = -1/N Σ log( exp(q·k / τ) / Σ_j exp(q·k_j / τ) )
```
Positive pair: (bird's-eye patch at X, satellite crop at X).
Negatives: all other satellite crops in the batch. Temperature τ = 0.07.

### Inference

All inference commands run from `nn/`.

#### Step 1 — Build the FAISS index (offline, once per map)

```bash
python inference.py build \
    --tiles-dir ../images/src/processed/ \
    --index-out index.faiss \
    --patch-size 256 --stride 100
```

> Put **only satellite images** in `--tiles-dir`. Other images (FPV, bird's-eye)
> in the same folder will be indexed too and pollute results.

Output:
```
[1/1] satellite_map.png (570×570px, eff_patch=256px): 16 patches
Index saved → index.faiss  (16 vectors, D=128)
```

The patch size is clamped to `min(patch_size, H, W)` automatically, so this
works for satellite maps of any size.

#### Step 2 — Query (per frame, full two-stage pipeline)

```bash
python inference.py query \
    --image  ../images/dst/dst/birds_eye_map_20260407_113340_150_-40_0_0_110_80.png \
    --index  index.faiss \
    --tiles-dir ../images/src/processed/ \
    --top-k 5
```

Omit `--tiles-dir` to skip NCC and return coarse results only.

Output:
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
  NCC score  : 0.3469  (1.0 = perfect match)
  Total time : 72.7 ms
```

> Cosine scores < 0.1 are expected with ImageNet-only weights. They improve
> significantly after contrastive fine-tuning on real drone+satellite pairs.

#### Step 3 — Visualize the visible ground footprint

Highlights the exact ground region visible in the bird's-eye image directly on
the satellite map. Does **not** require a FAISS index or trained weights — runs
purely on geometry + NCC.

```bash
python inference.py visualize \
    --image     ../images/dst/dst/birds_eye_map_20260407_113340_150_-40_0_0_110_80.png \
    --satellite ../images/src/processed/satellite_map.png \
    --out       result.png
```

Output `result.png`:
- **Cyan fill** — ground region visible from the drone (trapezoid mask)
- **Yellow contour** — border of the visible footprint
- **Red dot + crosshair** — drone position projected onto the satellite
- **Legend** — NCC score and drone coordinates

#### Step 4 — Export to ONNX (RPi5 deployment)

```bash
python inference.py export \
    --checkpoint checkpoints/final.pt \
    --out encoder.onnx
```

Run on RPi5 with `onnxruntime`:
```bash
pip install onnxruntime faiss-cpu opencv-python
```

**Latency budget:**

| Stage | CPU (FP32) | RPi5 target (ONNX INT8) |
|-------|-----------|------------------------|
| Encode | ~57 ms | ~30 ms |
| FAISS search | < 1 ms | < 1 ms |
| NCC fine align | ~15 ms | ~15 ms |
| **Total** | **~73 ms** | **~50 ms** |

---

## Dependencies

### C++
- CMake ≥ 3.16, OpenCV, C++17 compiler

### Python

| Package | Purpose |
|---------|---------|
| `torch` + `torchvision` | Model training and inference |
| `faiss-cpu` | Nearest-neighbour retrieval |
| `opencv-python` | Image I/O, NCC template matching, visualisation |
| `onnx` + `onnxruntime` | Export and deploy on RPi5 |
| `numpy`, `Pillow` | Array ops, image transforms |

---

## Known Limitations

- **Hardcoded input path** — `IMAGE_PATH` in `cv/main.cpp` must be changed manually to point to a different FPV image.
- **No real training data** — the encoder uses ImageNet weights only. Localisation accuracy will be low until fine-tuned on real drone + satellite pairs. See `nn/data/dataset.py` (`RealPairDataset`) for the expected data layout.
- **NCC sensitive to yaw** — if the drone heading differs greatly from north, the trapezoid match may fail. A compass reading or multi-rotation NCC search would help.
- **Single-scale encoder** — may struggle in texture-poor areas (open fields, water).
- **Satellite map smaller than design** — the repo's `satellite_map.png` (570×570) is a single small reference image; a real deployment uses 500 × 1 km² tiles.
