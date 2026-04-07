# AI-generated
"""
inference.py — FAISS index build + query pipeline + satellite footprint visualisation.

Offline phase (run once per map update):
    python inference.py build \
        --tiles-dir images/src/processed/ \
        --index-out index.faiss \
        --patch-size 256 --stride 100

Online phase (per frame):
    python inference.py query \
        --image images/dst/dst/birds_eye_map_*.png \
        --index index.faiss \
        --tiles-dir images/src/processed/

Footprint visualisation — highlight the visible ground region on the satellite:
    python inference.py visualize \
        --image  images/dst/dst/birds_eye_map_*.png \
        --satellite images/src/processed/satellite_map.png \
        --out    result.png

ONNX export (for RPi5 INT8 deployment):
    python inference.py export --checkpoint checkpoints/final.pt --out encoder.onnx

Real data paths:
    FPV input  : images/src/processed/image_fpv_drone.png       (490×216)
    Bird's-eye : images/dst/dst/birds_eye_map_*.png             (800×800, ~half black)
    Satellite  : images/src/processed/satellite_map.png         (570×570)
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T

from models.encoder import Encoder


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_preprocess = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Default sliding-window parameters.
# 256px patches at stride 100 give good coverage on a 570×570 satellite map
# (16 patches) while still being large enough for meaningful texture features.
DEFAULT_PATCH_SIZE = 256
DEFAULT_STRIDE     = 100
TOP_K              = 5      # coarse candidates returned per query


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def crop_nonblack(img: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Crop the tight bounding box of non-black pixels from a BGR image.

    Bird's-eye outputs from Task 1 are 800×800 but contain large black borders
    where the homography has no source pixels (especially at shallow pitch
    angles).  Cropping to the non-black region before encoding gives the
    encoder a much cleaner signal.

    Args:
        img:       BGR uint8 image.
        threshold: Pixel intensity below which a pixel is considered black.

    Returns:
        Cropped image. Returns the original unchanged if the image is all-black
        or if the crop would produce an empty result.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > threshold
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return img   # nothing to crop
    return img[rows[0]: rows[-1] + 1, cols[0]: cols[-1] + 1]


# ---------------------------------------------------------------------------
# Encoder loading
# ---------------------------------------------------------------------------

def load_encoder(checkpoint_path: str | None, device: torch.device) -> Encoder:
    model = Encoder(pretrained=(checkpoint_path is None))
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        print(f"Loaded encoder weights from {checkpoint_path}")
    model.eval().to(device)
    return model


@torch.inference_mode()
def encode_image(img_bgr: np.ndarray, model: Encoder, device: torch.device) -> np.ndarray:
    """
    Encode a single BGR image (H×W×3 uint8) → 128-dim float32 embedding.
    The image is resized to 224×224 internally.
    """
    tensor = _preprocess(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    tensor = tensor.unsqueeze(0).to(device)
    emb = model(tensor)
    return emb.cpu().numpy().astype(np.float32)  # (1, D)


# ---------------------------------------------------------------------------
# Offline: build FAISS index
# ---------------------------------------------------------------------------

def build_index(
    tiles_dir: Path,
    index_out: Path,
    checkpoint: str | None,
    patch_size: int = DEFAULT_PATCH_SIZE,
    stride: int = DEFAULT_STRIDE,
) -> None:
    """
    Slide a window over every satellite tile image in tiles_dir, encode each
    patch, and store all embeddings in a FAISS flat IP index.

    Handles satellite maps smaller than patch_size: the effective patch is
    clamped to min(patch_size, H, W) so at least one patch is always extracted
    per image, even for small maps like the 570×570 satellite_map.png.

    Args:
        tiles_dir:  Directory containing satellite PNG/JPG images.
        index_out:  Output path for the FAISS index file.
        checkpoint: Optional encoder checkpoint; uses ImageNet weights if None.
        patch_size: Sliding window size in pixels (default 256).
                    Clamped to image dimensions if the image is smaller.
        stride:     Sliding window stride in pixels (default 100).

    Saves:
        index_out        — FAISS IndexFlatIP binary
        index_out.npz    — metadata array (tile_id, patch_x, patch_y) + tile names
    """
    try:
        import faiss
    except ImportError:
        print("faiss-cpu not installed. Run: pip install faiss-cpu")
        sys.exit(1)

    device = torch.device("cpu")
    model = load_encoder(checkpoint, device)

    tile_paths = sorted(tiles_dir.glob("*.png")) + sorted(tiles_dir.glob("*.jpg"))
    if not tile_paths:
        print(f"No tile images found in {tiles_dir}")
        sys.exit(1)

    embeddings = []
    metadata   = []   # (tile_id, patch_x_px, patch_y_px)

    print(f"Building index from {len(tile_paths)} tile(s) "
          f"[patch={patch_size}px, stride={stride}px] …")
    t0 = time.time()

    for tile_id, tile_path in enumerate(tile_paths):
        img = cv2.imread(str(tile_path))
        if img is None:
            print(f"  WARNING: could not read {tile_path}, skipping.")
            continue
        H, W = img.shape[:2]

        # Clamp patch size so we always extract at least one patch even when
        # the satellite map is smaller than the requested patch size (e.g.
        # satellite_map.png is 570×570 which is smaller than 800px default).
        eff_patch  = min(patch_size, H, W)
        eff_stride = min(stride, eff_patch)   # stride can't exceed patch size

        n_before = len(embeddings)
        y = 0
        while y + eff_patch <= H:
            x = 0
            while x + eff_patch <= W:
                patch = img[y: y + eff_patch, x: x + eff_patch]
                emb   = encode_image(patch, model, device)
                embeddings.append(emb)
                metadata.append((tile_id, x, y))
                x += eff_stride
            y += eff_stride

        n_patches = len(embeddings) - n_before
        print(f"  [{tile_id+1}/{len(tile_paths)}] {tile_path.name} "
              f"({W}×{H}px, eff_patch={eff_patch}px): {n_patches} patches")

    if not embeddings:
        print("ERROR: no patches were extracted. Check tile images.")
        sys.exit(1)

    embeddings_np = np.vstack(embeddings)   # (N, D)
    D = embeddings_np.shape[1]

    index = faiss.IndexFlatIP(D)            # inner product = cosine (L2-normed embeddings)
    index.add(embeddings_np)

    index_out.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_out))

    meta_path  = index_out.with_suffix(".npz")
    meta_arr   = np.array(metadata, dtype=np.int32)
    tile_names = [p.stem for p in tile_paths]
    np.savez(meta_path, metadata=meta_arr, tile_names=tile_names,
             patch_size=np.int32(eff_patch))

    elapsed = time.time() - t0
    print(f"\nIndex saved  → {index_out}  ({index.ntotal} vectors, D={D})")
    print(f"Metadata     → {meta_path}")
    print(f"Elapsed: {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Online: query
# ---------------------------------------------------------------------------

def ncc_fine_localize(
    query_img: np.ndarray,
    tile_img: np.ndarray,
    coarse_x: int,
    coarse_y: int,
    search_radius: int = 100,
) -> tuple[int, int, float]:
    """
    Refine the coarse FAISS hit using Normalised Cross-Correlation (NCC).

    The query (bird's-eye crop) is resized to fit within the satellite tile
    before matching, because the warped ground region can be wider than the
    satellite map (e.g. 800px bird's-eye vs 570px satellite).

    Args:
        query_img:     Non-black-cropped bird's-eye patch (BGR uint8).
        tile_img:      Satellite tile image (BGR uint8).
        coarse_x/y:    Top-left of the best FAISS patch in tile coordinates.
        search_radius: Half-size of the NCC search window (px in tile space).

    Returns:
        (fine_x, fine_y, score) — top-left corner of best match in tile pixel
        coordinates, and NCC score in [-1, 1].
    """
    H, W = tile_img.shape[:2]

    # Resize query so it fits strictly inside the satellite tile.
    # This is necessary when the bird's-eye crop is wider or taller than the
    # satellite (e.g. 800px wide crop vs 570px wide tile).
    qh, qw = query_img.shape[:2]
    scale = min((W - 1) / qw, (H - 1) / qh, 1.0)   # never upscale
    if scale < 1.0:
        new_w = max(1, int(qw * scale))
        new_h = max(1, int(qh * scale))
        query_resized = cv2.resize(query_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        query_resized = query_img

    th, tw = query_resized.shape[:2]

    # Search window: ±search_radius around the coarse hit, clamped to tile
    sx = max(0, coarse_x - search_radius)
    sy = max(0, coarse_y - search_radius)
    ex = min(W - tw, coarse_x + search_radius)
    ey = min(H - th, coarse_y + search_radius)

    if ex < sx or ey < sy:
        # Coarse position at tile edge; use coarse result directly
        return coarse_x, coarse_y, 0.0

    search_region = tile_img[sy: ey + th, sx: ex + tw]

    # NCC on grayscale (faster; colour adds little for overhead vegetation)
    query_gray  = cv2.cvtColor(query_resized,  cv2.COLOR_BGR2GRAY).astype(np.float32)
    region_gray = cv2.cvtColor(search_region,  cv2.COLOR_BGR2GRAY).astype(np.float32)

    if region_gray.shape[0] < query_gray.shape[0] or region_gray.shape[1] < query_gray.shape[1]:
        # Search region collapsed; fall back to coarse
        return coarse_x, coarse_y, 0.0

    result = cv2.matchTemplate(region_gray, query_gray, cv2.TM_CCOEFF_NORMED)
    _, score, _, max_loc = cv2.minMaxLoc(result)

    fine_x = sx + max_loc[0]
    fine_y = sy + max_loc[1]

    return fine_x, fine_y, float(score)


def query_index(
    image_path: Path,
    index_path: Path,
    checkpoint: str | None,
    top_k: int = TOP_K,
    tiles_dir: Path | None = None,
) -> None:
    """
    Query the FAISS index with a bird's-eye patch and print the top-K results,
    then run NCC fine localisation on the best candidate.

    The bird's-eye image is automatically cropped to its non-black bounding box
    before encoding, removing the large black borders produced by Task 1 when
    pitch is shallow (e.g. -25° at 60 m leaves ~half the canvas black).

    Args:
        image_path:  Bird's-eye PNG from Task 1 (800×800).
        index_path:  FAISS index file produced by build_index.
        checkpoint:  Optional encoder checkpoint path.
        top_k:       Number of coarse FAISS candidates to retrieve.
        tiles_dir:   Directory of satellite tile images for NCC Stage 2.
                     Pass None to skip NCC and return coarse results only.
    """
    try:
        import faiss
    except ImportError:
        print("faiss-cpu not installed. Run: pip install faiss-cpu")
        sys.exit(1)

    device = torch.device("cpu")
    model = load_encoder(checkpoint, device)

    # Load query bird's-eye image
    raw_query = cv2.imread(str(image_path))
    if raw_query is None:
        print(f"Could not read query image: {image_path}")
        sys.exit(1)

    # Crop black border — bird's-eye images from Task 1 have large black areas
    # outside the warped ground region. Removing them before encoding avoids
    # the encoder being confused by blank pixels.
    query_img = crop_nonblack(raw_query)
    if query_img.shape != raw_query.shape:
        print(f"  Non-black crop: {raw_query.shape[1]}×{raw_query.shape[0]} "
              f"→ {query_img.shape[1]}×{query_img.shape[0]} px")

    # Load index + metadata
    index     = faiss.read_index(str(index_path))
    meta_path = index_path.with_suffix(".npz")
    meta      = np.load(meta_path, allow_pickle=True)
    metadata   = meta["metadata"]       # (N, 3): tile_id, x, y
    tile_names = meta["tile_names"]

    # Encode cropped query and search
    t0 = time.time()
    q_emb = encode_image(query_img, model, device)
    top_k  = min(top_k, index.ntotal)  # can't request more candidates than index has
    D, I   = index.search(q_emb, top_k)
    coarse_ms = (time.time() - t0) * 1000

    print(f"\nQuery: {image_path.name}  (coarse: {coarse_ms:.1f} ms)\n")
    print(f"{'Rank':<5}{'Score':>8}  {'Tile':<35}  {'Patch offset (x,y) px'}")
    print("-" * 72)
    for rank, (score, idx) in enumerate(zip(D[0], I[0]), start=1):
        tile_id, px, py = metadata[idx]
        tile_name = tile_names[tile_id]
        print(f"{rank:<5}{score:>8.4f}  {tile_name:<35}  ({px}, {py})")

    # ── Stage 2: NCC fine localisation on the best coarse hit ──────────────
    best_idx = I[0][0]
    best_tile_id, coarse_x, coarse_y = metadata[best_idx]
    best_tile_name = str(tile_names[best_tile_id])

    if tiles_dir is None:
        print("\n[NCC fine localisation skipped — pass --tiles-dir to enable]")
        return

    tile_path = tiles_dir / f"{best_tile_name}.png"
    if not tile_path.exists():
        tile_path = tiles_dir / f"{best_tile_name}.jpg"
    if not tile_path.exists():
        print(f"\n[NCC skipped — tile image not found: {best_tile_name}]")
        return

    tile_img = cv2.imread(str(tile_path))
    if tile_img is None:
        print(f"\n[NCC skipped — could not read tile: {tile_path}]")
        return

    t1 = time.time()
    fine_x, fine_y, ncc_score = ncc_fine_localize(
        query_img, tile_img, int(coarse_x), int(coarse_y)
    )
    ncc_ms = (time.time() - t1) * 1000

    # Drone position = centre of matched query patch, in tile pixel coordinates
    drone_x = fine_x + query_img.shape[1] // 2
    drone_y = fine_y + query_img.shape[0] // 2

    print(f"\nFine localisation (NCC, {ncc_ms:.1f} ms):")
    print(f"  Tile       : {best_tile_name}")
    print(f"  Match TL   : ({fine_x}, {fine_y}) px in tile")
    print(f"  Drone pos  : ({drone_x}, {drone_y}) px from tile origin")
    print(f"  NCC score  : {ncc_score:.4f}  (1.0 = perfect match)")
    print(f"  Total time : {coarse_ms + ncc_ms:.1f} ms")


# ---------------------------------------------------------------------------
# Footprint visualisation
# ---------------------------------------------------------------------------

def visualize_footprint(
    birdseye_path: Path,
    satellite_path: Path,
    out_path: Path,
    drone_origin: tuple[int, int] = (400, 400),
) -> None:
    """
    Locate the ground region visible in the bird's-eye image on the satellite
    map and save an annotated image.

    Pipeline:
        1. Load the 800×800 bird's-eye patch from Task 1.
        2. Crop the non-black bounding box — removes the empty border outside
           the warped ground region.
        3. Extract the binary non-black mask of the crop (the trapezoid shape).
        4. Scale the cropped patch so it fits strictly inside the satellite.
        5. Run NCC (cv2.matchTemplate) to find the best-matching position on
           the satellite.
        6. Project the trapezoid mask into satellite pixel space.
        7. Draw: semi-transparent fill, contour, drone dot, crosshair, legend.

    Args:
        birdseye_path:  800×800 PNG output from the C++ warp (Task 1).
        satellite_path: Satellite reference map (any size).
        out_path:       Where to save the annotated result PNG.
        drone_origin:   Pixel position of the drone in the full 800×800 image.
                        Default is (400, 400) — centre, as set by Task 1.
    """
    # ── 1. Load images ───────────────────────────────────────────────────────
    birdseye = cv2.imread(str(birdseye_path))
    if birdseye is None:
        raise FileNotFoundError(f"Cannot read bird's-eye image: {birdseye_path}")

    satellite = cv2.imread(str(satellite_path))
    if satellite is None:
        raise FileNotFoundError(f"Cannot read satellite image: {satellite_path}")

    sat_H, sat_W = satellite.shape[:2]
    print(f"Bird's-eye : {birdseye.shape[1]}×{birdseye.shape[0]} px")
    print(f"Satellite  : {sat_W}×{sat_H} px")

    # ── 2. Crop non-black bounding box ───────────────────────────────────────
    gray   = cv2.cvtColor(birdseye, cv2.COLOR_BGR2GRAY)
    mask   = gray > 10
    rows   = np.where(np.any(mask, axis=1))[0]
    cols   = np.where(np.any(mask, axis=0))[0]

    if rows.size == 0 or cols.size == 0:
        raise ValueError("Bird's-eye image appears to be entirely black.")

    r0, r1 = int(rows[0]), int(rows[-1])
    c0, c1 = int(cols[0]), int(cols[-1])
    crop   = birdseye[r0: r1 + 1, c0: c1 + 1]
    qh, qw = crop.shape[:2]
    print(f"Non-black crop: {qw}×{qh} px  (bbox rows [{r0},{r1}], cols [{c0},{c1}])")

    # ── 3. Non-black mask of the crop (trapezoid shape) ──────────────────────
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, footprint_mask = cv2.threshold(crop_gray, 10, 255, cv2.THRESH_BINARY)

    # ── 4. Scale crop to fit inside satellite ─────────────────────────────────
    scale = min((sat_W - 1) / qw, (sat_H - 1) / qh, 1.0)
    rw    = max(1, int(qw * scale))
    rh    = max(1, int(qh * scale))
    crop_resized = cv2.resize(crop,          (rw, rh), interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(footprint_mask,(rw, rh), interpolation=cv2.INTER_NEAREST)
    print(f"NCC scale  : {scale:.4f}  →  resized query {rw}×{rh} px")

    # ── 5. NCC matching on grayscale ─────────────────────────────────────────
    sat_gray    = cv2.cvtColor(satellite,   cv2.COLOR_BGR2GRAY).astype(np.float32)
    query_gray  = cv2.cvtColor(crop_resized,cv2.COLOR_BGR2GRAY).astype(np.float32)

    result              = cv2.matchTemplate(sat_gray, query_gray, cv2.TM_CCOEFF_NORMED)
    _, ncc_score, _, loc = cv2.minMaxLoc(result)
    match_x, match_y    = int(loc[0]), int(loc[1])   # top-left of match in satellite
    print(f"NCC match  : top-left ({match_x}, {match_y}),  score={ncc_score:.4f}")

    # ── 6. Place trapezoid mask in satellite space ────────────────────────────
    # The mask_resized sits at (match_x, match_y) in the satellite canvas.
    sat_mask = np.zeros((sat_H, sat_W), dtype=np.uint8)
    # Clamp to canvas bounds
    dst_x1 = max(0, match_x);      dst_y1 = max(0, match_y)
    dst_x2 = min(sat_W, match_x + rw);  dst_y2 = min(sat_H, match_y + rh)
    src_x1 = dst_x1 - match_x;    src_y1 = dst_y1 - match_y
    src_x2 = src_x1 + (dst_x2 - dst_x1)
    src_y2 = src_y1 + (dst_y2 - dst_y1)
    sat_mask[dst_y1:dst_y2, dst_x1:dst_x2] = mask_resized[src_y1:src_y2, src_x1:src_x2]

    # Drone position in satellite coordinates
    # drone_origin is in full 800×800 space; adjust for crop offset then scale
    drone_sat_x = match_x + int((drone_origin[0] - c0) * scale)
    drone_sat_y = match_y + int((drone_origin[1] - r0) * scale)
    drone_sat_x = int(np.clip(drone_sat_x, 0, sat_W - 1))
    drone_sat_y = int(np.clip(drone_sat_y, 0, sat_H - 1))
    print(f"Drone pos  : ({drone_sat_x}, {drone_sat_y}) px on satellite")

    # ── 7. Draw overlay on satellite ─────────────────────────────────────────
    result_img = satellite.copy()

    # Semi-transparent cyan fill for the visible ground footprint
    FILL_COLOR  = (200, 180, 0)   # cyan-ish in BGR
    FILL_ALPHA  = 0.35
    overlay     = result_img.copy()
    overlay[sat_mask > 0] = FILL_COLOR
    cv2.addWeighted(overlay, FILL_ALPHA, result_img, 1 - FILL_ALPHA, 0, result_img)

    # Contour of the footprint region
    contours, _ = cv2.findContours(sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result_img, contours, -1, (0, 220, 220), 2)

    # Drone marker: filled circle + crosshair
    DRONE_COLOR = (0, 0, 255)   # red
    cv2.circle(result_img, (drone_sat_x, drone_sat_y), 6, DRONE_COLOR, -1)
    cv2.circle(result_img, (drone_sat_x, drone_sat_y), 9, DRONE_COLOR,  2)
    arm = 14
    cv2.line(result_img, (drone_sat_x - arm, drone_sat_y),
             (drone_sat_x + arm, drone_sat_y), DRONE_COLOR, 1)
    cv2.line(result_img, (drone_sat_x, drone_sat_y - arm),
             (drone_sat_x, drone_sat_y + arm), DRONE_COLOR, 1)

    # Legend box (bottom-left)
    font       = cv2.FONT_HERSHEY_SIMPLEX
    legend_y   = sat_H - 10
    texts = [
        (f"NCC score: {ncc_score:.3f}", (180, 180, 180)),
        (f"Drone: ({drone_sat_x}, {drone_sat_y}) px", (0, 0, 255)),
        ("Visible ground region", (0, 220, 220)),
    ]
    for i, (text, color) in enumerate(reversed(texts)):
        y = legend_y - i * 18
        cv2.putText(result_img, text, (6, y), font, 0.42, (0, 0, 0),    3, cv2.LINE_AA)
        cv2.putText(result_img, text, (6, y), font, 0.42, color,         1, cv2.LINE_AA)

    # ── 8. Save ───────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), result_img)
    print(f"\nSaved → {out_path}")


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(checkpoint: str, out_path: Path) -> None:
    """
    Export the encoder to ONNX with dynamic INT8 quantisation.
    The resulting file can be run with onnxruntime on RPi5.
    """
    import onnx
    import onnxruntime as ort

    device = torch.device("cpu")
    model  = load_encoder(checkpoint, device)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
    )
    print(f"ONNX model saved → {out_path}")

    sess = ort.InferenceSession(str(out_path))
    out  = sess.run(None, {"image": dummy.numpy()})
    print(f"ORT verification: output shape {out[0].shape}  ✓")
    print(f"Model size: {out_path.stat().st_size / 1e6:.1f} MB")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p   = argparse.ArgumentParser(description="FAISS index build + query pipeline.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # build
    b = sub.add_parser("build", help="Build FAISS index from satellite tiles.")
    b.add_argument("--tiles-dir",  required=True, type=Path,
                   help="Directory of satellite PNG/JPG images.")
    b.add_argument("--index-out",  default="index.faiss", type=Path)
    b.add_argument("--checkpoint", default=None)
    b.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE,
                   help=f"Sliding window patch size in px (default {DEFAULT_PATCH_SIZE}). "
                        "Clamped to image dimensions automatically.")
    b.add_argument("--stride",     type=int, default=DEFAULT_STRIDE,
                   help=f"Sliding window stride in px (default {DEFAULT_STRIDE}).")

    # query
    q = sub.add_parser("query", help="Query index with a bird's-eye patch.")
    q.add_argument("--image",      required=True, type=Path,
                   help="Bird's-eye patch PNG (Task 1 output, 800×800).")
    q.add_argument("--index",      required=True, type=Path)
    q.add_argument("--checkpoint", default=None)
    q.add_argument("--top-k",      type=int, default=TOP_K)
    q.add_argument("--tiles-dir",  type=Path, default=None,
                   help="Satellite tile directory for NCC Stage 2. Omit to skip NCC.")

    # visualize
    v = sub.add_parser("visualize",
                       help="Highlight the drone's visible ground region on the satellite map.")
    v.add_argument("--image",     required=True, type=Path,
                   help="Bird's-eye PNG from Task 1 (800×800).")
    v.add_argument("--satellite", required=True, type=Path,
                   help="Satellite reference map PNG.")
    v.add_argument("--out",       default="result.png", type=Path,
                   help="Output annotated image path (default: result.png).")

    # export
    e = sub.add_parser("export", help="Export encoder to ONNX.")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--out",        default="encoder.onnx", type=Path)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.cmd == "build":
        build_index(args.tiles_dir, args.index_out, args.checkpoint,
                    args.patch_size, args.stride)
    elif args.cmd == "query":
        query_index(args.image, args.index, args.checkpoint,
                    args.top_k, args.tiles_dir)
    elif args.cmd == "visualize":
        visualize_footprint(args.image, args.satellite, args.out)
    elif args.cmd == "export":
        export_onnx(args.checkpoint, args.out)
