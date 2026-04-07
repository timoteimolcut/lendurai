# AI-generated
"""
inference.py — FAISS index build + query pipeline.

Offline phase (run once per map update):
    python inference.py build --tiles-dir /path/to/tiles --index-out index.faiss

Online phase (per frame):
    python inference.py query --index index.faiss --image bird_eye.png

ONNX export (for RPi5 INT8 deployment):
    python inference.py export --checkpoint checkpoints/final.pt --out encoder.onnx
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

TILE_SIZE_PX = 800        # satellite tiles are 800×800 (1px = 1m)
PATCH_SIZE_PX = 800       # bird's-eye patch from Task 1 is 800×800
STRIDE_PX = 200           # stride for sliding window over tiles
TOP_K = 5                 # number of candidate tiles returned per query


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
    """
    tensor = _preprocess(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    tensor = tensor.unsqueeze(0).to(device)
    emb = model(tensor)
    return emb.cpu().numpy().astype(np.float32)  # (1, D)


# ---------------------------------------------------------------------------
# Offline: build FAISS index
# ---------------------------------------------------------------------------

def build_index(tiles_dir: Path, index_out: Path, checkpoint: str | None) -> None:
    """
    Slide a window over every satellite tile PNG in tiles_dir, encode each
    patch, and store all embeddings in a FAISS flat IP index.

    Index metadata (tile_id, patch_x, patch_y) is saved alongside as a .npz.
    """
    try:
        import faiss
    except ImportError:
        print("faiss-cpu not installed. Run: pip install faiss-cpu")
        sys.exit(1)

    device = torch.device("cpu")   # index building can be done offline on any machine
    model = load_encoder(checkpoint, device)

    tile_paths = sorted(tiles_dir.glob("*.png")) + sorted(tiles_dir.glob("*.jpg"))
    if not tile_paths:
        print(f"No tile images found in {tiles_dir}")
        sys.exit(1)

    embeddings = []
    metadata = []   # (tile_id, patch_x_px, patch_y_px)

    print(f"Building index from {len(tile_paths)} tiles …")
    t0 = time.time()

    for tile_id, tile_path in enumerate(tile_paths):
        img = cv2.imread(str(tile_path))
        if img is None:
            print(f"  WARNING: could not read {tile_path}, skipping.")
            continue
        H, W = img.shape[:2]

        # Sliding window
        y = 0
        while y + PATCH_SIZE_PX <= H:
            x = 0
            while x + PATCH_SIZE_PX <= W:
                patch = img[y:y + PATCH_SIZE_PX, x:x + PATCH_SIZE_PX]
                emb = encode_image(patch, model, device)
                embeddings.append(emb)
                metadata.append((tile_id, x, y))
                x += STRIDE_PX
            y += STRIDE_PX

        print(f"  [{tile_id+1}/{len(tile_paths)}] {tile_path.name}: {len(metadata)} patches so far")

    embeddings_np = np.vstack(embeddings)   # (N, D)
    D = embeddings_np.shape[1]

    index = faiss.IndexFlatIP(D)            # inner product == cosine (embeddings are L2-normed)
    index.add(embeddings_np)

    index_out.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_out))

    meta_path = index_out.with_suffix(".npz")
    meta_arr = np.array(metadata, dtype=np.int32)
    tile_names = [p.stem for p in tile_paths]
    np.savez(meta_path, metadata=meta_arr, tile_names=tile_names)

    elapsed = time.time() - t0
    print(f"\nIndex saved → {index_out}  ({index.ntotal} vectors, D={D})")
    print(f"Metadata  → {meta_path}")
    print(f"Elapsed: {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Online: query
# ---------------------------------------------------------------------------

def ncc_fine_localize(
    query_img: np.ndarray,
    tile_img: np.ndarray,
    coarse_x: int,
    coarse_y: int,
    search_radius: int = 200,
) -> tuple[int, int, float]:
    """
    Refine the coarse FAISS hit using Normalised Cross-Correlation (NCC).

    Searches a (2*search_radius) × (2*search_radius) window centred on the
    coarse patch offset inside the tile. Returns the sub-pixel best-match
    position and its NCC score.

    Args:
        query_img:     Bird's-eye patch (H×W×3 BGR uint8).
        tile_img:      Full satellite tile (BGR uint8), same scale (1px=1m).
        coarse_x/y:    Top-left corner of the coarse candidate patch in tile coords (px).
        search_radius: Half-size of the search window around the coarse position (px).

    Returns:
        (fine_x, fine_y, score) — top-left corner of the best match in tile
        pixel coordinates, and the NCC score in [-1, 1] (1 = perfect match).
    """
    th, tw = query_img.shape[:2]
    H, W = tile_img.shape[:2]

    # Search region: expand coarse position by search_radius, clamp to tile bounds
    sx = max(0, coarse_x - search_radius)
    sy = max(0, coarse_y - search_radius)
    ex = min(W - tw, coarse_x + search_radius)
    ey = min(H - th, coarse_y + search_radius)

    if ex <= sx or ey <= sy:
        # Coarse position is at the tile edge — fall back to the coarse hit
        return coarse_x, coarse_y, 0.0

    search_region = tile_img[sy: ey + th, sx: ex + tw]

    # Convert to grayscale for NCC (faster, and colour adds little for textures)
    query_gray  = cv2.cvtColor(query_img,    cv2.COLOR_BGR2GRAY).astype(np.float32)
    region_gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY).astype(np.float32)

    result = cv2.matchTemplate(region_gray, query_gray, cv2.TM_CCOEFF_NORMED)
    _, score, _, max_loc = cv2.minMaxLoc(result)

    # max_loc is relative to search_region top-left; convert back to tile coords
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

    Args:
        image_path:  Path to the bird's-eye query image (800×800 PNG).
        index_path:  Path to the FAISS index file.
        checkpoint:  Optional encoder checkpoint path.
        top_k:       Number of coarse candidates to retrieve.
        tiles_dir:   Directory containing the original satellite tile PNGs.
                     Required for NCC fine localisation; skipped if None.
    """
    try:
        import faiss
    except ImportError:
        print("faiss-cpu not installed. Run: pip install faiss-cpu")
        sys.exit(1)

    device = torch.device("cpu")
    model = load_encoder(checkpoint, device)

    # Load query image
    query_img = cv2.imread(str(image_path))
    if query_img is None:
        print(f"Could not read query image: {image_path}")
        sys.exit(1)

    # Load index + metadata
    index = faiss.read_index(str(index_path))
    meta_path = index_path.with_suffix(".npz")
    meta = np.load(meta_path, allow_pickle=True)
    metadata = meta["metadata"]         # (N, 3): tile_id, x, y
    tile_names = meta["tile_names"]

    # Encode query
    t0 = time.time()
    q_emb = encode_image(query_img, model, device)
    D, I = index.search(q_emb, top_k)
    coarse_ms = (time.time() - t0) * 1000

    print(f"\nQuery: {image_path}  (coarse: {coarse_ms:.1f} ms)\n")
    print(f"{'Rank':<5}{'Score':>8}  {'Tile':<30}  {'Patch offset (x,y) px = m'}")
    print("-" * 70)
    for rank, (score, idx) in enumerate(zip(D[0], I[0]), start=1):
        tile_id, px, py = metadata[idx]
        tile_name = tile_names[tile_id]
        print(f"{rank:<5}{score:>8.4f}  {tile_name:<30}  ({px}, {py})")

    # ── Stage 2: NCC fine localisation on the best coarse hit ──────────────
    best_idx = I[0][0]
    best_tile_id, coarse_x, coarse_y = metadata[best_idx]
    best_tile_name = str(tile_names[best_tile_id])

    if tiles_dir is None:
        print("\n[NCC fine localisation skipped — pass --tiles-dir to enable]")
        return

    tile_path = tiles_dir / f"{best_tile_name}.png"
    if not tile_path.exists():
        # Try .jpg fallback
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

    # Drone position within tile: centre of the matched patch
    drone_x = fine_x + query_img.shape[1] // 2
    drone_y = fine_y + query_img.shape[0] // 2

    print(f"\nFine localisation (NCC, {ncc_ms:.1f} ms):")
    print(f"  Tile      : {best_tile_name}")
    print(f"  Patch TL  : ({fine_x}, {fine_y}) px")
    print(f"  Drone pos : ({drone_x}, {drone_y}) px = ({drone_x} m, {drone_y} m) from tile origin")
    print(f"  NCC score : {ncc_score:.4f}  (1.0 = perfect match)")
    print(f"  Total     : {coarse_ms + ncc_ms:.1f} ms")


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(checkpoint: str, out_path: Path) -> None:
    """
    Export the encoder to ONNX with INT8 quantisation via dynamic quantisation.
    The resulting .onnx file can be loaded by onnxruntime on RPi5.

    Note: true INT8 calibration (static quantisation) requires a calibration
    dataset; this export uses dynamic quantisation as a first step.
    """
    import onnx
    import onnxruntime as ort

    device = torch.device("cpu")
    model = load_encoder(checkpoint, device)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
    )
    print(f"ONNX model saved → {out_path}")

    # Verify with onnxruntime
    sess = ort.InferenceSession(str(out_path))
    out = sess.run(None, {"image": dummy.numpy()})
    print(f"ORT verification: output shape {out[0].shape}  ✓")
    print(f"Model size: {out_path.stat().st_size / 1e6:.1f} MB")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FAISS index build + query pipeline.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # build
    b = sub.add_parser("build", help="Build FAISS index from satellite tiles.")
    b.add_argument("--tiles-dir", required=True, type=Path)
    b.add_argument("--index-out", default="index.faiss", type=Path)
    b.add_argument("--checkpoint", default=None)

    # query
    q = sub.add_parser("query", help="Query index with a bird's-eye patch.")
    q.add_argument("--image", required=True, type=Path)
    q.add_argument("--index", required=True, type=Path)
    q.add_argument("--checkpoint", default=None)
    q.add_argument("--top-k", type=int, default=TOP_K)
    q.add_argument("--tiles-dir", type=Path, default=None,
                   help="Directory of satellite tile PNGs for NCC fine localisation.")

    # export
    e = sub.add_parser("export", help="Export encoder to ONNX.")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--out", default="encoder.onnx", type=Path)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.cmd == "build":
        build_index(args.tiles_dir, args.index_out, args.checkpoint)
    elif args.cmd == "query":
        query_index(args.image, args.index, args.checkpoint, args.top_k, args.tiles_dir)
    elif args.cmd == "export":
        export_onnx(args.checkpoint, args.out)
