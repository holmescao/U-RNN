"""
tools/downsample_dataset.py
============================
Create a spatially and/or temporally downsampled copy of the UrbanFlood24 dataset.

This script reduces the resolution of an existing dataset so that U-RNN can be
trained on machines with limited GPU memory and storage.  The original dataset
(2 m / 1 min) is ~20 GB compressed.  A typical lightweight version (8 m / 10 min)
occupies < 1 GB.

Usage
-----
    python tools/downsample_dataset.py \\
        --src_root  ./data/urbanflood24 \\
        --dst_root  ./data/urbanflood24_lite \\
        --spatial_factor 4 \\
        --temporal_factor 10

Arguments
---------
--src_root       Path to the original dataset root  (must contain train/ and test/).
--dst_root       Output path for the downsampled dataset.
--spatial_factor Spatial downsampling factor (integer ≥ 1).
                 E.g. 4 converts 2 m → 8 m by averaging 4×4 blocks.
--temporal_factor Temporal downsampling factor (integer ≥ 1).
                 E.g. 10 converts 1-min → 10-min steps by summing bins (for
                 rainfall) or sampling (for flood/DEM).
--splits         Which splits to process: "train", "test", or "train test"
                 (default: both).

Output structure
----------------
The output mirrors the input structure exactly:
    <dst_root>/
    ├── train/
    │   ├── flood/<location>/<event>/
    │   │   ├── flood.npy   (T', H', W')
    │   │   └── rainfall.npy (T', H', W')    [if spatial rainfall grids]
    │   │                  or (T',)          [if scalar rainfall]
    │   └── geodata/<location>/
    │       ├── absolute_DSM.npy (H', W')
    │       ├── impervious.npy   (H', W')
    │       └── manhole.npy      (H', W')
    └── test/  (same structure)

Notes
-----
* Flood depth is downsampled spatially by block-averaging (preserving mass).
* Rainfall is summed over each temporal block so total rainfall is conserved.
* DSM (DEM), impervious, and manhole maps are downsampled by block-averaging.
* When spatial_factor = 1 and temporal_factor = 1, the script copies without
  modification (useful for format validation).
"""

import argparse
import os
import shutil
import sys

import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------

def spatial_downsample_2d(arr, factor):
    """Block-average a 2-D array (H, W) by `factor` in each spatial dimension."""
    H, W = arr.shape
    H_new = H // factor
    W_new = W // factor
    arr_crop = arr[:H_new * factor, :W_new * factor]
    return arr_crop.reshape(H_new, factor, W_new, factor).mean(axis=(1, 3))


def spatial_downsample_3d(arr, factor):
    """Block-average a 3-D array (T, H, W) spatially by `factor`."""
    T, H, W = arr.shape
    H_new = H // factor
    W_new = W // factor
    arr_crop = arr[:, :H_new * factor, :W_new * factor]
    return arr_crop.reshape(T, H_new, factor, W_new, factor).mean(axis=(2, 4))


def spatial_pad_2d(arr, target_h, target_w, fill=0.0):
    """Pad (H, W) to (target_h, target_w) by appending rows/cols at right and bottom."""
    H, W = arr.shape
    if H >= target_h and W >= target_w:
        return arr
    out = np.full((target_h, target_w), fill, dtype=arr.dtype)
    out[:H, :W] = arr
    return out


def spatial_pad_3d(arr, target_h, target_w, fill=0.0):
    """Pad (T, H, W) to (T, target_h, target_w) by appending at right and bottom."""
    T, H, W = arr.shape
    if H >= target_h and W >= target_w:
        return arr
    out = np.full((T, target_h, target_w), fill, dtype=arr.dtype)
    out[:, :H, :W] = arr
    return out


# ---------------------------------------------------------------------------
# Temporal helpers
# ---------------------------------------------------------------------------

def temporal_downsample_rainfall(arr, factor):
    """
    Temporally downsample rainfall by *summing* each block of `factor` steps.
    This preserves the total rainfall accumulation.

    Supports shapes (T,) and (T, H, W).
    """
    T = arr.shape[0]
    T_new = T // factor
    rest = arr.shape[1:]  # () or (H, W)

    arr_crop = arr[:T_new * factor]
    if arr.ndim == 1:
        return arr_crop.reshape(T_new, factor).sum(axis=1)
    else:
        H, W = rest
        return arr_crop.reshape(T_new, factor, H, W).sum(axis=1)


def temporal_downsample_sample(arr, factor):
    """
    Temporally downsample by *sampling* every `factor`-th step.
    Used for flood depth (not a rate — sampling is more appropriate).

    Supports shapes (T, H, W).
    """
    T_new = arr.shape[0] // factor
    return arr[:T_new * factor:factor]


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_event_dir(event_src, event_dst, spatial_factor, temporal_factor,
                      pad_h=0, pad_w=0):
    """
    Process all .npy files in a single event directory.

    Convention
    ----------
    - flood.npy   → (T, H, W): spatial-temporal → block-avg space, sample time
    - rainfall.npy → (T,) or (T, H, W): sum in time, block-avg space

    Spatial padding (right + bottom, fill=0) is applied after downsampling
    when pad_h > 0 or pad_w > 0.
    """
    os.makedirs(event_dst, exist_ok=True)

    for fname in os.listdir(event_src):
        if not fname.endswith(".npy"):
            continue
        src_path = os.path.join(event_src, fname)
        dst_path = os.path.join(event_dst, fname)

        arr = np.load(src_path, allow_pickle=False)
        arr_out = _process_array(fname, arr, spatial_factor, temporal_factor,
                                 pad_h=pad_h, pad_w=pad_w)
        np.save(dst_path, arr_out)


def process_geodata_dir(geo_src, geo_dst, spatial_factor, pad_h=0, pad_w=0):
    """
    Process all .npy files in a geodata directory (static spatial maps).

    Padding fill values (applied after downsampling):
      absolute_DEM.npy → fill = 100 (m)  avoids low-DEM artifacts at boundary
      all others        → fill = 0
    """
    os.makedirs(geo_dst, exist_ok=True)

    for fname in os.listdir(geo_src):
        if not fname.endswith(".npy"):
            continue
        src_path = os.path.join(geo_src, fname)
        dst_path = os.path.join(geo_dst, fname)

        arr = np.load(src_path, allow_pickle=False).astype(np.float32)
        if arr.ndim == 2 and spatial_factor > 1:
            arr = spatial_downsample_2d(arr, spatial_factor)
        if arr.ndim == 2 and (pad_h > 0 or pad_w > 0):
            th = pad_h if pad_h > 0 else arr.shape[0]
            tw = pad_w if pad_w > 0 else arr.shape[1]
            # DEM: fill boundary with 100 m to avoid artificial low-elevation paths
            fill = 100.0 if fname == "absolute_DEM.npy" else 0.0
            arr = spatial_pad_2d(arr, th, tw, fill=fill)
        np.save(dst_path, arr)


def _process_array(fname, arr, spatial_factor, temporal_factor, pad_h=0, pad_w=0):
    """Dispatch downsampling + padding based on filename semantics."""
    arr = arr.astype(np.float32)

    if fname == "flood.npy":
        # Stored as (T, H, W) or (T, 1, H, W) — squeeze channel dim if present
        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr[:, 0, :, :]   # (T, 1, H, W) → (T, H, W)
        # (T, H, W): sample in time, block-avg in space
        if temporal_factor > 1:
            arr = temporal_downsample_sample(arr, temporal_factor)
        if spatial_factor > 1:
            arr = spatial_downsample_3d(arr, spatial_factor)
        if pad_h > 0 or pad_w > 0:
            th = pad_h if pad_h > 0 else arr.shape[1]
            tw = pad_w if pad_w > 0 else arr.shape[2]
            arr = spatial_pad_3d(arr, th, tw, fill=0.0)

    elif fname == "rainfall.npy":
        if arr.ndim == 1:
            # Scalar time series: sum over temporal blocks
            if temporal_factor > 1:
                arr = temporal_downsample_rainfall(arr, temporal_factor)
            # No spatial padding for scalar rainfall
        elif arr.ndim == 3:
            # (T, H, W) spatial rainfall
            if temporal_factor > 1:
                arr = temporal_downsample_rainfall(arr, temporal_factor)
            if spatial_factor > 1:
                arr = spatial_downsample_3d(arr, spatial_factor)
            if pad_h > 0 or pad_w > 0:
                th = pad_h if pad_h > 0 else arr.shape[1]
                tw = pad_w if pad_w > 0 else arr.shape[2]
                arr = spatial_pad_3d(arr, th, tw, fill=0.0)
        else:
            raise ValueError(
                f"Unexpected rainfall.npy shape {arr.shape} — expected 1-D or 3-D."
            )

    else:
        # Unknown file — copy as-is
        pass

    return arr


# ---------------------------------------------------------------------------
# Dataset walking
# ---------------------------------------------------------------------------

def process_split(split_src, split_dst, spatial_factor, temporal_factor,
                  pad_h=0, pad_w=0, verbose=True):
    """Walk a split (train or test) and process all events and geodata."""
    flood_src = os.path.join(split_src, "flood")
    geo_src = os.path.join(split_src, "geodata")
    flood_dst = os.path.join(split_dst, "flood")
    geo_dst = os.path.join(split_dst, "geodata")

    if not os.path.isdir(flood_src):
        print(f"  [SKIP] flood directory not found: {flood_src}", file=sys.stderr)
        return

    # Process event data
    locations = sorted(os.listdir(flood_src))
    for loc in tqdm(locations, desc="  Locations", leave=False):
        loc_src = os.path.join(flood_src, loc)
        loc_dst = os.path.join(flood_dst, loc)
        os.makedirs(loc_dst, exist_ok=True)

        events = sorted(os.listdir(loc_src))
        for evt in tqdm(events, desc=f"  {loc}", leave=False):
            event_src = os.path.join(loc_src, evt)
            event_dst = os.path.join(loc_dst, evt)
            if os.path.isdir(event_src):
                process_event_dir(event_src, event_dst, spatial_factor,
                                  temporal_factor, pad_h=pad_h, pad_w=pad_w)

    # Process geodata
    if os.path.isdir(geo_src):
        geo_locations = sorted(os.listdir(geo_src))
        for loc in tqdm(geo_locations, desc="  Geodata", leave=False):
            g_src = os.path.join(geo_src, loc)
            g_dst = os.path.join(geo_dst, loc)
            if os.path.isdir(g_src):
                process_geodata_dir(g_src, g_dst, spatial_factor,
                                    pad_h=pad_h, pad_w=pad_w)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Downsample the UrbanFlood24 dataset spatially and/or temporally."
    )
    parser.add_argument(
        "--src_root", type=str, required=True,
        help="Path to the source dataset root (must contain train/ and/or test/)."
    )
    parser.add_argument(
        "--dst_root", type=str, required=True,
        help="Output path for the downsampled dataset."
    )
    parser.add_argument(
        "--spatial_factor", type=int, default=1,
        help="Spatial downsampling factor (≥1). E.g. 4 converts 2 m → 8 m. Default: 1 (no change)."
    )
    parser.add_argument(
        "--temporal_factor", type=int, default=1,
        help="Temporal downsampling factor (≥1). E.g. 10 converts 1-min → 10-min steps. Default: 1 (no change)."
    )
    parser.add_argument(
        "--pad_h", type=int, default=0,
        help="Target height after padding (0 = no padding). E.g. 128 pads 125-row arrays to 128 rows."
             " Flood/rainfall filled with 0; DEM filled with 100."
    )
    parser.add_argument(
        "--pad_w", type=int, default=0,
        help="Target width after padding (0 = no padding). E.g. 128 pads 125-col arrays to 128 cols."
    )
    parser.add_argument(
        "--splits", type=str, nargs="+", default=["train", "test"],
        choices=["train", "test"],
        help="Which splits to process (default: train test)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.spatial_factor < 1 or args.temporal_factor < 1:
        sys.exit("Error: spatial_factor and temporal_factor must be ≥ 1.")

    print(f"Source dataset : {args.src_root}")
    print(f"Output dataset : {args.dst_root}")
    print(f"Spatial factor : {args.spatial_factor}×  (block-average)")
    print(f"Temporal factor: {args.temporal_factor}×  (rainfall: sum; flood: sample)")
    if args.pad_h or args.pad_w:
        print(f"Spatial padding: → {args.pad_h}×{args.pad_w}  (zeros; DEM→100m)")
    print(f"Splits         : {args.splits}")
    print()

    for split in args.splits:
        split_src = os.path.join(args.src_root, split)
        split_dst = os.path.join(args.dst_root, split)

        if not os.path.isdir(split_src):
            print(f"[SKIP] Split not found: {split_src}")
            continue

        print(f"Processing split: {split} ...")
        process_split(split_src, split_dst, args.spatial_factor, args.temporal_factor,
                      pad_h=args.pad_h, pad_w=args.pad_w)
        print(f"  Done → {split_dst}")

    print("\nDownsampling complete.")
    print(f"Downsampled dataset saved to: {args.dst_root}")


if __name__ == "__main__":
    main()
