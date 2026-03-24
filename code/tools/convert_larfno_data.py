"""
convert_larfno_data.py — Convert LarNO benchmark data to U-RNN compatible format.

Purpose:
    The LarNO benchmark (https://github.com/holmescao/LarNO) stores flood data in a
    flat per-event structure.  U-RNN expects a split-based hierarchy with geodata
    separate from event data.  This script re-organises the data and fills in missing
    geodata channels (impervious surface, manhole mask) with zeros.

    NOTE for UKEA: the raw rainfall.npy is 1-min resolution (360 steps).  You MUST
    run ``tools/preprocess_ukea_rainfall.py`` first to aggregate rainfall to 5-min
    (36 steps) and keep only the 20 train/test events.  Point --larfno_root at the
    preprocessed output directory, not the raw LarNO source.

Supported datasets:
    futian  — Shenzhen Futian District (region1_20m), 20 m, 5 min, T=72, 400×560
    ukea    — UK urban catchment (ukea_8m_5min), 8 m, 5 min, T=36, 52×120
              (rainfall must already be aggregated to 5-min before running this script)

Input structure (LarNO benchmark, or preprocessed output for ukea):
    <larfno_root>/flood/<dataset>/
        <event>/
            h.npy        (T, H, W)   — water depth in metres
            rainfall.npy (T, H, W)   — rainfall intensity in mm/step
    <larfno_root>/geodata/<dataset>/
        dem.npy          (H, W)      — terrain elevation in metres

Output structure (U-RNN compatible):
    <dst_root>/
        train/
            flood/<location>/<event>/
                flood.npy      (T, H, W)  metres  [h.npy copied as-is; _prepare_target does m→mm internally]
                rainfall.npy   (T, H, W)  mm/step
            geodata/<location>/
                absolute_DEM.npy   (H, W)  metres
                impervious.npy     (H, W)  zeros
                manhole.npy        (H, W)  zeros
        test/
            (same structure)

Usage:
    # Step 1 (UKEA only): aggregate rainfall from 1-min to 5-min
    python tools/preprocess_ukea_rainfall.py \
        --src_dir /path/to/larFNO/benchmark/urbanflood \
        --dst_dir /tmp/ukea_preprocessed

    # Step 2: structural conversion
    python tools/convert_larfno_data.py --dataset futian \
        --larfno_root /path/to/larFNO/benchmark/urbanflood \
        --dst_root /root/autodl-tmp/data/larfno_futian

    python tools/convert_larfno_data.py --dataset ukea \
        --larfno_root /tmp/ukea_preprocessed \
        --dst_root /root/autodl-tmp/data/larfno_ukea
"""

import argparse
import os
import shutil

import numpy as np


# ─── Dataset-specific constants ────────────────────────────────────────────────

DATASET_CONFIGS = {
    "futian": {
        "src_flood_dir": "region1_20m",
        "src_geodata_dir": "region1_20m",
        "location_name": "region1",
        # Spatial padding: (H_pad_before, H_pad_after, W_pad_before, W_pad_after)
        "pad_H": (0, 0),   # 400 rows — already divisible by 4
        "pad_W": (0, 0),   # 560 cols — already divisible by 4
        # Temporal aggregation: None = keep as-is (5-min resolution, T=72)
        "temporal_agg": None,
        "train_events": [
            "event1", "event11", "event22", "event31",
            "event40", "event50", "event56", "event63",
        ],
        "test_events": [
            "event65", "event70", "event74", "event78",
        ],
    },
    "ukea": {
        "src_flood_dir": "ukea_8m_5min",
        "src_geodata_dir": "ukea_8m_5min",
        "location_name": "ukea",
        # Spatial padding: 50 → 52 rows (52 divisible by 4; 120 already divisible)
        "pad_H": (0, 2),   # pad 2 rows at bottom
        "pad_W": (0, 0),
        # Temporal aggregation: none — rainfall must already be at 5-min resolution
        # (36 steps) before running this script.  Use tools/preprocess_ukea_rainfall.py
        # to aggregate raw 1-min rainfall to 5-min as a one-time preprocessing step.
        "temporal_agg": None,
        "train_events": [
            "r100y_p0.1_d3h_1", "r100y_p0.6_d3h_1",
            "r200y_p0.3_d3h_1", "r200y_p0.7_d3h_1",
            "r300y_p0.4_d3h_1", "r300y_p0.9_d3h_1",
            "r500y_p0.2_d3h_1", "r500y_p0.6_d3h_1",
        ],
        "test_events": [
            "r100y_p0.5_d3h_1", "r100y_p0.7_d3h_1", "r100y_p0.8_d3h_1",
            "r200y_p0.5_d3h_1",
            "r300y_p0.1_d3h_1", "r300y_p0.5_d3h_1", "r300y_p0.6_d3h_1", "r300y_p0.8_d3h_1",
            "r500y_p0.1_d3h_1", "r500y_p0.3_d3h_1", "r500y_p0.4_d3h_1", "r500y_p0.9_d3h_1",
        ],
    },
}


# ─── Helpers ────────────────────────────────────────────────────────────────────

def _pad_spatial(arr, pad_H, pad_W, pad_value=0.0):
    """Zero-pad (or constant-pad) a spatial or spatio-temporal array.

    Parameters
    ----------
    arr : np.ndarray
        Shape (H, W) or (T, H, W).
    pad_H : tuple (before, after)
        Number of rows to pad at top and bottom.
    pad_W : tuple (before, after)
        Number of columns to pad at left and right.
    pad_value : float
        Fill value (0 for rainfall/flood; large value for DEM walls).
    """
    if arr.ndim == 2:   # (H, W)
        return np.pad(arr,
                      (pad_H, pad_W),
                      mode='constant', constant_values=pad_value)
    elif arr.ndim == 3:  # (T, H, W)
        return np.pad(arr,
                      ((0, 0), pad_H, pad_W),
                      mode='constant', constant_values=pad_value)
    else:
        raise ValueError(f"Unexpected array ndim: {arr.ndim}")


def _save_event(event_dir, flood, rainfall):
    """Save one event's flood and rainfall arrays."""
    os.makedirs(event_dir, exist_ok=True)
    np.save(os.path.join(event_dir, "flood.npy"),    flood.astype(np.float32))
    np.save(os.path.join(event_dir, "rainfall.npy"), rainfall.astype(np.float32))


def _save_geodata(geo_dir, dem, H, W):
    """Save static geodata: DEM, and zero-filled placeholders for impervious and manhole.

    Why zeros for impervious and manhole?
    --------------------------------------
    U-RNN's ``preprocess_inputs`` always concatenates three static spatial channels
    [DEM, impervious, manhole] into the model input (total C = nums×2 + 3).  All
    three files must exist on disk or the DataLoader will raise a KeyError.

    The LarNO benchmark does not provide impervious surface or drainage manhole data.
    Zero-filling is the physically reasonable default:
      • impervious = 0  →  "fully pervious" (maximum infiltration, conservative assumption)
      • manhole    = 0  →  "no drainage inlets" (no pipe network connectivity)

    These placeholders allow the model to train on LarNO data without code changes.
    """
    os.makedirs(geo_dir, exist_ok=True)
    np.save(os.path.join(geo_dir, "absolute_DEM.npy"), dem.astype(np.float32))
    np.save(os.path.join(geo_dir, "impervious.npy"),
            np.zeros((H, W), dtype=np.float32))
    np.save(os.path.join(geo_dir, "manhole.npy"),
            np.zeros((H, W), dtype=np.float32))
    print(f"  Geodata saved → {geo_dir}  (DEM: {dem.shape}, impervious/manhole: zeros)")


# ─── Main conversion ────────────────────────────────────────────────────────────

def convert_dataset(dataset, larfno_root, dst_root, dry_run=False):
    """Convert a single LarNO dataset to U-RNN format.

    Parameters
    ----------
    dataset : str
        One of ``"futian"`` or ``"ukea"``.
    larfno_root : str
        Path to the LarNO ``benchmark/urbanflood/`` directory.
    dst_root : str
        Destination root for the converted dataset.
    dry_run : bool
        If True, print actions without writing files.
    """
    cfg = DATASET_CONFIGS[dataset]

    src_flood_root       = os.path.join(larfno_root, "flood", cfg["src_flood_dir"])
    src_geo_root         = os.path.join(larfno_root, "geodata", cfg["src_geodata_dir"])
    location             = cfg["location_name"]
    pad_H, pad_W         = cfg["pad_H"], cfg["pad_W"]

    # ── Load and process DEM ─────────────────────────────────────────────────
    dem_src = np.load(os.path.join(src_geo_root, "dem.npy"))
    # Pad DEM with its max value (acts as a wall / barrier) in added pixels
    dem_wall_value = float(dem_src.max())
    dem = _pad_spatial(dem_src, pad_H, pad_W, pad_value=dem_wall_value)
    H, W = dem.shape
    print(f"\n[{dataset}]  DEM: {dem_src.shape} → padded {dem.shape}  (wall={dem_wall_value:.1f} m)")

    # ── Process each split ───────────────────────────────────────────────────
    for split, event_list in [("train", cfg["train_events"]),
                               ("test",  cfg["test_events"])]:

        # Save geodata once per split
        geo_dst = os.path.join(dst_root, split, "geodata", location)
        if not dry_run:
            _save_geodata(geo_dst, dem, H, W)

        # Save each event
        for event_name in event_list:
            src_event = os.path.join(src_flood_root, event_name)
            if not os.path.isdir(src_event):
                print(f"  [WARN] Missing event: {src_event}")
                continue

            h_path   = os.path.join(src_event, "h.npy")
            r_path   = os.path.join(src_event, "rainfall.npy")

            if not os.path.exists(h_path) or not os.path.exists(r_path):
                print(f"  [WARN] Incomplete event (missing h.npy or rainfall.npy): {src_event}")
                continue

            flood    = np.load(h_path).astype(np.float64)             # metres (kept as-is; _prepare_target converts m→mm)
            rainfall = np.load(r_path).astype(np.float64)    # (T, H_src, W_src) mm/step

            # Spatial padding
            flood    = _pad_spatial(flood,    pad_H, pad_W, pad_value=0.0)
            rainfall = _pad_spatial(rainfall, pad_H, pad_W, pad_value=0.0)

            dst_event = os.path.join(dst_root, split, "flood", location, event_name)
            print(f"  {split}/{event_name}: flood {flood.shape}  rainfall {rainfall.shape}"
                  f"  → {dst_event}")

            if not dry_run:
                _save_event(dst_event, flood, rainfall)

    print(f"\n[{dataset}]  Conversion complete → {dst_root}")


# ─── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert LarNO benchmark data to U-RNN compatible format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", required=True, choices=["futian", "ukea"],
        help="Which LarNO dataset to convert.",
    )
    parser.add_argument(
        "--larfno_root", required=True,
        help="Path to the LarNO benchmark/urbanflood/ directory.",
    )
    parser.add_argument(
        "--dst_root", required=True,
        help="Output root directory for the converted U-RNN dataset.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print file operations without actually writing files.",
    )
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY RUN] No files will be written.\n")

    convert_dataset(
        dataset=args.dataset,
        larfno_root=args.larfno_root,
        dst_root=args.dst_root,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
