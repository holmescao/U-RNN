"""
tools/preprocess_ukea_rainfall.py — One-time UKEA rainfall pre-processing.

Background
----------
The LarNO UKEA dataset (ukea_8m_5min) stores rainfall at 1-min resolution
(360 time steps), but the flood simulation output h.npy is at 5-min resolution
(36 time steps covering the 3-hour active rain period).

To align the two:
  1. Crop rainfall to the first 180 steps  (180 min × 1 min/step = 3 h active rain).
  2. Sum every 5 consecutive 1-min values  → 36 steps at 5-min resolution.
     Each aggregated value = total rainfall [mm] accumulated over 5 minutes.
  3. Keep only the 20 canonical events (8 train + 12 test); delete all others.

This produces a "cleaned" copy of the UKEA flood directory where:
  • h.npy        (36, H, W)  — unchanged (already 5-min)
  • rainfall.npy (36, H, W)  — aggregated from 360×1-min to 36×5-min

After running this script, point convert_larfno_data.py at --larfno_root <dst_dir>
to complete the U-RNN structural conversion (rename, geodata, spatial padding).

Usage
-----
    python tools/preprocess_ukea_rainfall.py \
        --src_dir /path/to/larFNO/benchmark/urbanflood \
        --dst_dir /tmp/ukea_preprocessed

    # Then convert to U-RNN format:
    python tools/convert_larfno_data.py --dataset ukea \
        --larfno_root /tmp/ukea_preprocessed \
        --dst_root /root/autodl-tmp/data/larfno_ukea

Arguments
---------
--src_dir   Root of the LarNO benchmark/urbanflood/ directory.
            Expected sub-path: flood/ukea_8m_5min/<event>/{h.npy, rainfall.npy}
                               geodata/ukea_8m_5min/dem.npy
--dst_dir   Output root (same sub-path structure, but rainfall aggregated).
            The script writes only the 20 canonical events.
--dry_run   Print planned actions without writing any files.
"""

import argparse
import os
import shutil

import numpy as np


# ── Canonical event lists (8 train + 12 test = 20 total) ────────────────────

TRAIN_EVENTS = [
    "r100y_p0.1_d3h_1", "r100y_p0.6_d3h_1",
    "r200y_p0.3_d3h_1", "r200y_p0.7_d3h_1",
    "r300y_p0.4_d3h_1", "r300y_p0.9_d3h_1",
    "r500y_p0.2_d3h_1", "r500y_p0.6_d3h_1",
]

TEST_EVENTS = [
    "r100y_p0.5_d3h_1", "r100y_p0.7_d3h_1", "r100y_p0.8_d3h_1",
    "r200y_p0.5_d3h_1",
    "r300y_p0.1_d3h_1", "r300y_p0.5_d3h_1", "r300y_p0.6_d3h_1", "r300y_p0.8_d3h_1",
    "r500y_p0.1_d3h_1", "r500y_p0.3_d3h_1", "r500y_p0.4_d3h_1", "r500y_p0.9_d3h_1",
]

KEEP_EVENTS = set(TRAIN_EVENTS + TEST_EVENTS)

FLOOD_SUBDIR   = os.path.join("flood",   "ukea_8m_5min")
GEODATA_SUBDIR = os.path.join("geodata", "ukea_8m_5min")

# Temporal aggregation constants
RAINFALL_ACTIVE_STEPS = 180   # raw 1-min steps that contain non-zero rain (= 3 h)
TEMPORAL_AGG_FACTOR   = 5     # 1-min → 5-min; 180 / 5 = 36 steps


# ── Core helper ──────────────────────────────────────────────────────────────

def aggregate_rainfall(arr):
    """Crop and sum-aggregate raw UKEA 1-min rainfall to 5-min.

    Parameters
    ----------
    arr : np.ndarray, shape (360, H, W)
        Raw rainfall in mm/min at 1-min resolution.

    Returns
    -------
    np.ndarray, shape (36, H, W)
        Aggregated rainfall in mm/5min.
    """
    assert arr.ndim == 3, f"Expected (T, H, W), got {arr.shape}"
    assert arr.shape[0] >= RAINFALL_ACTIVE_STEPS, (
        f"rainfall.npy has only {arr.shape[0]} steps, expected ≥ {RAINFALL_ACTIVE_STEPS}"
    )
    active = arr[:RAINFALL_ACTIVE_STEPS]          # (180, H, W)
    T_new = RAINFALL_ACTIVE_STEPS // TEMPORAL_AGG_FACTOR  # 36
    _, H, W = active.shape
    return active.reshape(T_new, TEMPORAL_AGG_FACTOR, H, W).sum(axis=1)  # (36, H, W)


# ── Main ─────────────────────────────────────────────────────────────────────

def preprocess(src_dir, dst_dir, dry_run=False):
    src_flood   = os.path.join(src_dir, FLOOD_SUBDIR)
    src_geodata = os.path.join(src_dir, GEODATA_SUBDIR)
    dst_flood   = os.path.join(dst_dir, FLOOD_SUBDIR)
    dst_geodata = os.path.join(dst_dir, GEODATA_SUBDIR)

    # ── Enumerate source events ───────────────────────────────────────────
    if not os.path.isdir(src_flood):
        raise FileNotFoundError(f"Source flood directory not found: {src_flood}")

    all_events = sorted(os.listdir(src_flood))
    kept   = [e for e in all_events if e in KEEP_EVENTS]
    skipped = [e for e in all_events if e not in KEEP_EVENTS]

    print(f"Source : {src_flood}")
    print(f"Output : {dst_flood}")
    print(f"Events : {len(kept)} kept / {len(skipped)} skipped (not in canonical list)")
    if skipped:
        print(f"  Skipped: {skipped}")
    print()

    missing = KEEP_EVENTS - set(all_events)
    if missing:
        print(f"[WARN] {len(missing)} canonical events missing from source: {sorted(missing)}")

    # ── Process each kept event ───────────────────────────────────────────
    for event in kept:
        src_event = os.path.join(src_flood, event)
        dst_event = os.path.join(dst_flood, event)

        h_src = os.path.join(src_event, "h.npy")
        r_src = os.path.join(src_event, "rainfall.npy")

        if not os.path.exists(h_src) or not os.path.exists(r_src):
            print(f"  [WARN] Incomplete event (missing h.npy or rainfall.npy): {src_event}")
            continue

        rainfall_raw = np.load(r_src).astype(np.float64)  # (360, H, W)
        rainfall_agg = aggregate_rainfall(rainfall_raw)    # (36, H, W)

        flood = np.load(h_src)  # (36, H, W) — pass through unchanged

        print(f"  {event}: rainfall {rainfall_raw.shape} → {rainfall_agg.shape}  "
              f"flood {flood.shape}")

        if not dry_run:
            os.makedirs(dst_event, exist_ok=True)
            np.save(os.path.join(dst_event, "h.npy"),        flood.astype(np.float32))
            np.save(os.path.join(dst_event, "rainfall.npy"), rainfall_agg.astype(np.float32))

    # ── Copy geodata unchanged ────────────────────────────────────────────
    dem_src = os.path.join(src_geodata, "dem.npy")
    dem_dst = os.path.join(dst_geodata, "dem.npy")
    if os.path.exists(dem_src):
        print(f"\nCopying geodata: {dem_src} → {dem_dst}")
        if not dry_run:
            os.makedirs(dst_geodata, exist_ok=True)
            shutil.copy2(dem_src, dem_dst)
    else:
        print(f"[WARN] geodata/dem.npy not found at {dem_src}")

    print(f"\nPreprocessing complete → {dst_dir}")
    print(f"Next step: python tools/convert_larfno_data.py --dataset ukea "
          f"--larfno_root {dst_dir} --dst_root <u-rnn-data-dir>")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate UKEA 1-min rainfall to 5-min and keep only canonical events.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--src_dir", required=True,
        help="Path to the LarNO benchmark/urbanflood/ directory (raw source).",
    )
    parser.add_argument(
        "--dst_dir", required=True,
        help="Output directory for the preprocessed data (same sub-path structure).",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print planned actions without writing any files.",
    )
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY RUN] No files will be written.\n")

    preprocess(src_dir=args.src_dir, dst_dir=args.dst_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
