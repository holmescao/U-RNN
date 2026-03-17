"""
tools/compare_metrics.py — U-RNN vs LarNO benchmark comparison

Reads U-RNN test metrics from Excel output files produced by test.py, merges
them with published LarNO numbers (hard-coded below), and prints a paper-style
LaTeX table.

Usage
-----
    # Compare on Futian dataset:
    python tools/compare_metrics.py \
        --urnn_xlsx ../exp/<timestamp>/metrics/metrics_epoch<N>.xlsx \
        --dataset   futian

    # Compare on UKEA dataset:
    python tools/compare_metrics.py \
        --urnn_xlsx ../exp/<timestamp>/metrics/metrics_epoch<N>.xlsx \
        --dataset   ukea

    # Also save a CSV copy:
    python tools/compare_metrics.py \
        --urnn_xlsx ../exp/<timestamp>/metrics/metrics_epoch<N>.xlsx \
        --dataset   futian \
        --save_csv  ../exp/<timestamp>/metrics/comparison_futian.csv

Outputs
-------
    - Tabulated comparison (rich table in terminal)
    - Optional CSV for further analysis
    - LaTeX table snippet (stdout, copy-pasteable into paper)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Published LarNO metrics (from LarNO paper / benchmark)
# Source: Shen et al. (2024) — fill in / update from the paper as needed.
# Keys  : dataset name ("futian" / "ukea")
# Values: dict mapping event_name → {"R2": float, "RMSE": float, "CSI": float}
# ──────────────────────────────────────────────────────────────────────────────
LARFNO_METRICS = {
    "futian": {
        # TODO: fill in from LarNO paper after obtaining their test metrics
        # Format: "event_name": {"R2": ..., "RMSE_m": ..., "CSI": ...}
        "event65": {"R2": None, "RMSE_m": None, "CSI": None},
        "event70": {"R2": None, "RMSE_m": None, "CSI": None},
        "event74": {"R2": None, "RMSE_m": None, "CSI": None},
        "event78": {"R2": None, "RMSE_m": None, "CSI": None},
    },
    "ukea": {
        # TODO: fill in from LarNO paper after obtaining their test metrics
        "r100y_d3h":  {"R2": None, "RMSE_m": None, "CSI": None},
        "r200y_d3h":  {"R2": None, "RMSE_m": None, "CSI": None},
        "r300y_d3h":  {"R2": None, "RMSE_m": None, "CSI": None},
        "r500y_d3h":  {"R2": None, "RMSE_m": None, "CSI": None},
    },
}


def load_urnn_metrics(xlsx_path: str) -> pd.DataFrame:
    """Load per-event metrics produced by test.py into a DataFrame."""
    if not os.path.isfile(xlsx_path):
        sys.exit(f"[ERROR] File not found: {xlsx_path}")
    df = pd.read_excel(xlsx_path)
    # Normalise column names: strip whitespace, lower-case
    df.columns = [c.strip() for c in df.columns]
    # Expected columns: Event, R2, RMSE(m), MAE(m), PeakR2, CSI
    if "Event" not in df.columns:
        sys.exit(f"[ERROR] 'Event' column not found. Columns: {list(df.columns)}")
    df = df.rename(columns={"RMSE(m)": "RMSE_m", "MAE(m)": "MAE_m"})
    df["Event"] = df["Event"].astype(str).str.strip()
    return df


def build_comparison(urnn_df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Merge U-RNN metrics with LarNO published numbers."""
    if dataset not in LARFNO_METRICS:
        sys.exit(f"[ERROR] Unknown dataset '{dataset}'. Choose from: {list(LARFNO_METRICS)}")

    larfno_ref = LARFNO_METRICS[dataset]
    rows = []
    for _, row in urnn_df.iterrows():
        event = row["Event"]
        r2_urnn    = row.get("R2",      np.nan)
        rmse_urnn  = row.get("RMSE_m",  np.nan)
        csi_urnn   = row.get("CSI",     np.nan)

        ref = larfno_ref.get(event, {})
        r2_lf   = ref.get("R2",     np.nan)
        rmse_lf = ref.get("RMSE_m", np.nan)
        csi_lf  = ref.get("CSI",    np.nan)

        rows.append({
            "Event":        event,
            "U-RNN R²":    r2_urnn,
            "LarNO R²":   r2_lf,
            "ΔR²":         r2_urnn - r2_lf if not np.isnan(r2_lf) else np.nan,
            "U-RNN RMSE":  rmse_urnn,
            "LarNO RMSE": rmse_lf,
            "ΔRMSE":       rmse_urnn - rmse_lf if not np.isnan(rmse_lf) else np.nan,
            "U-RNN CSI":   csi_urnn,
            "LarNO CSI":  csi_lf,
            "ΔCSI":        csi_urnn - csi_lf if not np.isnan(csi_lf) else np.nan,
        })

    # Append mean row
    cmp = pd.DataFrame(rows)
    mean_row = cmp.select_dtypes(include=np.number).mean().to_dict()
    mean_row["Event"] = "Mean"
    cmp = pd.concat([cmp, pd.DataFrame([mean_row])], ignore_index=True)
    return cmp


def print_table(cmp: pd.DataFrame, dataset: str) -> None:
    """Print a formatted comparison table."""
    print(f"\n{'='*72}")
    print(f"  U-RNN vs LarNO — {dataset.upper()} dataset")
    print(f"{'='*72}")

    def fmt(v, sign=False):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "   N/A  "
        prefix = "+" if sign and v > 0 else ""
        return f"{prefix}{v:+.4f}" if sign else f"{v:.4f}"

    header = (f"{'Event':<22} {'U-RNN R²':>9} {'LarNO R²':>10} {'ΔR²':>8} "
              f"{'U-RNN RMSE':>11} {'LarNO RMSE':>12} {'ΔRMSE':>8} "
              f"{'U-RNN CSI':>10} {'LarNO CSI':>11} {'ΔCSI':>7}")
    print(header)
    print("-" * len(header))
    for _, row in cmp.iterrows():
        is_mean = row["Event"] == "Mean"
        line = (
            f"{'** ' + row['Event'] + ' **' if is_mean else row['Event']:<22} "
            f"{fmt(row['U-RNN R²']):>9} {fmt(row['LarNO R²']):>10} {fmt(row['ΔR²'], sign=True):>8} "
            f"{fmt(row['U-RNN RMSE']):>11} {fmt(row['LarNO RMSE']):>12} {fmt(row['ΔRMSE'], sign=True):>8} "
            f"{fmt(row['U-RNN CSI']):>10} {fmt(row['LarNO CSI']):>11} {fmt(row['ΔCSI'], sign=True):>7}"
        )
        if is_mean:
            print("-" * len(header))
        print(line)
    print("=" * 72)


def print_latex(cmp: pd.DataFrame, dataset: str) -> None:
    """Print a LaTeX table snippet."""
    print(f"\n% ── LaTeX table: U-RNN vs LarNO ({dataset}) ──────────────")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{U-RNN vs.\ LarNO on " + dataset.upper() + r" dataset}")
    print(r"\begin{tabular}{lcccccc}")
    print(r"\toprule")
    print(r"Event & \multicolumn{2}{c}{R\textsuperscript{2}} & "
          r"\multicolumn{2}{c}{RMSE (m)} & \multicolumn{2}{c}{CSI} \\")
    print(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}")
    print(r" & U-RNN & LarNO & U-RNN & LarNO & U-RNN & LarNO \\")
    print(r"\midrule")

    def lf(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "--"
        return f"{v:.4f}"

    for _, row in cmp.iterrows():
        event = row["Event"]
        if event == "Mean":
            print(r"\midrule")
            event = r"\textbf{Mean}"
        print(f"{event} & {lf(row['U-RNN R²'])} & {lf(row['LarNO R²'])} & "
              f"{lf(row['U-RNN RMSE'])} & {lf(row['LarNO RMSE'])} & "
              f"{lf(row['U-RNN CSI'])} & {lf(row['LarNO CSI'])} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare U-RNN vs LarNO metrics on Futian or UKEA datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--urnn_xlsx", required=True,
                        help="Path to metrics_epoch<N>.xlsx produced by test.py.")
    parser.add_argument("--dataset",   required=True, choices=["futian", "ukea"],
                        help="Dataset name for LarNO reference lookup.")
    parser.add_argument("--save_csv",  default="",
                        help="Optional path to save the comparison as CSV.")
    parser.add_argument("--latex",     action="store_true",
                        help="Also print a LaTeX table snippet.")
    args = parser.parse_args()

    urnn_df = load_urnn_metrics(args.urnn_xlsx)
    cmp = build_comparison(urnn_df, args.dataset)
    print_table(cmp, args.dataset)

    if args.save_csv:
        cmp.to_csv(args.save_csv, index=False)
        print(f"\nSaved CSV → {args.save_csv}")

    if args.latex:
        print_latex(cmp, args.dataset)


if __name__ == "__main__":
    main()
