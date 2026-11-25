from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
TLX_COLS = [
    "tlx_mental",
    "tlx_physical",
    "tlx_temporal",
    "tlx_performance",
    "tlx_effort",
    "tlx_frustration",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze paired HMI pilot study CSV.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT / "eval" / "hmi_pilot_results.csv",
        help="Path to recorded pilot CSV (defaults to eval/hmi_pilot_results.csv).",
    )
    args = parser.parse_args()

    csv_path = args.csv
    if not csv_path.exists():
        print(f"No recorded results at {csv_path}. Copy eval/hmi_pilot_template.csv after running pilot sessions.")
        return

    df = pd.read_csv(csv_path)
    df["nasa_tlx_raw"] = df[TLX_COLS].mean(axis=1)

    piv = df.pivot(index="participant_id", columns="condition")
    metrics = ["time_sec", "accuracy_score", "trust_calibrated", "nasa_tlx_raw", "tlx_mental", "tlx_frustration"]

    rows = []
    for m in metrics:
        std = piv[(m, "standard")].dropna()
        cop = piv[(m, "chromcopilot")].dropna()
        common = std.index.intersection(cop.index)
        a = std.loc[common].to_numpy(dtype=np.float64)
        b = cop.loc[common].to_numpy(dtype=np.float64)
        diff = b - a
        n = len(common)

        p_val = None
        if n >= 4 and np.any(diff != 0):
            p_val = round(float(stats.wilcoxon(b, a).pvalue), 4)
        sd_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0
        dz = round(float(np.mean(diff) / sd_diff), 3) if sd_diff > 0 else None

        rows.append(
            {
                "metric": m,
                "n_paired": int(n),
                "standard_mean": round(float(np.mean(a)), 2),
                "copilot_mean": round(float(np.mean(b)), 2),
                "mean_delta": round(float(np.mean(diff)), 2),
                "cohens_dz": dz,
                "wilcoxon_p": p_val,
            }
        )

    out_dir = REPO_ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "hmi_pilot_summary.csv", index=False)
    (out_dir / "hmi_pilot_summary.json").write_text(json.dumps(rows, indent=2))

    print("=== Pilot HMI User Study Summary (Paired Within-Subjects) ===")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()