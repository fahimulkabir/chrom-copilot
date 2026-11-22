from __future__ import annotations

import json
import time
from pathlib import Path

import cooler
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def main() -> None:
    meta_path = PROCESSED_DIR / "metadata.json"
    assert meta_path.exists(), f"Missing {meta_path}"
    meta = json.loads(meta_path.read_text())

    assert meta["dataset_id"] == "rao2014_gm12878_k562"
    assert meta["genome_build"] == "hg19"
    resolutions = meta["available_resolutions_bp"]

    # Load Parquet tables once (mirrors Part 2 in-memory data_store.py)
    df_3d = pd.read_parquet(PROCESSED_DIR / meta["structures_3d_path"])
    df_1d_bins = pd.read_parquet(PROCESSED_DIR / meta["tracks_1d"]["bin_signals_path"])
    df_genes = pd.read_parquet(PROCESSED_DIR / meta["tracks_1d"]["genes_path"])
    assert not df_3d.empty and not df_1d_bins.empty and not df_genes.empty

    # 1. Validate .mcool integrity, UCSC chrom names, and ICE weights
    coolers: dict[tuple[str, int], cooler.Cooler] = {}
    for cid, cinfo in meta["cohorts"].items():
        mcool_path = PROCESSED_DIR / cinfo["mcool_path"]
        assert mcool_path.exists(), f"Missing {mcool_path}"
        for res in resolutions:
            clr = cooler.Cooler(f"{mcool_path}::/resolutions/{res}")
            assert all(c.startswith("chr") for c in clr.chromnames), f"Non-UCSC chrom in {mcool_path}"
            assert "weight" in clr.bins().columns, f"Missing ICE weight column in {cid} @ {res}"
            coolers[(cid, res)] = clr

    # 2. Validate exact bin-index alignment across 1D, 2D, and 3D
    mismatches = 0
    for res in resolutions:
        sub_3d = df_3d[df_3d["resolution_bp"] == res]
        sub_1d = df_1d_bins[df_1d_bins["resolution_bp"] == res]

        assert (sub_3d["start"] == sub_3d["bin_index"] * res).all(), f"3D bin_index mismatch at {res}"
        assert (sub_1d["start"] == sub_1d["bin_index"] * res).all(), f"1D bin_index mismatch at {res}"

        for cid in meta["cohorts"]:
            clr = coolers[(cid, res)]
            for locus_name, loc in meta["target_loci"].items():
                region = f"{loc['chrom']}:{loc['start']}-{loc['end']}"
                bins_2d = clr.bins().fetch(region)
                expected_indices = set((bins_2d["start"] // res).astype(int).tolist())

                mask_3d = (
                    (sub_3d["entity_id"] == cid)
                    & (sub_3d["chrom"] == loc["chrom"])
                    & (sub_3d["start"] >= loc["start"])
                    & (sub_3d["end"] <= loc["end"])
                )
                found_3d = set(sub_3d.loc[mask_3d, "bin_index"].astype(int).tolist())

                mask_1d = (
                    (sub_1d["chrom"] == loc["chrom"])
                    & (sub_1d["start"] >= loc["start"])
                    & (sub_1d["end"] <= loc["end"])
                )
                found_1d = set(sub_1d.loc[mask_1d, "bin_index"].astype(int).tolist())

                if expected_indices != found_3d or expected_indices != found_1d:
                    mismatches += 1
                    print(
                        f"MISMATCH {cid} {locus_name} @{res}: "
                        f"2D={len(expected_indices)}, 3D={len(found_3d)}, 1D={len(found_1d)}"
                    )

    assert mismatches == 0, f"Found {mismatches} bin alignment mismatches!"
    print("[PASS] Zero bin-index mismatches across 1D tracks, 2D .mcool, and 3D Parquet.")

    # 3. Benchmark locus load latency (< 200 ms target)
    latencies_ms: list[float] = []
    for locus_name, loc in meta["target_loci"].items():
        region = f"{loc['chrom']}:{loc['start']}-{loc['end']}"
        t0 = time.perf_counter()
        _mat = coolers[("GM12878", 50_000)].matrix(balance=True).fetch(region)
        _c3d = df_3d[
            (df_3d["entity_id"] == "GM12878")
            & (df_3d["resolution_bp"] == 50_000)
            & (df_3d["chrom"] == loc["chrom"])
            & (df_3d["start"] >= loc["start"])
            & (df_3d["end"] <= loc["end"])
        ]
        _t1d = df_1d_bins[
            (df_1d_bins["resolution_bp"] == 50_000)
            & (df_1d_bins["chrom"] == loc["chrom"])
            & (df_1d_bins["start"] >= loc["start"])
            & (df_1d_bins["end"] <= loc["end"])
        ]
        ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(ms)
        print(f"  Locus {locus_name:5s} (50 kb): 2D={_mat.shape}, 3D={len(_c3d)} beads, 1D={len(_t1d)} bins -> {ms:.1f} ms")

    max_ms = max(latencies_ms)
    assert max_ms < 200.0, f"Locus load exceeded 200 ms budget ({max_ms:.1f} ms)"
    print(f"[PASS] Max locus load latency: {max_ms:.1f} ms (under 200 ms Exit Gate).")
    print("\nPART 1 EXIT GATE PASSED.")


if __name__ == "__main__":
    main()