from __future__ import annotations

import json
import time
from pathlib import Path

import cooler
import hictkpy
import numpy as np
import pandas as pd
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
OUT_DIR = REPO_ROOT / "data" / "processed" / "contacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_FILES = {
    "GM12878": RAW_DIR / "GSE63525_GM12878_insitu_primary+replicate_combined.hic",
    "K562": RAW_DIR / "GSE63525_K562_combined.hic",
}

TARGET_CHROMS = ["chr3", "chr8", "chr11", "chr18"]
RESOLUTIONS = [50_000, 100_000, 250_000]

# hg19 2.5 Mb target windows around MYC, HBB/LCR, BCL2, and GATA2
TARGET_LOCI = {
    "MYC": {"chrom": "chr8", "start": 127_500_000, "end": 130_000_000},
    "HBB": {"chrom": "chr11", "start": 4_000_000, "end": 6_500_000},
    "BCL2": {"chrom": "chr18", "start": 59_500_000, "end": 62_000_000},
    "GATA2": {"chrom": "chr3", "start": 127_000_000, "end": 129_500_000},
}


def resolve_hic_chrom_name(avail_chroms: dict[str, int], ucsc_chrom: str) -> str:
    """Match either 'chr8' or bare '8' in a .hic chromosome table."""
    if ucsc_chrom in avail_chroms:
        return ucsc_chrom
    bare = ucsc_chrom.removeprefix("chr")
    if bare in avail_chroms:
        return bare
    raise KeyError(f"Chromosome {ucsc_chrom} (or {bare}) not found in .hic file.")


def convert_hic_to_mcool(entity_id: str, hic_path: Path, out_mcool: Path) -> dict:
    if not hic_path.exists():
        raise FileNotFoundError(f"Missing raw .hic file: {hic_path}")

    if out_mcool.exists():
        out_mcool.unlink()

    sparsity_stats: dict[str, dict[str, dict[str, float]]] = {}

    for res in RESOLUTIONS:
        t0 = time.perf_counter()
        print(f"[{entity_id}] Extracting {TARGET_CHROMS} at {res // 1000} kb from {hic_path.name}...")
        hf = hictkpy.File(str(hic_path), res)
        avail = hf.chromosomes()

        chrom_sizes: dict[str, int] = {}
        raw_name_map: dict[str, str] = {}
        for ucsc in TARGET_CHROMS:
            raw_name = resolve_hic_chrom_name(avail, ucsc)
            raw_name_map[ucsc] = raw_name
            chrom_sizes[ucsc] = int(avail[raw_name])

        # Build canonical bin table with UCSC chromosome names
        chromsizes_series = pd.Series(chrom_sizes)
        bins_df = cooler.util.binnify(chromsizes_series, res)

        # Compute starting global bin index per chromosome
        chrom_offsets: dict[str, int] = {}
        for ucsc in TARGET_CHROMS:
            chrom_offsets[ucsc] = int(bins_df.index[bins_df["chrom"] == ucsc][0])

        pixel_chunks: list[pd.DataFrame] = []
        for ucsc in TARGET_CHROMS:
            raw_name = raw_name_map[ucsc]
            sel = hf.fetch(raw_name)
            mat = sel.to_csr()
            upper = sparse.triu(mat, k=0).tocoo()

            mask = np.isfinite(upper.data) & (upper.data > 0)
            offset = chrom_offsets[ucsc]
            chunk = pd.DataFrame(
                {
                    "bin1_id": upper.row[mask].astype(np.int64) + offset,
                    "bin2_id": upper.col[mask].astype(np.int64) + offset,
                    "count": upper.data[mask].astype(np.float64),
                }
            )
            pixel_chunks.append(chunk)

        pixels_df = pd.concat(pixel_chunks, ignore_index=True)
        pixels_df.sort_values(["bin1_id", "bin2_id"], inplace=True, ignore_index=True)

        cool_uri = f"{out_mcool}::/resolutions/{res}"
        cooler.create_cooler(
            cool_uri,
            bins=bins_df,
            pixels=pixels_df,
            dtypes={"count": np.float64},
            assembly="hg19",
            mode="a",
            ordered=True,
        )

        # ICE balance cis contacts so cooltools (insulation & compartments) runs natively
        clr = cooler.Cooler(cool_uri)
        try:
            cooler.balance_cooler(clr, cis_only=True, store=True, store_name="weight")
        except Exception as exc:
            print(f"  Warning: standard ICE failed ({exc}); retrying with relaxed mad_max=0...")
            cooler.balance_cooler(clr, cis_only=True, mad_max=0, store=True, store_name="weight")

        # Measure baseline sparsity in each 2.5 Mb target locus
        res_key = str(res)
        sparsity_stats[res_key] = {}
        for locus_name, loc in TARGET_LOCI.items():
            region = f"{loc['chrom']}:{loc['start']}-{loc['end']}"
            sub = clr.matrix(balance=False).fetch(region)
            n = sub.shape[0]
            tri_idx = np.triu_indices(n, k=0)
            vals = sub[tri_idx]
            zero_frac = float(np.mean(vals <= 0))
            sparsity_stats[res_key][locus_name] = {
                "n_bins": int(n),
                "total_contacts": float(np.nansum(vals)),
                "sparsity_fraction": round(zero_frac, 4),
            }

        elapsed = time.perf_counter() - t0
        print(f"  Done {res // 1000} kb: {len(pixels_df):,} upper-triangle pixels in {elapsed:.1f}s")

    return sparsity_stats


def main() -> None:
    summary: dict[str, dict] = {}
    for entity_id, hic_path in RAW_FILES.items():
        out_mcool = OUT_DIR / f"{entity_id}.mcool"
        summary[entity_id] = convert_hic_to_mcool(entity_id, hic_path, out_mcool)

    summary_path = OUT_DIR / "sparsity_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved baseline sparsity summary to {summary_path}")


if __name__ == "__main__":
    main()