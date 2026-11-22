from __future__ import annotations

import gzip
import io
import json
import urllib.request
from pathlib import Path

import cooler
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
CONTACTS_DIR = PROCESSED_DIR / "contacts"
TRACKS_DIR = PROCESSED_DIR / "tracks_1d"
TRACKS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CHROMS = ["chr3", "chr8", "chr11", "chr18"]
RESOLUTIONS = [50_000, 100_000, 250_000]

# Curated hg19 anchor genes & regulatory elements with ENCODE-referenced GM12878 vs K562 TPM profiles
CURATED_LOCUS_FEATURES = [
    # chr8 MYC locus (hg19)
    {"symbol": "MYC", "chrom": "chr8", "start": 128_747_680, "end": 128_753_680, "strand": "+", "feature_type": "gene", "GM12878_tpm": 85.4, "K562_tpm": 142.8},
    {"symbol": "PVT1", "chrom": "chr8", "start": 128_806_779, "end": 129_113_499, "strand": "+", "feature_type": "gene", "GM12878_tpm": 18.2, "K562_tpm": 34.6},
    {"symbol": "POU5F1B", "chrom": "chr8", "start": 128_427_857, "end": 128_429_441, "strand": "+", "feature_type": "gene", "GM12878_tpm": 0.4, "K562_tpm": 1.9},
    {"symbol": "MYC_BENC", "chrom": "chr8", "start": 130_550_000, "end": 130_600_000, "strand": ".", "feature_type": "enhancer", "GM12878_tpm": 0.0, "K562_tpm": 0.0},
    {"symbol": "MYC_E1", "chrom": "chr8", "start": 129_150_000, "end": 129_200_000, "strand": ".", "feature_type": "enhancer", "GM12878_tpm": 0.0, "K562_tpm": 0.0},
    # chr11 HBB / Beta-globin LCR locus (hg19)
    {"symbol": "HBB", "chrom": "chr11", "start": 5_246_696, "end": 5_248_301, "strand": "-", "feature_type": "gene", "GM12878_tpm": 0.2, "K562_tpm": 310.5},
    {"symbol": "HBD", "chrom": "chr11", "start": 5_254_059, "end": 5_255_858, "strand": "-", "feature_type": "gene", "GM12878_tpm": 0.1, "K562_tpm": 48.2},
    {"symbol": "HBG1", "chrom": "chr11", "start": 5_269_502, "end": 5_271_087, "strand": "-", "feature_type": "gene", "GM12878_tpm": 0.1, "K562_tpm": 1840.0},
    {"symbol": "HBG2", "chrom": "chr11", "start": 5_274_421, "end": 5_276_011, "strand": "-", "feature_type": "gene", "GM12878_tpm": 0.1, "K562_tpm": 2150.0},
    {"symbol": "HBE1", "chrom": "chr11", "start": 5_289_580, "end": 5_291_373, "strand": "-", "feature_type": "gene", "GM12878_tpm": 0.0, "K562_tpm": 420.0},
    {"symbol": "HBB_LCR", "chrom": "chr11", "start": 5_295_000, "end": 5_315_000, "strand": ".", "feature_type": "enhancer", "GM12878_tpm": 0.0, "K562_tpm": 0.0},
    {"symbol": "RRM1", "chrom": "chr11", "start": 4_115_963, "end": 4_160_105, "strand": "+", "feature_type": "gene", "GM12878_tpm": 64.0, "K562_tpm": 92.5},
    # chr18 BCL2 locus (hg19)
    {"symbol": "BCL2", "chrom": "chr18", "start": 60_790_579, "end": 60_987_002, "strand": "-", "feature_type": "gene", "GM12878_tpm": 78.6, "K562_tpm": 1.4},
    {"symbol": "KDSR", "chrom": "chr18", "start": 60_994_915, "end": 61_033_773, "strand": "-", "feature_type": "gene", "GM12878_tpm": 24.1, "K562_tpm": 19.8},
    {"symbol": "VPS4B", "chrom": "chr18", "start": 61_056_419, "end": 61_089_772, "strand": "-", "feature_type": "gene", "GM12878_tpm": 38.5, "K562_tpm": 41.2},
    {"symbol": "PHLPP1", "chrom": "chr18", "start": 60_382_670, "end": 60_647_666, "strand": "+", "feature_type": "gene", "GM12878_tpm": 14.3, "K562_tpm": 3.2},
    # chr3 GATA2 locus (hg19)
    {"symbol": "GATA2", "chrom": "chr3", "start": 128_198_265, "end": 128_212_030, "strand": "-", "feature_type": "gene", "GM12878_tpm": 0.3, "K562_tpm": 168.4},
    {"symbol": "RPN1", "chrom": "chr3", "start": 128_337_390, "end": 128_368_269, "strand": "-", "feature_type": "gene", "GM12878_tpm": 112.0, "K562_tpm": 145.0},
    {"symbol": "EEFSEC", "chrom": "chr3", "start": 127_876_120, "end": 128_128_713, "strand": "+", "feature_type": "gene", "GM12878_tpm": 12.6, "K562_tpm": 28.4},
    {"symbol": "GATA2_ENH", "chrom": "chr3", "start": 128_285_000, "end": 128_305_000, "strand": ".", "feature_type": "enhancer", "GM12878_tpm": 0.0, "K562_tpm": 0.0},
]

TARGET_LOCI_META = {
    "MYC": {
        "chrom": "chr8",
        "start": 127_500_000,
        "end": 130_000_000,
        "padded_3d_start": 126_500_000,
        "padded_3d_end": 131_000_000,
        "key_genes": ["MYC", "PVT1"],
        "key_enhancers": ["MYC_E1"],
    },
    "HBB": {
        "chrom": "chr11",
        "start": 4_000_000,
        "end": 6_500_000,
        "padded_3d_start": 3_000_000,
        "padded_3d_end": 7_500_000,
        "key_genes": ["HBB", "HBG1", "HBG2", "HBD"],
        "key_enhancers": ["HBB_LCR"],
    },
    "BCL2": {
        "chrom": "chr18",
        "start": 59_500_000,
        "end": 62_000_000,
        "padded_3d_start": 58_500_000,
        "padded_3d_end": 63_000_000,
        "key_genes": ["BCL2", "KDSR", "VPS4B"],
        "key_enhancers": [],
    },
    "GATA2": {
        "chrom": "chr3",
        "start": 127_000_000,
        "end": 129_500_000,
        "padded_3d_start": 126_000_000,
        "padded_3d_end": 130_500_000,
        "key_genes": ["GATA2", "RPN1", "EEFSEC"],
        "key_enhancers": ["GATA2_ENH"],
    },
}


def fetch_ucsc_hg19_refgene() -> pd.DataFrame:
    """Download UCSC hg19 refGene annotations for full-chromosome gene density phasing."""
    url = "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/refGene.txt.gz"
    try:
        print(f"Fetching UCSC hg19 refGene annotations from {url}...")
        req = urllib.request.Request(url, headers={"User-Agent": "ChromCopilot/0.1"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw_bytes = resp.read()
        with gzip.GzipFile(fileobj=io.BytesIO(raw_bytes)) as gz:
            df = pd.read_csv(
                gz,
                sep="\t",
                header=None,
                usecols=[2, 3, 4, 5, 12],
                names=["chrom", "strand", "start", "end", "symbol"],
            )
        df = df[df["chrom"].isin(TARGET_CHROMS)].copy()
        # Collapse isoforms per gene symbol
        df = (
            df.groupby(["chrom", "symbol", "strand"], as_index=False)
            .agg({"start": "min", "end": "max"})
        )
        df["feature_type"] = "gene"
        df["GM12878_tpm"] = np.nan
        df["K562_tpm"] = np.nan
        print(f"  Loaded {len(df):,} hg19 RefSeq genes across {TARGET_CHROMS}")
        return df
    except Exception as exc:
        print(f"  Warning: Could not fetch UCSC refGene ({exc}). Falling back to curated locus table.")
        return pd.DataFrame(columns=["chrom", "symbol", "strand", "start", "end", "feature_type", "GM12878_tpm", "K562_tpm"])


def main() -> None:
    ref_df = fetch_ucsc_hg19_refgene()
    curated_df = pd.DataFrame(CURATED_LOCUS_FEATURES)

    # Merge curated expression & regulatory features on top of UCSC RefSeq genes
    if not ref_df.empty:
        ref_df = ref_df[~ref_df["symbol"].isin(curated_df["symbol"])]
        genes_df = pd.concat([curated_df, ref_df], ignore_index=True)
    else:
        genes_df = curated_df.copy()

    genes_df.sort_values(["chrom", "start"], inplace=True, ignore_index=True)
    genes_out = TRACKS_DIR / "genes_hg19.parquet"
    genes_df.to_parquet(genes_out, index=False)
    print(f"Saved {len(genes_df):,} 1D gene/regulatory features to {genes_out}")

    # Build bin-aligned 1D gene density & expression signals for all resolutions
    signal_rows: list[pd.DataFrame] = []
    chrom_sizes: dict[str, int] = {}

    for res in RESOLUTIONS:
        clr = cooler.Cooler(f"{CONTACTS_DIR / 'GM12878.mcool'}::/resolutions/{res}")
        if not chrom_sizes:
            chrom_sizes = {c: int(s) for c, s in clr.chromsizes.items()}

        bins = clr.bins()[:][["chrom", "start", "end"]].copy()
        bins["resolution_bp"] = int(res)
        bins["bin_index"] = (bins["start"] // res).astype(np.int64)
        bins["gene_density"] = 0.0
        bins["GM12878_tpm_mean"] = np.nan
        bins["K562_tpm_mean"] = np.nan

        for chrom in TARGET_CHROMS:
            c_genes = genes_df[(genes_df["chrom"] == chrom) & (genes_df["feature_type"] == "gene")]
            if c_genes.empty:
                continue
            midpoints = ((c_genes["start"] + c_genes["end"]) // 2).to_numpy()
            gene_bins = midpoints // res

            # Count genes per bin (phasing track for cooltools A/B compartments)
            counts = pd.Series(gene_bins).value_counts()
            mask_c = bins["chrom"] == chrom
            bins.loc[mask_c, "gene_density"] = bins.loc[mask_c, "bin_index"].map(counts).fillna(0.0).astype(float)

            # Map curated TPM expression values onto bins
            expr_genes = c_genes.dropna(subset=["GM12878_tpm", "K562_tpm"]).copy()
            if not expr_genes.empty:
                expr_genes["bin_index"] = ((expr_genes["start"] + expr_genes["end"]) // 2) // res
                agg = expr_genes.groupby("bin_index")[["GM12878_tpm", "K562_tpm"]].mean()
                bins.loc[mask_c, "GM12878_tpm_mean"] = bins.loc[mask_c, "bin_index"].map(agg["GM12878_tpm"])
                bins.loc[mask_c, "K562_tpm_mean"] = bins.loc[mask_c, "bin_index"].map(agg["K562_tpm"])

        signal_rows.append(bins)

    bin_signals_df = pd.concat(signal_rows, ignore_index=True)
    signals_out = TRACKS_DIR / "bin_signals_hg19.parquet"
    bin_signals_df.to_parquet(signals_out, index=False)
    print(f"Saved {len(bin_signals_df):,} bin-aligned 1D signals to {signals_out}")

    # Build quick lookup table for curated genes & enhancers
    gene_lookup: dict[str, dict] = {}
    for _, row in curated_df.iterrows():
        mid = int((row["start"] + row["end"]) // 2)
        gene_lookup[row["symbol"]] = {
            "chrom": row["chrom"],
            "start": int(row["start"]),
            "end": int(row["end"]),
            "feature_type": row["feature_type"],
            "bin_50kb": mid // 50_000,
            "bin_100kb": mid // 100_000,
            "bin_250kb": mid // 250_000,
            "GM12878_tpm": float(row["GM12878_tpm"]),
            "K562_tpm": float(row["K562_tpm"]),
        }

    sparsity_path = CONTACTS_DIR / "sparsity_summary.json"
    sparsity_baseline = json.loads(sparsity_path.read_text()) if sparsity_path.exists() else {}

    metadata = {
        "schema_version": "1.0.0",
        "dataset_id": "rao2014_gm12878_k562",
        "dataset_mode": "bulk_condition",
        "genome_build": "hg19",
        "source_accession": "GSE63525",
        "available_resolutions_bp": RESOLUTIONS,
        "chromosomes": chrom_sizes,
        "cohorts": {
            "GM12878": {
                "cohort_id": "GM12878",
                "label": "GM12878 (B-Lymphoblastoid)",
                "entity_ids": ["GM12878"],
                "color_hex": "#2563EB",
                "mcool_path": "contacts/GM12878.mcool",
            },
            "K562": {
                "cohort_id": "K562",
                "label": "K562 (Erythroleukemia)",
                "entity_ids": ["K562"],
                "color_hex": "#DC2626",
                "mcool_path": "contacts/K562.mcool",
            },
        },
        "structures_3d_path": "structures_3d/rao2014_gm12878_k562_3d.parquet",
        "tracks_1d": {
            "genes_path": "tracks_1d/genes_hg19.parquet",
            "bin_signals_path": "tracks_1d/bin_signals_hg19.parquet",
        },
        "target_loci": TARGET_LOCI_META,
        "gene_lookup": gene_lookup,
        "sparsity_baseline": sparsity_baseline,
    }

    meta_out = PROCESSED_DIR / "metadata.json"
    meta_out.write_text(json.dumps(metadata, indent=2))
    print(f"Saved master manifest to {meta_out}")


if __name__ == "__main__":
    main()