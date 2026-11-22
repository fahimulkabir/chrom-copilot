from __future__ import annotations

from pathlib import Path

import cooler
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import shortest_path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTACTS_DIR = REPO_ROOT / "data" / "processed" / "contacts"
OUT_DIR = REPO_ROOT / "data" / "processed" / "structures_3d"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENTITIES = ["GM12878", "K562"]
RESOLUTIONS = [50_000, 100_000, 250_000]
ALPHA = 0.5  # Contact-to-distance conversion exponent

# 4.5 Mb padded 3D windows (+/- 1 Mb around each 2.5 Mb target locus)
PADDED_3D_WINDOWS = {
    "MYC": {"chrom": "chr8", "start": 126_500_000, "end": 131_000_000},
    "HBB": {"chrom": "chr11", "start": 3_000_000, "end": 7_500_000},
    "BCL2": {"chrom": "chr18", "start": 58_500_000, "end": 63_000_000},
    "GATA2": {"chrom": "chr3", "start": 126_000_000, "end": 130_500_000},
}


def contact_matrix_to_3d(mat: np.ndarray, alpha: float = ALPHA) -> tuple[np.ndarray, float]:
    """Deterministic shortest-path + Classical MDS 3D polymer reconstruction."""
    n = mat.shape[0]
    clean = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    clean = 0.5 * (clean + clean.T)
    np.fill_diagonal(clean, 0.0)

    # Convert positive contacts to wish distances
    pos_vals = clean[clean > 0]
    if pos_vals.size == 0:
        raise ValueError("Empty contact window; cannot reconstruct 3D structure.")

    # Normalize contact scale so different cell lines stay in comparable units
    scale = float(np.median(pos_vals))
    norm_c = clean / scale

    dist_graph = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(dist_graph, 0.0)
    pos_mask = norm_c > 0
    dist_graph[pos_mask] = np.power(norm_c[pos_mask], -alpha)

    # Ensure polymer backbone connectivity for adjacent bins (i, i+1)
    adj_dists = [dist_graph[i, i + 1] for i in range(n - 1) if np.isfinite(dist_graph[i, i + 1])]
    backbone_step = float(np.median(adj_dists)) if adj_dists else 1.0
    for i in range(n - 1):
        if not np.isfinite(dist_graph[i, i + 1]) or dist_graph[i, i + 1] > backbone_step * 2.5:
            dist_graph[i, i + 1] = backbone_step
            dist_graph[i + 1, i] = backbone_step

    # Geodesic completion for sparse/zero entries
    D = shortest_path(dist_graph, directed=False)
    D = np.nan_to_num(D, nan=backbone_step * n, posinf=backbone_step * n)

    # Classical MDS: eigendecomposition of double-centered -0.5 * J * D^2 * J
    D2 = D ** 2
    row_mean = D2.mean(axis=1, keepdims=True)
    col_mean = D2.mean(axis=0, keepdims=True)
    grand_mean = D2.mean()
    B = -0.5 * (D2 - row_mean - col_mean + grand_mean)

    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1][:3]
    top_vals = np.maximum(eigvals[idx], 1e-9)
    coords = eigvecs[:, idx] * np.sqrt(top_vals)

    # Center at origin and compute Radius of Gyration (Rg)
    coords -= coords.mean(axis=0, keepdims=True)
    rg = float(np.sqrt(np.mean(np.sum(coords ** 2, axis=1))))
    return coords, rg


def kabsch_align(mobile: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Rigid rotation aligning mobile onto ref WITHOUT rescaling (preserves true Rg)."""
    H = mobile.T @ ref
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return mobile @ R.T


def main() -> None:
    rows: list[dict] = []

    for res in RESOLUTIONS:
        clrs = {
            ent: cooler.Cooler(f"{CONTACTS_DIR / f'{ent}.mcool'}::/resolutions/{res}")
            for ent in ENTITIES
        }

        for locus_name, win in PADDED_3D_WINDOWS.items():
            chrom = win["chrom"]
            region = f"{chrom}:{win['start']}-{win['end']}"
            ref_coords: np.ndarray | None = None

            for ent in ENTITIES:
                clr = clrs[ent]
                bins = clr.bins().fetch(region)
                mat = clr.matrix(balance=True).fetch(region)

                # Fallback to raw counts if balanced matrix is too sparse in any sub-window
                if np.count_nonzero(np.isfinite(mat) & (mat > 0)) < len(bins):
                    mat = clr.matrix(balance=False).fetch(region)

                coords, rg = contact_matrix_to_3d(mat)
                if ref_coords is None:
                    ref_coords = coords
                elif coords.shape == ref_coords.shape:
                    coords = kabsch_align(coords, ref_coords)

                for local_i, (_, b_row) in enumerate(bins.iterrows()):
                    start_bp = int(b_row["start"])
                    end_bp = int(b_row["end"])
                    chrom_bin_idx = start_bp // res
                    rows.append(
                        {
                            "entity_id": ent,
                            "locus_name": locus_name,
                            "chrom": chrom,
                            "resolution_bp": int(res),
                            "bin_index": int(chrom_bin_idx),
                            "start": start_bp,
                            "end": end_bp,
                            "x": float(coords[local_i, 0]),
                            "y": float(coords[local_i, 1]),
                            "z": float(coords[local_i, 2]),
                            "rg": float(rg),
                        }
                    )

            print(f"Reconstructed 3D folds for {locus_name} ({chrom}) at {res // 1000} kb")

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / "rao2014_gm12878_k562_3d.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved {len(df):,} 3D coordinate records to {out_path}")


if __name__ == "__main__":
    main()