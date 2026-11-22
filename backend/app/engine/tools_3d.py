# Computes deterministic 3D Euclidean distances (supporting both raw model units and Rg-normalized units) and powers Point-to-Query spatial neighborhood searches with overlapping gene annotations.

from __future__ import annotations

import numpy as np
import pandas as pd

from app.engine.data_store import DataStore
from app.schemas.tools import (
    Pairwise3DDistanceInput,
    Pairwise3DDistanceOutput,
    SpatialNeighborBin,
    SpatialNeighbors3DInput,
    SpatialNeighbors3DOutput,
)
from app.schemas.viewport import GenomicInterval


def _interval_centroid(
    df_ent: pd.DataFrame, interval: GenomicInterval, resolution_bp: int
) -> tuple[np.ndarray, float]:
    """Return 3D centroid and Rg for bins overlapping interval (or nearest midpoint bin)."""
    overlap = df_ent[(df_ent["end"] > interval.start) & (df_ent["start"] < interval.end)]
    if overlap.empty:
        mid_bin = ((interval.start + interval.end) // 2) // resolution_bp
        overlap = df_ent[df_ent["bin_index"] == mid_bin]
    if overlap.empty:
        raise ValueError(
            f"No 3D coordinates cached for {interval.chrom}:{interval.start}-{interval.end} "
            f"at {resolution_bp} bp."
        )
    centroid = overlap[["x", "y", "z"]].mean(axis=0).to_numpy(dtype=np.float64)
    rg = float(overlap["rg"].iloc[0])
    return centroid, rg


def run_measure_3d_pairwise_distance(
    store: DataStore, req: Pairwise3DDistanceInput
) -> Pairwise3DDistanceOutput:
    store.validate_dataset(req.dataset_id)
    if req.locus_a.chrom != req.locus_b.chrom:
        raise ValueError(
            "Inter-chromosomal 3D pairwise distance is not supported for cis locus folds."
        )

    chrom = req.locus_a.chrom
    entity_distances: dict[str, float] = {}
    warnings: list[str] = []

    for ent in req.cohort.entity_ids:
        df_ent = store.get_3d_coords(ent, chrom, req.resolution_bp)
        if df_ent.empty:
            warnings.append(f"No 3D fold found for entity '{ent}' on {chrom}.")
            continue

        pos_a, rg_a = _interval_centroid(df_ent, req.locus_a, req.resolution_bp)
        pos_b, _ = _interval_centroid(df_ent, req.locus_b, req.resolution_bp)
        dist = float(np.linalg.norm(pos_a - pos_b))
        if req.normalize_by_radius_of_gyration and rg_a > 0:
            dist /= rg_a
        entity_distances[ent] = round(dist, 5)

    if not entity_distances:
        raise ValueError("Could not compute 3D pairwise distance for any entity in cohort.")

    vals = np.array(list(entity_distances.values()), dtype=np.float64)
    q75, q25 = np.percentile(vals, [75, 25])
    units = "rg_normalized" if req.normalize_by_radius_of_gyration else "model_units"

    return Pairwise3DDistanceOutput(
        dataset_id=req.dataset_id,
        n_entities_used=len(entity_distances),
        contact_sparsity_fraction=None,
        warnings=warnings,
        entity_distances=entity_distances,
        mean_distance=round(float(np.mean(vals)), 5),
        median_distance=round(float(np.median(vals)), 5),
        iqr_distance=round(float(q75 - q25), 5),
        units=units,
    )


def run_find_3d_spatial_neighbors(
    store: DataStore, req: SpatialNeighbors3DInput
) -> SpatialNeighbors3DOutput:
    store.validate_dataset(req.dataset_id)
    warnings: list[str] = []

    # Collect distances per candidate bin across entities in the cohort
    bin_dists: dict[int, list[float]] = {}
    bin_meta: dict[int, tuple[int, int]] = {}
    valid_entities = 0

    for ent in req.cohort.entity_ids:
        df_ent = store.get_3d_coords(ent, req.anchor_chrom, req.resolution_bp)
        anchor_rows = df_ent[df_ent["bin_index"] == req.anchor_bin_index]
        if anchor_rows.empty:
            warnings.append(
                f"Anchor bin {req.anchor_bin_index} not found in 3D fold for '{ent}'."
            )
            continue

        valid_entities += 1
        anchor_xyz = anchor_rows.iloc[0][["x", "y", "z"]].to_numpy(dtype=np.float64)

        # Exclude polymer-backbone adjacent bins
        candidates = df_ent[
            (df_ent["bin_index"] - req.anchor_bin_index).abs()
            >= req.min_linear_separation_bins
        ]
        if candidates.empty:
            continue

        coords = candidates[["x", "y", "z"]].to_numpy(dtype=np.float64)
        dists = np.linalg.norm(coords - anchor_xyz, axis=1)

        for idx, dist in zip(candidates["bin_index"].tolist(), dists.tolist()):
            b_idx = int(idx)
            bin_dists.setdefault(b_idx, []).append(float(dist))

        for _, row in candidates.iterrows():
            b_idx = int(row["bin_index"])
            if b_idx not in bin_meta:
                bin_meta[b_idx] = (int(row["start"]), int(row["end"]))

    if valid_entities == 0:
        raise ValueError(
            f"Anchor bin {req.anchor_bin_index} on {req.anchor_chrom} ({req.resolution_bp} bp) "
            f"not found in cached 3D folds."
        )

    neighbors: list[SpatialNeighborBin] = []
    for b_idx, d_list in bin_dists.items():
        within = [d for d in d_list if d <= req.radius_threshold]
        if not within:
            continue
        start_bp, end_bp = bin_meta[b_idx]
        genes = store.get_overlapping_genes(req.anchor_chrom, start_bp, end_bp)
        neighbors.append(
            SpatialNeighborBin(
                chrom=req.anchor_chrom,
                bin_index=b_idx,
                start=start_bp,
                end=end_bp,
                mean_3d_distance=round(float(np.mean(d_list)), 5),
                fraction_entities_within_radius=round(len(within) / valid_entities, 4),
                overlapping_genes=genes,
            )
        )

    neighbors.sort(key=lambda nb: nb.mean_3d_distance)

    return SpatialNeighbors3DOutput(
        dataset_id=req.dataset_id,
        n_entities_used=valid_entities,
        contact_sparsity_fraction=None,
        warnings=warnings,
        anchor_bin_index=req.anchor_bin_index,
        neighbors=neighbors,
    )