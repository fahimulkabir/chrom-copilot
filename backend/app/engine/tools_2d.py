# Uses cooler and cooltools directly for deterministic, peer-review-grade 2D chromatin metrics.

from __future__ import annotations

import cooltools
import numpy as np
import pandas as pd

from app.engine.data_store import DataStore
from app.schemas.tools import (
    ABCompartmentInput,
    ABCompartmentOutput,
    BoundaryCall,
    CompartmentSwitch,
    ContactMapInput,
    ContactMapOutput,
    SparsePixel,
    TadInsulationInput,
    TadInsulationOutput,
)


def _upper_triangle_sparsity(mat: np.ndarray) -> float:
    n = mat.shape[0]
    if n == 0:
        return 1.0
    vals = mat[np.triu_indices(n, k=0)]
    valid_pos = np.isfinite(vals) & (vals > 0)
    return round(float(1.0 - np.mean(valid_pos)), 4)


def run_get_contact_map(store: DataStore, req: ContactMapInput) -> ContactMapOutput:
    store.validate_dataset(req.dataset_id)
    warnings: list[str] = []

    use_balance = req.normalization in ("ice_balanced", "obs_over_exp")
    mat, bins_df = store.get_matrix(
        req.cohort.entity_ids,
        req.locus.chrom,
        req.locus.start,
        req.locus.end,
        req.resolution_bp,
        balance=use_balance,
    )

    # Measure sparsity on the raw contact matrix so ICE NaNs don't distort zero rate
    raw_mat, _ = store.get_matrix(
        req.cohort.entity_ids,
        req.locus.chrom,
        req.locus.start,
        req.locus.end,
        req.resolution_bp,
        balance=False,
    )
    sparsity = _upper_triangle_sparsity(raw_mat)
    if sparsity > 0.90:
        warnings.append(
            f"High contact sparsity ({sparsity:.1%}) at {req.resolution_bp // 1000} kb."
        )

    n = mat.shape[0]
    work = mat.copy()

    if req.normalization == "obs_over_exp":
        oe = np.full_like(work, np.nan, dtype=np.float64)
        for diag_k in range(n):
            diag_vals = np.diagonal(work, offset=diag_k)
            finite = diag_vals[np.isfinite(diag_vals) & (diag_vals > 0)]
            if finite.size == 0:
                continue
            exp_val = float(np.mean(finite))
            if exp_val <= 0:
                continue
            idx_i = np.arange(0, n - diag_k)
            idx_j = idx_i + diag_k
            ratios = work[idx_i, idx_j] / exp_val
            oe[idx_i, idx_j] = ratios
        work = oe

    bin_indices = (bins_df["start"].to_numpy(dtype=np.int64) // req.resolution_bp).tolist()
    tri_i, tri_j = np.triu_indices(n, k=0)
    vals = work[tri_i, tri_j]
    keep = np.isfinite(vals) & (vals > 0)

    pixels = [
        SparsePixel(
            bin_i=int(bin_indices[i]),
            bin_j=int(bin_indices[j]),
            value=round(float(v), 6),
        )
        for i, j, v in zip(tri_i[keep], tri_j[keep], vals[keep])
    ]

    total_raw = float(np.nansum(raw_mat[np.triu_indices(n, k=0)]))
    return ContactMapOutput(
        dataset_id=req.dataset_id,
        n_entities_used=len(req.cohort.entity_ids),
        contact_sparsity_fraction=sparsity,
        warnings=warnings,
        locus=req.locus,
        resolution_bp=req.resolution_bp,
        n_bins=n,
        total_contacts=round(total_raw, 2),
        sparse_pixels=pixels,
    )


def run_compute_tad_insulation(
    store: DataStore, req: TadInsulationInput
) -> TadInsulationOutput:
    store.validate_dataset(req.dataset_id)
    warnings: list[str] = []

    res = req.resolution_bp
    # cooltools requires window_bp to be a multiple of resolution_bp and >= 3 * res
    window_bp = req.window_bp
    snapped = max(3 * res, int(round(window_bp / res) * res))
    if snapped != window_bp:
        warnings.append(
            f"Adjusted insulation window from {window_bp} bp to {snapped} bp "
            f"(must be a multiple of {res} bp and >= 3 bins)."
        )
        window_bp = snapped

    # Compute sparsity for the requested locus
    raw_mat, _ = store.get_matrix(
        req.cohort.entity_ids,
        req.locus.chrom,
        req.locus.start,
        req.locus.end,
        res,
        balance=False,
    )
    sparsity = _upper_triangle_sparsity(raw_mat)

    # Run cooltools.insulation across full chromosome (avoids window edge artifacts), then slice
    ent = req.cohort.entity_ids[0]
    if len(req.cohort.entity_ids) > 1:
        warnings.append(
            f"TAD insulation currently computed on primary cohort representative '{ent}'."
        )

    clr = store.get_cooler(ent, res)
    chrom = req.locus.chrom
    view_df = pd.DataFrame(
        [{"chrom": chrom, "start": 0, "end": int(clr.chromsizes[chrom]), "name": chrom}]
    )

    ins_df = cooltools.insulation(
        clr,
        window_bp=[window_bp],
        view_df=view_df,
        ignore_diags=2,
        clr_weight_name="weight",
    )

    sub = ins_df[
        (ins_df["chrom"] == chrom)
        & (ins_df["start"] >= req.locus.start)
        & (ins_df["end"] <= req.locus.end)
    ].reset_index(drop=True)

    score_col = f"log2_insulation_score_{window_bp}"
    strength_col = f"boundary_strength_{window_bp}"

    bin_positions = [int(s) for s in sub["start"].tolist()]
    scores: list[float | None] = [
        round(float(v), 4) if pd.notna(v) and np.isfinite(v) else None
        for v in sub[score_col].tolist()
    ]

    boundaries: list[BoundaryCall] = []
    for _, row in sub.iterrows():
        bs = row.get(strength_col, np.nan)
        sc = row.get(score_col, np.nan)
        if pd.notna(bs) and np.isfinite(bs) and float(bs) >= req.min_boundary_strength:
            start_bp = int(row["start"])
            boundaries.append(
                BoundaryCall(
                    bin_index=start_bp // res,
                    genomic_pos=start_bp,
                    insulation_score=round(float(sc), 4) if pd.notna(sc) else 0.0,
                    boundary_strength=round(float(bs), 4),
                )
            )

    return TadInsulationOutput(
        dataset_id=req.dataset_id,
        n_entities_used=len(req.cohort.entity_ids),
        contact_sparsity_fraction=sparsity,
        warnings=warnings,
        bin_positions=bin_positions,
        insulation_scores=scores,
        boundaries=boundaries,
    )


def _compute_cached_eigs(
    store: DataStore, entity_id: str, chrom: str, resolution_bp: int
) -> pd.DataFrame:
    cache_key = (entity_id, chrom, int(resolution_bp))
    if cache_key in store._eigs_cache:
        return store._eigs_cache[cache_key]

    disk_cache = store.cache_dir / f"eigs_{entity_id}_{chrom}_{resolution_bp}.parquet"
    if disk_cache.exists():
        eigvecs = pd.read_parquet(disk_cache)
        store._eigs_cache[cache_key] = eigvecs
        return eigvecs

    clr = store.get_cooler(entity_id, resolution_bp)
    view_df = pd.DataFrame(
        [{"chrom": chrom, "start": 0, "end": int(clr.chromsizes[chrom]), "name": chrom}]
    )
    phasing_df = store.get_1d_phasing_track(chrom, resolution_bp)

    _, eigvecs = cooltools.eigs_cis(
        clr,
        phasing_track=phasing_df,
        view_df=view_df,
        n_eigs=3,
        clr_weight_name="weight",
        ignore_diags=2,
    )
    eigvecs = eigvecs[eigvecs["chrom"] == chrom].reset_index(drop=True)
    eigvecs.to_parquet(disk_cache, index=False)
    store._eigs_cache[cache_key] = eigvecs
    return eigvecs


def run_compute_ab_compartments(
    store: DataStore, req: ABCompartmentInput
) -> ABCompartmentOutput:
    store.validate_dataset(req.dataset_id)
    warnings: list[str] = []

    res = req.resolution_bp
    if res not in store.metadata["available_resolutions_bp"]:
        raise ValueError(
            f"Resolution {res} bp not available. Choose from {store.metadata['available_resolutions_bp']}."
        )

    primary_ent = req.cohort.entity_ids[0]
    eigs_primary = _compute_cached_eigs(store, primary_ent, req.chrom, res)

    bin_starts = [int(s) for s in eigs_primary["start"].tolist()]
    e1_scores: list[float | None] = [
        round(float(v), 4) if pd.notna(v) and np.isfinite(v) else None
        for v in eigs_primary["E1"].tolist()
    ]

    switches: list[CompartmentSwitch] = []
    n_entities = len(req.cohort.entity_ids)

    if req.compare_cohort is not None:
        comp_ent = req.compare_cohort.entity_ids[0]
        n_entities += len(req.compare_cohort.entity_ids)
        eigs_comp = _compute_cached_eigs(store, comp_ent, req.chrom, res)

        merged = pd.merge(
            eigs_primary[["start", "end", "E1"]],
            eigs_comp[["start", "end", "E1"]],
            on=["start", "end"],
            suffixes=("_primary", "_compare"),
        )
        # Require sign flip with minimum magnitude (0.05) to ignore zero-crossing noise
        for _, row in merged.iterrows():
            v1, v2 = row["E1_primary"], row["E1_compare"]
            if pd.isna(v1) or pd.isna(v2):
                continue
            v1_f, v2_f = float(v1), float(v2)
            if abs(v1_f) < 0.05 or abs(v2_f) < 0.05:
                continue
            if v1_f > 0 and v2_f < 0:
                switches.append(
                    CompartmentSwitch(
                        bin_index=int(row["start"]) // res,
                        start=int(row["start"]),
                        end=int(row["end"]),
                        e1_primary=round(v1_f, 4),
                        e1_compare=round(v2_f, 4),
                        transition="A_to_B",
                    )
                )
            elif v1_f < 0 and v2_f > 0:
                switches.append(
                    CompartmentSwitch(
                        bin_index=int(row["start"]) // res,
                        start=int(row["start"]),
                        end=int(row["end"]),
                        e1_primary=round(v1_f, 4),
                        e1_compare=round(v2_f, 4),
                        transition="B_to_A",
                    )
                )

    return ABCompartmentOutput(
        dataset_id=req.dataset_id,
        n_entities_used=n_entities,
        contact_sparsity_fraction=None,
        warnings=warnings,
        chrom=req.chrom,
        resolution_bp=res,
        bin_starts=bin_starts,
        e1_scores=e1_scores,
        switches=switches,
    )