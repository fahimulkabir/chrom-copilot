# Enforces statistical integrity (automatically preventing bogus p-values when sample size is less than 3, such as bulk n=1 comparisons) and builds validated Zustand viewport patches with persistent bookmarks.

from __future__ import annotations

from uuid import uuid4

import numpy as np
from scipy import stats

from app.engine.data_store import DataStore
from app.schemas.tools import (
    DifferentialTestInput,
    DifferentialTestOutput,
    EmitViewportSyncInput,
    EmitViewportSyncOutput,
)


def run_differential_test(
    store: DataStore, req: DifferentialTestInput
) -> DifferentialTestOutput:
    store.validate_dataset(req.dataset_id)
    warnings: list[str] = []

    a = np.asarray(req.group_a_values, dtype=np.float64)
    b = np.asarray(req.group_b_values, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        raise ValueError("Both groups must contain at least one finite numeric value.")

    med_a = float(np.median(a))
    med_b = float(np.median(b))
    n_total = int(a.size + b.size)

    # Safeguard: never fabricate p-values when sample size is too small (e.g. bulk n=1)
    if req.test_type == "delta_only" or a.size < 3 or b.size < 3:
        if req.test_type != "delta_only":
            warnings.append(
                f"Sample sizes too small (n_a={a.size}, n_b={b.size}; need >= 3 per group) "
                f"for '{req.test_type}'. Downgraded to delta_only with statistically_valid=False."
            )
        return DifferentialTestOutput(
            dataset_id=req.dataset_id,
            n_entities_used=n_total,
            contact_sparsity_fraction=None,
            warnings=warnings,
            metric_name=req.metric_name,
            test_type="delta_only",
            group_a_median=round(med_a, 6),
            group_b_median=round(med_b, 6),
            effect_size=round(med_b - med_a, 6),
            statistic=None,
            p_value=None,
            statistically_valid=False,
        )

    if req.test_type == "mann_whitney_u":
        res = stats.mannwhitneyu(b, a, alternative="two-sided")
        u_stat = float(res.statistic)
        p_val = float(res.pvalue)
        # Rank-biserial correlation in [-1, 1]; positive means group B > group A
        effect = (2.0 * u_stat) / (a.size * b.size) - 1.0
    elif req.test_type == "welch_t":
        res = stats.ttest_ind(b, a, equal_var=False)
        u_stat = float(res.statistic)
        p_val = float(res.pvalue)
        pooled_sd = float(np.sqrt(0.5 * (np.var(a, ddof=1) + np.var(b, ddof=1))))
        effect = (float(np.mean(b)) - float(np.mean(a))) / pooled_sd if pooled_sd > 0 else 0.0
    else:  # deterministic permutation test on median difference
        rng = np.random.default_rng(42)
        obs_diff = med_b - med_a
        combined = np.concatenate([a, b])
        n_a = a.size
        extreme = 0
        for _ in range(req.n_permutations):
            perm = rng.permutation(combined)
            diff = float(np.median(perm[n_a:]) - np.median(perm[:n_a]))
            if abs(diff) >= abs(obs_diff):
                extreme += 1
        u_stat = obs_diff
        p_val = (extreme + 1.0) / (req.n_permutations + 1.0)
        effect = obs_diff

    return DifferentialTestOutput(
        dataset_id=req.dataset_id,
        n_entities_used=n_total,
        contact_sparsity_fraction=None,
        warnings=warnings,
        metric_name=req.metric_name,
        test_type=req.test_type,
        group_a_median=round(med_a, 6),
        group_b_median=round(med_b, 6),
        effect_size=round(float(effect), 6),
        statistic=round(float(u_stat), 6),
        p_value=round(float(min(max(p_val, 0.0), 1.0)), 6),
        statistically_valid=True,
    )


def run_emit_viewport_sync(
    store: DataStore, req: EmitViewportSyncInput
) -> EmitViewportSyncOutput:
    store.validate_dataset(req.dataset_id)
    warnings: list[str] = []
    bookmark_id = f"bk_{uuid4().hex[:8]}"

    patch: dict = {
        "dataset_id": req.dataset_id,
        "locus": req.locus.model_dump(),
        "resolution_bp": req.resolution_bp,
        "active_cohorts": [c.model_dump() for c in req.active_cohorts],
        "highlighted_bins": [h.model_dump() for h in req.highlighted_bins],
        "bookmark_id": bookmark_id,
    }

    # Compute smooth Three.js camera focus target if camera_focus_bin is provided
    if req.camera_focus_bin is not None:
        primary_ent = req.active_cohorts[0].entity_ids[0]
        df_3d = store.get_3d_coords(primary_ent, req.locus.chrom, req.resolution_bp)
        focus_row = df_3d[df_3d["bin_index"] == req.camera_focus_bin]
        if not focus_row.empty:
            tx = float(focus_row.iloc[0]["x"])
            ty = float(focus_row.iloc[0]["y"])
            tz = float(focus_row.iloc[0]["z"])
            patch["camera_3d"] = {
                "target": [round(tx, 4), round(ty, 4), round(tz, 4)],
                "position": [round(tx + 4.0, 4), round(ty + 3.0, 4), round(tz + 6.0, 4)],
            }
        else:
            warnings.append(
                f"camera_focus_bin {req.camera_focus_bin} not in cached 3D fold; keeping current camera."
            )

    store.save_bookmark(bookmark_id, patch)

    n_entities = sum(len(c.entity_ids) for c in req.active_cohorts)
    return EmitViewportSyncOutput(
        dataset_id=req.dataset_id,
        n_entities_used=n_entities,
        contact_sparsity_fraction=None,
        warnings=warnings,
        bookmark_id=bookmark_id,
        viewport_patch=patch,
    )