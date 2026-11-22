# Stratifies a cohort by gene expression (<entity_id>_tpm) into high- vs. low-expressing sub-cohorts, gracefully handling 2-condition bulk datasets as well as future multi-cell cohorts.

from __future__ import annotations

import numpy as np

from app.engine.data_store import DataStore
from app.schemas.tools import StratifyBy1DInput, StratifyBy1DOutput
from app.schemas.viewport import CohortSpec


def run_stratify_by_1d_feature(
    store: DataStore, req: StratifyBy1DInput
) -> StratifyBy1DOutput:
    store.validate_dataset(req.dataset_id)
    warnings: list[str] = []

    entities = req.source_cohort.entity_ids
    if len(entities) < 2:
        raise ValueError(
            "Stratification requires at least 2 entities in source_cohort.entity_ids "
            "(e.g., ['GM12878', 'K562'] in bulk mode)."
        )

    # Case-insensitive lookup in 1D gene table
    df = store.genes_1d
    match = df[df["symbol"].str.upper() == req.feature_name.upper()]
    if match.empty:
        raise ValueError(
            f"1D feature/gene '{req.feature_name}' not found in dataset '{req.dataset_id}'."
        )

    row = match.iloc[0]
    canonical_symbol = str(row["symbol"])
    values: dict[str, float] = {}
    for ent in entities:
        col = f"{ent}_tpm"
        if col not in match.columns or np.isnan(row[col]):
            raise ValueError(
                f"No 1D expression value ('{col}') available for gene '{canonical_symbol}'."
            )
        values[ent] = float(row[col])

    sorted_ents = sorted(entities, key=lambda e: values[e])
    arr = np.array([values[e] for e in entities], dtype=np.float64)

    if req.split_method == "condition_group" or len(entities) == 2:
        if len(entities) == 2 and req.split_method != "condition_group":
            warnings.append(
                "Bulk 2-condition cohort detected: splitting directly into higher vs. lower expressing condition."
            )
        low_ids = [sorted_ents[0]]
        high_ids = [sorted_ents[-1]]
    elif req.split_method == "quantile":
        hi_thr = float(np.quantile(arr, req.high_cutoff))
        lo_thr = float(np.quantile(arr, req.low_cutoff))
        high_ids = [e for e in entities if values[e] >= hi_thr]
        low_ids = [e for e in entities if values[e] <= lo_thr]
        if not high_ids or not low_ids:
            mid = len(sorted_ents) // 2
            low_ids, high_ids = sorted_ents[:mid], sorted_ents[mid:]
            warnings.append("Quantile thresholds produced an empty group; fell back to median split.")
    else:  # threshold
        high_ids = [e for e in entities if values[e] >= req.high_cutoff]
        low_ids = [e for e in entities if values[e] <= req.low_cutoff]
        if not high_ids or not low_ids:
            raise ValueError(
                f"Threshold split (low <= {req.low_cutoff}, high >= {req.high_cutoff}) "
                f"yielded an empty cohort for '{canonical_symbol}'."
            )

    hi_vals = np.array([values[e] for e in high_ids], dtype=np.float64)
    lo_vals = np.array([values[e] for e in low_ids], dtype=np.float64)

    return StratifyBy1DOutput(
        dataset_id=req.dataset_id,
        n_entities_used=len(entities),
        contact_sparsity_fraction=None,
        warnings=warnings,
        feature_name=canonical_symbol,
        high_cohort=CohortSpec(
            cohort_id=f"{canonical_symbol.lower()}_high",
            label=f"{canonical_symbol}-High ({', '.join(high_ids)})",
            entity_ids=high_ids,
            color_hex="#DC2626",
        ),
        low_cohort=CohortSpec(
            cohort_id=f"{canonical_symbol.lower()}_low",
            label=f"{canonical_symbol}-Low ({', '.join(low_ids)})",
            entity_ids=low_ids,
            color_hex="#2563EB",
        ),
        summary_stats={
            "high_mean": round(float(np.mean(hi_vals)), 4),
            "low_mean": round(float(np.mean(lo_vals)), 4),
            "high_median": round(float(np.median(hi_vals)), 4),
            "low_median": round(float(np.median(lo_vals)), 4),
        },
    )