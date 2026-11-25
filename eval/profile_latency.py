# Produces the empirical Table 3 (Interactive Latency & Scalability) for your course report and paper—measuring both per-tool warm latency on real GM12878/K562 data and compute scaling from 100 to 2,000 single-cell entities.

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.agent.binder import bind_provenance
from app.agent.planner import generate_plan
from app.agent.verifier import verify_plan
from app.engine.data_store import get_data_store
from app.engine.tools_1d import run_stratify_by_1d_feature
from app.engine.tools_2d import (
    run_compute_ab_compartments,
    run_compute_tad_insulation,
    run_get_contact_map,
)
from app.engine.tools_3d import (
    run_find_3d_spatial_neighbors,
    run_measure_3d_pairwise_distance,
)
from app.engine.tools_stats import run_differential_test, run_emit_viewport_sync
from app.schemas.tools import (
    ABCompartmentInput,
    ContactMapInput,
    DifferentialTestInput,
    EmitViewportSyncInput,
    Pairwise3DDistanceInput,
    SpatialNeighbors3DInput,
    StratifyBy1DInput,
    TadInsulationInput,
)
from app.schemas.viewport import CohortSpec, GenomicInterval, ViewportState

DATASET_ID = "rao2014_gm12878_k562"
GM = CohortSpec(cohort_id="GM12878", label="GM12878", entity_ids=["GM12878"])
K562 = CohortSpec(cohort_id="K562", label="K562", entity_ids=["K562"])
BOTH = CohortSpec(cohort_id="all", label="GM12878+K562", entity_ids=["GM12878", "K562"])


def _bench(fn, n_runs: int = 5) -> tuple[float, float]:
    fn()  # Warm-up (populates caches)
    times_ms: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return round(float(np.mean(times_ms)), 2), round(float(np.percentile(times_ms, 95)), 2)


def main() -> None:
    store = get_data_store()
    myc_iv = GenomicInterval(chrom="chr8", start=127_500_000, end=130_000_000)
    hbb_a = GenomicInterval(chrom="chr11", start=5_246_696, end=5_248_301)
    hbb_b = GenomicInterval(chrom="chr11", start=5_295_000, end=5_315_000)

    # 1. Per-Tool Deterministic Latency Profile
    tool_cases = [
        ("stratify_by_1d_feature", "1D", lambda: run_stratify_by_1d_feature(store, StratifyBy1DInput(dataset_id=DATASET_ID, feature_name="HBB", source_cohort=BOTH, split_method="condition_group"))),
        ("get_contact_map (50kb)", "2D", lambda: run_get_contact_map(store, ContactMapInput(dataset_id=DATASET_ID, cohort=GM, locus=myc_iv, resolution_bp=50_000))),
        ("compute_tad_insulation (50kb)", "2D", lambda: run_compute_tad_insulation(store, TadInsulationInput(dataset_id=DATASET_ID, cohort=GM, locus=myc_iv, resolution_bp=50_000, window_bp=250_000))),
        ("compute_ab_compartments (250kb, cached)", "2D", lambda: run_compute_ab_compartments(store, ABCompartmentInput(dataset_id=DATASET_ID, cohort=GM, compare_cohort=K562, chrom="chr18", resolution_bp=250_000))),
        ("measure_3d_pairwise_distance (50kb)", "3D", lambda: run_measure_3d_pairwise_distance(store, Pairwise3DDistanceInput(dataset_id=DATASET_ID, cohort=BOTH, locus_a=hbb_a, locus_b=hbb_b, resolution_bp=50_000))),
        ("find_3d_spatial_neighbors (50kb)", "3D", lambda: run_find_3d_spatial_neighbors(store, SpatialNeighbors3DInput(dataset_id=DATASET_ID, cohort=K562, anchor_chrom="chr8", anchor_bin_index=2575, resolution_bp=50_000, radius_threshold=4.5))),
        ("run_differential_test (MWU, n=50)", "Stats", lambda: run_differential_test(store, DifferentialTestInput(dataset_id=DATASET_ID, metric_name="insulation", group_a_label="GM", group_b_label="K562", group_a_values=list(np.linspace(-0.2, 0.2, 50)), group_b_values=list(np.linspace(-0.5, 0.1, 50)), test_type="mann_whitney_u"))),
        ("emit_viewport_sync", "Sync", lambda: run_emit_viewport_sync(store, EmitViewportSyncInput(dataset_id=DATASET_ID, locus=myc_iv, resolution_bp=50_000, active_cohorts=[GM, K562], camera_focus_bin=2575))),
    ]

    tool_rows = []
    for name, modality, fn in tool_cases:
        mean_ms, p95_ms = _bench(fn)
        tool_rows.append({"tool": name, "modality": modality, "mean_ms": mean_ms, "p95_ms": p95_ms})

    # 2. End-to-End Stage Breakdown (Plan -> Verify Gate -> Compute -> Bind)
    vp = ViewportState(
        dataset_id=DATASET_ID,
        dataset_mode="bulk_condition",
        genome_build="hg19",
        locus=myc_iv,
        resolution_bp=50_000,
        active_cohorts=[GM, K562],
        camera_3d={"position": (12.0, 9.0, 16.0), "target": (0.0, 0.0, 0.0)},
    )
    query = "Compare MYC TAD insulation between GM12878 and K562 at 50kb"
    plan_mean, _ = _bench(lambda: generate_plan(store, query, vp, use_llm=False))
    plan, _, _ = generate_plan(store, query, vp, use_llm=False)
    gate_mean, _ = _bench(lambda: verify_plan(store, plan, vp))

    sample_out = [
        run_compute_tad_insulation(store, TadInsulationInput(dataset_id=DATASET_ID, cohort=GM, locus=myc_iv, resolution_bp=50_000, window_bp=250_000)).model_dump(mode="json")
    ]
    bind_mean, _ = _bench(lambda: bind_provenance("prof", query, sample_out, use_llm=False))

    stage_rows = [
        {"stage": "1. Plan (Deterministic)", "mean_ms": plan_mean},
        {"stage": "2. Pre-Execution Verify Gate", "mean_ms": gate_mean},
        {"stage": "3. Deterministic Tool Compute (TAD x2 + MWU)", "mean_ms": round(tool_rows[2]["mean_ms"] * 2 + tool_rows[6]["mean_ms"], 2)},
        {"stage": "4. Provenance Binder", "mean_ms": bind_mean},
    ]

    # 3. Single-Cell Cohort Scalability Simulation (100, 500, 1000, 2000 cells)
    rng = np.random.default_rng(42)
    scale_rows = []
    for n_cells in (100, 500, 1000, 2000):
        # Simulate per-cell 3D coordinate array (n_cells, 50 bins, 3) and sparse 50x50 contact stack
        coords = rng.normal(size=(n_cells, 50, 3)).astype(np.float32)
        mats = (rng.random(size=(n_cells, 50, 50)) > 0.96).astype(np.float32)

        def _sc_workload():
            # Pairwise 3D Euclidean distance between bin 10 and bin 25 across all n_cells
            dists = np.linalg.norm(coords[:, 10, :] - coords[:, 25, :], axis=1)
            # Pseudobulk contact aggregation
            _ = mats.mean(axis=0)
            # Mann-Whitney U split into half-cohorts
            half = n_cells // 2
            run_differential_test(
                store,
                DifferentialTestInput(
                    dataset_id=DATASET_ID,
                    metric_name="sc_3d_distance",
                    group_a_label="Low",
                    group_b_label="High",
                    group_a_values=dists[:half].tolist(),
                    group_b_values=dists[half:].tolist(),
                    test_type="mann_whitney_u",
                ),
            )

        mean_ms, p95_ms = _bench(_sc_workload, n_runs=5)
        scale_rows.append({"n_cells": n_cells, "mean_ms": mean_ms, "p95_ms": p95_ms})

    out_dir = REPO_ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(tool_rows).to_csv(out_dir / "latency_tools.csv", index=False)
    pd.DataFrame(scale_rows).to_csv(out_dir / "latency_scalability.csv", index=False)
    (out_dir / "latency_profile.json").write_text(
        json.dumps({"tools": tool_rows, "stages": stage_rows, "scalability": scale_rows}, indent=2)
    )

    print("=== Per-Tool Deterministic Latency (ms) ===")
    print(pd.DataFrame(tool_rows).to_string(index=False))
    print("\n=== Pipeline Stage Breakdown (ms) ===")
    print(pd.DataFrame(stage_rows).to_string(index=False))
    print("\n=== Cohort Scalability (100–2,000 Cells, ms) ===")
    print(pd.DataFrame(scale_rows).to_string(index=False))


if __name__ == "__main__":
    main()