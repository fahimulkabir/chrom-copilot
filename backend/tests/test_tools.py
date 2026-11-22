# Tests all 8 deterministic tools, the data streaming routes, edge-case safeguards (bulk n=1 statistical downgrade), and verifies the under-1.5-second latency Exit Gate on your real GM12878 and K562 data.

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
DATASET_ID = "rao2014_gm12878_k562"

GM_COHORT = {"cohort_id": "GM12878", "label": "GM12878", "entity_ids": ["GM12878"]}
K562_COHORT = {"cohort_id": "K562", "label": "K562", "entity_ids": ["K562"]}
BOTH_COHORT = {
    "cohort_id": "all",
    "label": "GM12878 + K562",
    "entity_ids": ["GM12878", "K562"],
}


def test_tool1_stratify_1d_hbb():
    t0 = time.perf_counter()
    resp = client.post(
        "/tools/stratify-1d",
        json={
            "dataset_id": DATASET_ID,
            "feature_name": "HBB",
            "source_cohort": BOTH_COHORT,
            "split_method": "condition_group",
        },
    )
    elapsed = time.perf_counter() - t0
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["tool_call_id"].startswith("tc_")
    assert data["high_cohort"]["entity_ids"] == ["K562"]
    assert data["low_cohort"]["entity_ids"] == ["GM12878"]
    assert data["summary_stats"]["high_mean"] > data["summary_stats"]["low_mean"]
    assert elapsed < 1.5


def test_tool2_contact_map_myc_ice_and_oe():
    for norm in ("ice_balanced", "obs_over_exp"):
        t0 = time.perf_counter()
        resp = client.post(
            "/tools/contact-map",
            json={
                "dataset_id": DATASET_ID,
                "cohort": GM_COHORT,
                "locus": {"chrom": "chr8", "start": 127_500_000, "end": 130_000_000},
                "resolution_bp": 50_000,
                "normalization": norm,
            },
        )
        elapsed = time.perf_counter() - t0
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["n_bins"] == 50
        assert len(data["sparse_pixels"]) > 100
        assert 0.0 <= data["contact_sparsity_fraction"] < 0.5
        assert elapsed < 1.5


def test_tool3_tad_insulation_myc():
    t0 = time.perf_counter()
    resp = client.post(
        "/tools/tad-insulation",
        json={
            "dataset_id": DATASET_ID,
            "cohort": GM_COHORT,
            "locus": {"chrom": "chr8", "start": 127_500_000, "end": 130_000_000},
            "resolution_bp": 50_000,
            "window_bp": 250_000,
            "min_boundary_strength": 0.15,
        },
    )
    elapsed = time.perf_counter() - t0
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data["bin_positions"]) == 50
    assert len(data["insulation_scores"]) == 50
    assert any(v is not None for v in data["insulation_scores"])
    assert elapsed < 1.5


def test_tool4_ab_compartments_and_switches():
    payload = {
        "dataset_id": DATASET_ID,
        "cohort": GM_COHORT,
        "compare_cohort": K562_COHORT,
        "chrom": "chr18",
        "resolution_bp": 250_000,
    }
    # First call populates disk + memory cache (allows cold Numba/eigendecomp warmup)
    cold = client.post("/tools/ab-compartments", json=payload)
    assert cold.status_code == 200, cold.text

    # Interactive cached call must stay well under 1.5s
    t0 = time.perf_counter()
    resp = client.post("/tools/ab-compartments", json=payload)
    elapsed = time.perf_counter() - t0
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data["bin_starts"]) > 200
    assert any(v is not None for v in data["e1_scores"])
    assert isinstance(data["switches"], list)
    assert len(data["switches"]) > 0
    assert elapsed < 1.5


def test_tool5_3d_pairwise_distance_hbb_lcr():
    t0 = time.perf_counter()
    resp = client.post(
        "/tools/3d-distance",
        json={
            "dataset_id": DATASET_ID,
            "cohort": BOTH_COHORT,
            "locus_a": {"chrom": "chr11", "start": 5_246_696, "end": 5_248_301},  # HBB
            "locus_b": {"chrom": "chr11", "start": 5_295_000, "end": 5_315_000},  # HBB_LCR
            "resolution_bp": 50_000,
            "normalize_by_radius_of_gyration": True,
        },
    )
    elapsed = time.perf_counter() - t0
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert set(data["entity_distances"].keys()) == {"GM12878", "K562"}
    assert data["units"] == "rg_normalized"
    assert data["mean_distance"] > 0
    assert elapsed < 1.5


def test_tool6_3d_spatial_neighbors_myc():
    myc_bin_50kb = 128_750_000 // 50_000  # bin 2575
    t0 = time.perf_counter()
    resp = client.post(
        "/tools/3d-neighbors",
        json={
            "dataset_id": DATASET_ID,
            "cohort": K562_COHORT,
            "anchor_chrom": "chr8",
            "anchor_bin_index": myc_bin_50kb,
            "resolution_bp": 50_000,
            "radius_threshold": 5.0,
            "min_linear_separation_bins": 3,
        },
    )
    elapsed = time.perf_counter() - t0
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["anchor_bin_index"] == myc_bin_50kb
    assert len(data["neighbors"]) > 0
    assert all(abs(nb["bin_index"] - myc_bin_50kb) >= 3 for nb in data["neighbors"])
    assert elapsed < 1.5


def test_tool7_differential_test_safeguard_and_valid_stats():
    # 1. Bulk n=1 safeguard: must refuse fake p-value and downgrade to delta_only
    r_bulk = client.post(
        "/tools/differential-test",
        json={
            "dataset_id": DATASET_ID,
            "metric_name": "3d_distance_HBB_LCR",
            "group_a_label": "GM12878",
            "group_b_label": "K562",
            "group_a_values": [0.45],
            "group_b_values": [0.22],
            "test_type": "mann_whitney_u",
        },
    )
    assert r_bulk.status_code == 200, r_bulk.text
    d_bulk = r_bulk.json()
    assert d_bulk["statistically_valid"] is False
    assert d_bulk["test_type"] == "delta_only"
    assert d_bulk["p_value"] is None
    assert len(d_bulk["warnings"]) >= 1

    # 2. Multi-value test (e.g., bin-wise insulation or single-cell cohort): valid p-value
    r_valid = client.post(
        "/tools/differential-test",
        json={
            "dataset_id": DATASET_ID,
            "metric_name": "insulation_profile",
            "group_a_label": "GM12878",
            "group_b_label": "K562",
            "group_a_values": [0.1, 0.15, 0.2, 0.18, 0.22],
            "group_b_values": [-0.5, -0.45, -0.6, -0.52, -0.48],
            "test_type": "mann_whitney_u",
        },
    )
    assert r_valid.status_code == 200, r_valid.text
    d_valid = r_valid.json()
    assert d_valid["statistically_valid"] is True
    assert d_valid["p_value"] is not None and d_valid["p_value"] < 0.05


def test_tool8_viewport_sync_and_bookmark_restore():
    myc_bin_50kb = 128_750_000 // 50_000
    resp = client.post(
        "/tools/viewport-sync",
        json={
            "dataset_id": DATASET_ID,
            "locus": {"chrom": "chr8", "start": 127_500_000, "end": 130_000_000},
            "resolution_bp": 50_000,
            "active_cohorts": [GM_COHORT, K562_COHORT],
            "highlighted_bins": [
                {"chrom": "chr8", "bin_index": myc_bin_50kb, "label": "MYC", "role": "promoter"}
            ],
            "camera_focus_bin": myc_bin_50kb,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    bk_id = data["bookmark_id"]
    assert bk_id.startswith("bk_")
    assert "camera_3d" in data["viewport_patch"]

    # Confirm bookmark persists and can be restored via /data/bookmarks/{id}
    restore = client.get(f"/data/bookmarks/{bk_id}")
    assert restore.status_code == 200
    assert restore.json()["bookmark_id"] == bk_id


def test_ui_data_streaming_routes():
    assert client.get("/data/metadata").status_code == 200
    g = client.get("/data/genes", params={"chrom": "chr8", "start": 127_500_000, "end": 130_000_000})
    assert g.status_code == 200 and any(row["symbol"] == "MYC" for row in g.json())

    s3d = client.get(
        "/data/structure-3d",
        params={"entity_id": "GM12878", "chrom": "chr8", "start": 127_500_000, "end": 130_000_000, "resolution_bp": 50_000},
    )
    assert s3d.status_code == 200 and len(s3d.json()) == 50