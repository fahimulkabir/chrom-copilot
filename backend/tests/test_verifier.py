from __future__ import annotations

from fastapi.testclient import TestClient

from app.agent.binder import enforce_provenance_guardrails
from app.agent.verifier import verify_plan
from app.engine.data_store import get_data_store
from app.main import app
from app.schemas.agent import PlannedToolCall, ProvenanceClaim
from app.schemas.viewport import CohortSpec, GenomicInterval, ViewportState

client = TestClient(app)
DATASET_ID = "rao2014_gm12878_k562"


def _make_viewport(
    *,
    chrom: str = "chr8",
    start: int = 127_500_000,
    end: int = 130_000_000,
    resolution_bp: int = 50_000,
    selected_bins: list[int] | None = None,
    genome_build: str = "hg19",
    dataset_mode: str = "bulk_condition",
) -> ViewportState:
    return ViewportState(
        dataset_id=DATASET_ID,
        dataset_mode=dataset_mode,
        genome_build=genome_build,
        locus=GenomicInterval(chrom=chrom, start=start, end=end),
        resolution_bp=resolution_bp,
        active_cohorts=[
            CohortSpec(cohort_id="GM12878", label="GM12878", entity_ids=["GM12878"]),
            CohortSpec(cohort_id="K562", label="K562", entity_ids=["K562"]),
        ],
        spatial_selection_3d={"selected_bins": selected_bins or [], "radius_nm": 4.5},
        camera_3d={"position": (12.0, 9.0, 16.0), "target": (0.0, 0.0, 0.0)},
    )


# ---------------------------------------------------------------------------
# 1. Deterministic Plan Verification Gate Trap Tests (100% Catch Rate)
# ---------------------------------------------------------------------------

def test_verifier_blocks_disallowed_tool():
    store = get_data_store()
    plan = [
        PlannedToolCall(
            step_id="step_1",
            tool_name="execute_arbitrary_python",
            arguments={"code": "import os"},
            rationale="Attempt sandbox escape",
        )
    ]
    res = verify_plan(store, plan, _make_viewport())
    assert res.status == "blocked"
    assert any("disallowed tool" in w for w in res.warnings)


def test_verifier_blocks_genome_build_mismatch():
    store = get_data_store()
    vp = _make_viewport(genome_build="hg38")
    plan = [
        PlannedToolCall(
            step_id="step_1",
            tool_name="get_contact_map",
            arguments={
                "dataset_id": DATASET_ID,
                "cohort": {"cohort_id": "GM12878", "label": "GM12878", "entity_ids": ["GM12878"]},
                "locus": {"chrom": "chr8", "start": 127_500_000, "end": 130_000_000},
                "resolution_bp": 50_000,
                "normalization": "ice_balanced",
            },
            rationale="Test build mismatch",
        )
    ]
    res = verify_plan(store, plan, vp)
    assert res.status == "blocked"
    assert any("Genome build mismatch" in w for w in res.warnings)


def test_verifier_blocks_out_of_bounds_and_bad_resolution():
    store = get_data_store()
    plan = [
        PlannedToolCall(
            step_id="step_1",
            tool_name="get_contact_map",
            arguments={
                "dataset_id": DATASET_ID,
                "cohort": {"cohort_id": "GM12878", "label": "GM12878", "entity_ids": ["GM12878"]},
                "locus": {"chrom": "chr18", "start": 70_000_000, "end": 999_000_000},
                "resolution_bp": 25_000,  # Not built in Part 1 (50k/100k/250k)
                "normalization": "ice_balanced",
            },
            rationale="Out of bounds + unavailable resolution",
        )
    ]
    res = verify_plan(store, plan, _make_viewport())
    assert res.status == "blocked"
    assert any("unavailable resolution" in w for w in res.warnings)
    assert any("exceeds chromosome bounds" in w for w in res.warnings)


def test_verifier_catches_high_sparsity_trap(monkeypatch):
    import numpy as np
    import pandas as pd

    store = get_data_store()
    # Simulate a sparse contact matrix (99.9% zeros, like single-cell or dropout regions)
    sparse_mat = np.zeros((50, 50), dtype=np.float64)
    sparse_mat[0, 0] = 2.0
    monkeypatch.setattr(
        store, "get_matrix", lambda *args, **kwargs: (sparse_mat, pd.DataFrame())
    )

    plan = [
        PlannedToolCall(
            step_id="step_1",
            tool_name="get_contact_map",
            arguments={
                "dataset_id": DATASET_ID,
                "cohort": {"cohort_id": "K562", "label": "K562", "entity_ids": ["K562"]},
                "locus": {"chrom": "chr8", "start": 127_500_000, "end": 130_000_000},
                "resolution_bp": 50_000,
                "normalization": "ice_balanced",
            },
            rationale="Test high-sparsity gate against default 95% threshold",
        )
    ]
    res = verify_plan(store, plan, _make_viewport())  # tests real 0.95 threshold
    assert res.status == "needs_confirmation"
    assert any("contact sparsity" in w.lower() for w in res.warnings)
    assert res.clarification_prompt is not None


def test_verifier_catches_small_single_cell_cohort():
    store = get_data_store()
    vp = _make_viewport(dataset_mode="single_cell")
    small_cells = [f"cell_{i}" for i in range(8)]
    plan = [
        PlannedToolCall(
            step_id="step_1",
            tool_name="measure_3d_pairwise_distance",
            arguments={
                "dataset_id": DATASET_ID,
                "cohort": {"cohort_id": "sc_tiny", "label": "Tiny SC", "entity_ids": small_cells},
                "locus_a": {"chrom": "chr11", "start": 5_246_696, "end": 5_248_301},
                "locus_b": {"chrom": "chr11", "start": 5_295_000, "end": 5_315_000},
                "resolution_bp": 50_000,
                "normalize_by_radius_of_gyration": True,
            },
            rationale="Small sc cohort trap",
        )
    ]
    res = verify_plan(store, plan, vp, min_sc_cells=20)
    assert res.status == "needs_confirmation"
    assert any("only 8 single cells" in w for w in res.warnings)


def test_verifier_catches_bulk_n1_stat_test():
    store = get_data_store()
    plan = [
        PlannedToolCall(
            step_id="step_1",
            tool_name="run_differential_test",
            arguments={
                "dataset_id": DATASET_ID,
                "metric_name": "3d_distance_HBB",
                "group_a_label": "GM12878",
                "group_b_label": "K562",
                "group_a_values": [0.42],
                "group_b_values": [0.21],
                "test_type": "mann_whitney_u",
            },
            rationale="Bulk n=1 stat test trap",
        )
    ]
    res = verify_plan(store, plan, _make_viewport())
    assert res.status == "needs_confirmation"
    assert any("delta_only" in w for w in res.warnings)


def test_point_to_query_without_selected_bead_prompts_clarification():
    vp = _make_viewport(selected_bins=[])
    resp = client.post(
        "/agent/plan",
        json={
            "session_id": "test_trap",
            "user_query": "What genes are physically near this bead in 3D?",
            "viewport": vp.model_dump(),
            "use_llm": False,
        },
    )
    assert resp.status_code == 200, resp.text
    ver = resp.json()["verification"]
    assert ver["status"] == "needs_confirmation"
    assert ver["plan"] == []
    assert "click a bead" in ver["clarification_prompt"].lower()


# ---------------------------------------------------------------------------
# 2. Provenance Binder Zero-Hallucination Guardrail Test
# ---------------------------------------------------------------------------

def test_provenance_guardrail_strips_fake_ids_and_flags_ungrounded_numbers():
    raw_claims = [
        ProvenanceClaim(
            claim_id="c1",
            text="Valid deterministic distance was 0.312 Rg.",
            cited_tool_call_ids=["tc_real0001"],
            is_verified=True,
        ),
        ProvenanceClaim(
            claim_id="c2",
            text="MYC contact frequency jumped 4.8x (p = 0.0001).",
            cited_tool_call_ids=["tc_hallucinated"],
            is_verified=True,
        ),
    ]
    checked = enforce_provenance_guardrails(
        raw_claims, executed_ids={"tc_real0001"}, bookmark_id="bk_12345678"
    )
    assert checked[0].is_verified is True
    assert checked[0].cited_tool_call_ids == ["tc_real0001"]
    assert checked[0].bookmark_id == "bk_12345678"

    assert checked[1].is_verified is False
    assert checked[1].cited_tool_call_ids == []
    assert checked[1].text.startswith("[Hypothesis / Unverified]")


# ---------------------------------------------------------------------------
# 3. 10-Query Multi-Scale Grounded End-to-End Verification (Part 4 Exit Gate)
# ---------------------------------------------------------------------------

E2E_QUERIES = [
    ("Compare MYC TAD insulation between GM12878 and K562 at 50kb", _make_viewport()),
    ("Measure the 3D distance between HBB and HBB_LCR", _make_viewport()),
    ("Stratify cohorts by HBB expression and fetch the contact map", _make_viewport()),
    ("Detect A/B compartment switches on chr18 between GM12878 and K562", _make_viewport()),
    (
        "What genes are physically near this bead in 3D for K562?",
        _make_viewport(chrom="chr8", start=127_500_000, end=130_000_000, selected_bins=[2575]),
    ),
    ("Stratify by BCL2 expression and check TAD insulation at BCL2", _make_viewport()),
    ("Measure 3D distance between GATA2 and GATA2_ENH at 50kb", _make_viewport()),
    (
        "Find 3D spatial neighbors around the selected bead in GM12878",
        _make_viewport(chrom="chr11", start=4_000_000, end=6_500_000, selected_bins=[105]),
    ),
    ("Fetch the obs/exp contact map for K562 at the BCL2 locus at 100kb", _make_viewport()),
    ("Compute A/B compartment switches on chr11 at 250kb", _make_viewport()),
]


def test_ten_diverse_multiscale_queries_end_to_end():
    for idx, (query, vp) in enumerate(E2E_QUERIES, start=1):
        # Phase 1: Plan & Verify
        plan_resp = client.post(
            "/agent/plan",
            json={
                "session_id": f"e2e_{idx}",
                "user_query": query,
                "viewport": vp.model_dump(),
                "use_llm": False,
            },
        )
        assert plan_resp.status_code == 200, f"Query {idx} plan failed: {plan_resp.text}"
        pdata = plan_resp.json()
        ver = pdata["verification"]
        assert ver["status"] == "approved", f"Query {idx} unexpectedly {ver['status']}: {ver['warnings']}"
        assert len(ver["plan"]) >= 1

        # Phase 2: Execute Approved Plan & Bind Provenance
        exec_resp = client.post(
            "/agent/execute",
            json={
                "session_id": f"e2e_{idx}",
                "plan_id": pdata["plan_id"],
                "user_query": query,
                "viewport": vp.model_dump(),
                "approved_plan": ver["plan"],
                "use_llm": False,
            },
        )
        assert exec_resp.status_code == 200, f"Query {idx} execute failed: {exec_resp.text}"
        cdata = exec_resp.json()
        executed_ids = set(cdata["executed_tool_call_ids"])
        assert len(executed_ids) == len(ver["plan"])
        assert len(cdata["claims"]) >= 1

        # Verify 100% claim-to-evidence binding & ledger auditability
        for claim in cdata["claims"]:
            assert claim["is_verified"] is True, f"Query {idx} unverified claim: {claim}"
            assert len(claim["cited_tool_call_ids"]) >= 1
            for tid in claim["cited_tool_call_ids"]:
                assert tid in executed_ids
                prov = client.get(f"/agent/provenance/{tid}")
                assert prov.status_code == 200
                assert prov.json()["tool_call_id"] == tid