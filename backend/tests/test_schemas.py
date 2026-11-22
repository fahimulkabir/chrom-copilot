import pytest
from pydantic import ValidationError

from app.schemas.agent import CopilotResponse, PlanVerificationResult, ProvenanceClaim
from app.schemas.tools import DifferentialTestOutput, Pairwise3DDistanceInput
from app.schemas.viewport import CohortSpec, GenomicInterval, ViewportState


def test_viewport_state_valid():
    state = ViewportState(
        dataset_id="prostate_bulk_hic",
        dataset_mode="bulk_condition",
        genome_build="hg38",
        locus=GenomicInterval(chrom="chr8", start=127000000, end=129500000),
        resolution_bp=50000,
        active_cohorts=[
            CohortSpec(cohort_id="normal", label="RWPE1", entity_ids=["RWPE1"])
        ],
        camera_3d={"position": (10.0, 10.0, 10.0), "target": (0.0, 0.0, 0.0)},
    )
    assert state.schema_version == "1.0.0"
    assert state.locus.chrom == "chr8"


def test_invalid_genomic_interval_rejected():
    with pytest.raises(ValidationError):
        GenomicInterval(chrom="chr8", start=129500000, end=127000000)


def test_tool_and_provenance_contracts():
    req = Pairwise3DDistanceInput(
        dataset_id="prostate_bulk_hic",
        cohort=CohortSpec(cohort_id="tumor", label="LNCaP", entity_ids=["LNCaP"]),
        locus_a=GenomicInterval(chrom="chr8", start=127700000, end=127750000),
        locus_b=GenomicInterval(chrom="chr8", start=128000000, end=128050000),
    )
    assert req.normalize_by_radius_of_gyration is True

    res = DifferentialTestOutput(
        dataset_id="prostate_bulk_hic",
        n_entities_used=2,
        metric_name="3d_distance_MYC",
        test_type="delta_only",
        group_a_median=0.42,
        group_b_median=0.29,
        effect_size=-0.13,
        statistically_valid=False,
    )
    assert res.tool_call_id.startswith("tc_")

    claim = ProvenanceClaim(
        claim_id="c1",
        text="3D distance decreased by 0.13 Rg units.",
        cited_tool_call_ids=[res.tool_call_id],
        is_verified=True,
    )
    resp = CopilotResponse(
        session_id="s1",
        claims=[claim],
        executed_tool_call_ids=[res.tool_call_id],
    )
    assert resp.claims[0].is_verified is True


def test_plan_verification_contract():
    gate = PlanVerificationResult(
        status="needs_confirmation",
        plan=[],
        checks_passed=["genome_build_match"],
        warnings=["Contact sparsity exceeds 95% at 50 kb."],
        clarification_prompt="Switch to 100 kb resolution?",
    )
    assert gate.status == "needs_confirmation"