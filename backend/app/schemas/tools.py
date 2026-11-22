from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator
from app.schemas.viewport import CohortSpec, GenomicInterval, HighlightBinSpec


# ---------------------------------------------------------------------------
# Shared Base Primitives & Provenance Wrapper
# ---------------------------------------------------------------------------



class ProvenancedResult(BaseModel):
    tool_call_id: str = Field(default_factory=lambda: f"tc_{uuid4().hex[:10]}")
    tool_name: str
    dataset_id: str
    executed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    n_entities_used: int = Field(..., ge=0)
    contact_sparsity_fraction: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Fraction of zero entries in queried Hi-C region."
    )
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Tool 1: Stratify Cohort / Extract 1D Feature Signal
# ---------------------------------------------------------------------------

class StratifyBy1DInput(BaseModel):
    dataset_id: str
    feature_name: str = Field(..., description="Gene symbol (e.g., 'MYC') or 1D track key.")
    source_cohort: CohortSpec
    split_method: Literal["quantile", "threshold", "condition_group"] = "quantile"
    high_cutoff: float = Field(default=0.75, description="Upper quantile (0-1) or raw signal cutoff.")
    low_cutoff: float = Field(default=0.25, description="Lower quantile (0-1) or raw signal cutoff.")


class StratifyBy1DOutput(ProvenancedResult):
    tool_name: Literal["stratify_by_1d_feature"] = "stratify_by_1d_feature"
    feature_name: str
    high_cohort: CohortSpec
    low_cohort: CohortSpec
    summary_stats: dict[str, float] = Field(
        ..., description="Keys: high_mean, low_mean, high_median, low_median."
    )


# ---------------------------------------------------------------------------
# Tool 2: Fetch 2D Contact Map (Bulk or Single-Cell Pseudobulk)
# ---------------------------------------------------------------------------

class ContactMapInput(BaseModel):
    dataset_id: str
    cohort: CohortSpec
    locus: GenomicInterval
    resolution_bp: Literal[25000, 50000, 100000, 250000, 500000, 1000000] = 50000
    normalization: Literal["raw", "ice_balanced", "obs_over_exp"] = "ice_balanced"


class SparsePixel(BaseModel):
    bin_i: int
    bin_j: int
    value: float


class ContactMapOutput(ProvenancedResult):
    tool_name: Literal["get_contact_map"] = "get_contact_map"
    locus: GenomicInterval
    resolution_bp: int
    n_bins: int
    total_contacts: float
    sparse_pixels: list[SparsePixel] = Field(
        ..., description="Upper-triangle non-zero pixels for fast Gosling.js rendering."
    )


# ---------------------------------------------------------------------------
# Tool 3: Compute TAD Insulation Score & Boundary Calls (cooltools-backed)
# ---------------------------------------------------------------------------

class TadInsulationInput(BaseModel):
    dataset_id: str
    cohort: CohortSpec
    locus: GenomicInterval
    resolution_bp: Literal[25000, 50000, 100000] = 50000
    window_bp: int = Field(default=250000, ge=50000, description="Sliding diamond window size in bp.")
    min_boundary_strength: float = Field(default=0.2, ge=0.0)


class BoundaryCall(BaseModel):
    bin_index: int
    genomic_pos: int
    insulation_score: float
    boundary_strength: float


class TadInsulationOutput(ProvenancedResult):
    tool_name: Literal["compute_tad_insulation"] = "compute_tad_insulation"
    bin_positions: list[int]
    insulation_scores: list[Optional[float]]
    boundaries: list[BoundaryCall]


# ---------------------------------------------------------------------------
# Tool 4: Compute A/B Compartments (E1 Eigenvector phased by GC/gene density)
# ---------------------------------------------------------------------------

class ABCompartmentInput(BaseModel):
    dataset_id: str
    cohort: CohortSpec
    chrom: str = Field(..., pattern=r"^chr([0-9]+|X|Y)$")
    resolution_bp: Literal[100000, 250000, 500000, 1000000] = 250000
    compare_cohort: Optional[CohortSpec] = Field(
        default=None, description="Optional second cohort to detect A->B or B->A switches."
    )


class CompartmentSwitch(BaseModel):
    bin_index: int
    start: int
    end: int
    e1_primary: float
    e1_compare: float
    transition: Literal["A_to_B", "B_to_A"]


class ABCompartmentOutput(ProvenancedResult):
    tool_name: Literal["compute_ab_compartments"] = "compute_ab_compartments"
    chrom: str
    resolution_bp: int
    bin_starts: list[int]
    e1_scores: list[Optional[float]]
    switches: list[CompartmentSwitch] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Tool 5: Measure 3D Pairwise Euclidean Distance Between Two Loci
# ---------------------------------------------------------------------------

class Pairwise3DDistanceInput(BaseModel):
    dataset_id: str
    cohort: CohortSpec
    locus_a: GenomicInterval
    locus_b: GenomicInterval
    resolution_bp: Literal[25000, 50000, 100000, 250000, 500000] = 50000
    normalize_by_radius_of_gyration: bool = Field(
        default=True, description="Scale distances by chromosome Rg so 3D folds compare cleanly."
    )


class Pairwise3DDistanceOutput(ProvenancedResult):
    tool_name: Literal["measure_3d_pairwise_distance"] = "measure_3d_pairwise_distance"
    entity_distances: dict[str, float] = Field(
        ..., description="Mapping of entity_id -> 3D Euclidean distance (raw or Rg-normalized)."
    )
    mean_distance: float
    median_distance: float
    iqr_distance: float
    units: Literal["rg_normalized", "model_units"]


# ---------------------------------------------------------------------------
# Tool 6: Find 3D Spatial Neighbors (Powers "Point-to-Query" in Three.js)
# ---------------------------------------------------------------------------

class SpatialNeighbors3DInput(BaseModel):
    dataset_id: str
    cohort: CohortSpec
    anchor_chrom: str = Field(..., pattern=r"^chr([0-9]+|X|Y)$")
    anchor_bin_index: int = Field(..., ge=0)
    resolution_bp: Literal[25000, 50000, 100000, 250000, 500000] = 50000
    radius_threshold: float = Field(..., gt=0.0)
    min_linear_separation_bins: int = Field(
        default=3, ge=1, description="Exclude trivial polymer-backbone adjacent bins."
    )


class SpatialNeighborBin(BaseModel):
    chrom: str
    bin_index: int
    start: int
    end: int
    mean_3d_distance: float
    fraction_entities_within_radius: float = Field(..., ge=0.0, le=1.0)
    overlapping_genes: list[str] = Field(default_factory=list)


class SpatialNeighbors3DOutput(ProvenancedResult):
    tool_name: Literal["find_3d_spatial_neighbors"] = "find_3d_spatial_neighbors"
    anchor_bin_index: int
    neighbors: list[SpatialNeighborBin]


# ---------------------------------------------------------------------------
# Tool 7: Deterministic Differential Statistical Test
# ---------------------------------------------------------------------------

class DifferentialTestInput(BaseModel):
    dataset_id: str
    metric_name: str = Field(..., description="e.g., '3d_distance_MYC_enhancer' or 'tad_insulation'")
    group_a_label: str
    group_b_label: str
    group_a_values: list[float] = Field(..., min_length=1)
    group_b_values: list[float] = Field(..., min_length=1)
    test_type: Literal["mann_whitney_u", "permutation", "welch_t", "delta_only"] = "mann_whitney_u"
    n_permutations: int = Field(default=1000, ge=100, le=10000)


class DifferentialTestOutput(ProvenancedResult):
    tool_name: Literal["run_differential_test"] = "run_differential_test"
    metric_name: str
    test_type: str
    group_a_median: float
    group_b_median: float
    effect_size: float = Field(..., description="Rank-biserial r, Cohen's d, or raw delta.")
    statistic: Optional[float] = None
    p_value: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    statistically_valid: bool = Field(
        ..., description="False if sample size is too small (e.g., bulk n=1 per condition)."
    )


# ---------------------------------------------------------------------------
# Tool 8: Emit Atomic Viewport Synchronization Command ("Query-to-View")
# ---------------------------------------------------------------------------

class EmitViewportSyncInput(BaseModel):
    dataset_id: str
    locus: GenomicInterval
    resolution_bp: Literal[25000, 50000, 100000, 250000, 500000, 1000000]
    active_cohorts: list[CohortSpec] = Field(..., min_length=1, max_length=4)
    highlighted_bins: list[HighlightBinSpec] = Field(default_factory=list)
    camera_focus_bin: Optional[int] = Field(
        default=None, description="3D bin index to smoothly center the Three.js camera on."
    )


class EmitViewportSyncOutput(ProvenancedResult):
    tool_name: Literal["emit_viewport_sync"] = "emit_viewport_sync"
    bookmark_id: str
    viewport_patch: dict = Field(..., description="Validated JSON patch applied to Zustand store.")

