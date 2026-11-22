from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator


ResolutionBp = Literal[25000, 50000, 100000, 250000, 500000, 1000000]
GenomeBuild = Literal["hg38", "hg19", "mm10"]
DatasetMode = Literal["bulk_condition", "single_cell"]
BinRole = Literal["anchor", "enhancer", "promoter", "boundary", "neighbor"]


class GenomicInterval(BaseModel):
    chrom: str = Field(..., pattern=r"^chr([0-9]+|X|Y)$")
    start: int = Field(..., ge=0)
    end: int = Field(..., gt=0)

    @model_validator(mode="after")
    def validate_bounds(self) -> GenomicInterval:
        if self.end <= self.start:
            raise ValueError("end must be strictly greater than start")
        return self


class CohortSpec(BaseModel):
    cohort_id: str
    label: str
    entity_ids: list[str] = Field(
        ...,
        min_length=1,
        description="Condition/sample IDs (bulk_condition) or cell barcodes (single_cell).",
    )
    color_hex: str = Field(default="#2563EB", pattern=r"^#[0-9A-Fa-f]{6}$")


class HighlightBinSpec(BaseModel):
    chrom: str = Field(..., pattern=r"^chr([0-9]+|X|Y)$")
    bin_index: int = Field(..., ge=0)
    label: str
    role: BinRole


class SpatialSelection3D(BaseModel):
    selected_bins: list[int] = Field(default_factory=list)
    radius_nm: Optional[float] = Field(default=None, ge=0.0)


class Camera3D(BaseModel):
    position: tuple[float, float, float]
    target: tuple[float, float, float]


class ViewportState(BaseModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    dataset_id: str
    dataset_mode: DatasetMode
    genome_build: GenomeBuild
    locus: GenomicInterval
    resolution_bp: ResolutionBp
    active_cohorts: list[CohortSpec] = Field(..., min_length=1, max_length=4)
    active_1d_tracks: list[str] = Field(default_factory=list)
    highlighted_bins: list[HighlightBinSpec] = Field(default_factory=list)
    spatial_selection_3d: SpatialSelection3D = Field(default_factory=SpatialSelection3D)
    camera_3d: Camera3D
    bookmark_id: Optional[str] = None