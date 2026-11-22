export type ResolutionBp = 25000 | 50000 | 100000 | 250000 | 500000 | 1000000;
export type GenomeBuild = "hg38" | "hg19" | "mm10";
export type DatasetMode = "bulk_condition" | "single_cell";
export type BinRole =
  | "anchor"
  | "enhancer"
  | "promoter"
  | "boundary"
  | "neighbor";

export interface GenomicInterval {
  chrom: string;
  start: number;
  end: number;
}

export interface CohortSpec {
  cohort_id: string;
  label: string;
  entity_ids: string[];
  color_hex: string;
}

export interface HighlightBinSpec {
  chrom: string;
  bin_index: number;
  label: string;
  role: BinRole;
}

export interface SpatialSelection3D {
  selected_bins: number[];
  radius_nm: number | null;
}

export interface Camera3D {
  position: [number, number, number];
  target: [number, number, number];
}

export interface ViewportState {
  schema_version: "1.0.0";
  dataset_id: string;
  dataset_mode: DatasetMode;
  genome_build: GenomeBuild;
  locus: GenomicInterval;
  resolution_bp: ResolutionBp;
  active_cohorts: CohortSpec[];
  active_1d_tracks: string[];
  highlighted_bins: HighlightBinSpec[];
  spatial_selection_3d: SpatialSelection3D;
  camera_3d: Camera3D;
  bookmark_id: string | null;
}
