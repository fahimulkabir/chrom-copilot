// Connects directly to our Part 2 FastAPI endpoints (http://127.0.0.1:8000).

import type {
  CohortSpec,
  GenomicInterval,
  HighlightBinSpec,
  ResolutionBp,
  ViewportState,
} from "../types/viewport";

import type {
  AgentExecuteRequest,
  AgentPlanRequest,
  AgentPlanResponse,
  CopilotResponse,
  ProvenanceRecord,
} from "../types/agent";

const API_BASE = "http://127.0.0.1:8000";

export interface DatasetMetadata {
  dataset_id: string;
  dataset_mode: "bulk_condition" | "single_cell";
  genome_build: "hg38" | "hg19" | "mm10";
  available_resolutions_bp: ResolutionBp[];
  cohorts: Record<
    string,
    {
      cohort_id: string;
      label: string;
      entity_ids: string[];
      color_hex: string;
    }
  >;
  target_loci: Record<
    string,
    {
      chrom: string;
      start: number;
      end: number;
      key_genes: string[];
      key_enhancers: string[];
    }
  >;
  gene_lookup: Record<
    string,
    {
      chrom: string;
      start: number;
      end: number;
      feature_type: string;
      bin_50kb: number;
      bin_100kb: number;
      bin_250kb: number;
      GM12878_tpm: number;
      K562_tpm: number;
    }
  >;
}

export interface GeneFeature {
  symbol: string;
  chrom: string;
  start: number;
  end: number;
  strand: string;
  feature_type: string;
  GM12878_tpm: number | null;
  K562_tpm: number | null;
}

export interface Bead3D {
  bin_index: number;
  start: number;
  end: number;
  x: number;
  y: number;
  z: number;
  rg: number;
}

export interface SparsePixel {
  bin_i: number;
  bin_j: number;
  value: number;
}

export interface ContactMapResponse {
  tool_call_id: string;
  n_bins: number;
  total_contacts: number;
  contact_sparsity_fraction: number;
  sparse_pixels: SparsePixel[];
}

export interface TadInsulationResponse {
  tool_call_id: string;
  bin_positions: number[];
  insulation_scores: (number | null)[];
  boundaries: {
    bin_index: number;
    genomic_pos: number;
    insulation_score: number;
    boundary_strength: number;
  }[];
}

export interface ViewportSyncResponse {
  tool_call_id: string;
  bookmark_id: string;
  viewport_patch: Partial<ViewportState>;
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`API ${res.status}: ${detail}`);
  }
  return (await res.json()) as T;
}

export const api = {
  getMetadata(): Promise<DatasetMetadata> {
    return fetchJson(`${API_BASE}/data/metadata`);
  },

  getGenes(locus: GenomicInterval): Promise<GeneFeature[]> {
    const q = new URLSearchParams({
      chrom: locus.chrom,
      start: String(locus.start),
      end: String(locus.end),
    });
    return fetchJson(`${API_BASE}/data/genes?${q}`);
  },

  getStructure3D(
    entityId: string,
    locus: GenomicInterval,
    resolutionBp: ResolutionBp,
  ): Promise<Bead3D[]> {
    const q = new URLSearchParams({
      entity_id: entityId,
      chrom: locus.chrom,
      start: String(locus.start),
      end: String(locus.end),
      resolution_bp: String(resolutionBp),
    });
    return fetchJson(`${API_BASE}/data/structure-3d?${q}`);
  },

  getContactMap(params: {
    dataset_id: string;
    cohort: CohortSpec;
    locus: GenomicInterval;
    resolution_bp: ResolutionBp;
    normalization?: "raw" | "ice_balanced" | "obs_over_exp";
  }): Promise<ContactMapResponse> {
    return fetchJson(`${API_BASE}/tools/contact-map`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        normalization: "ice_balanced",
        ...params,
      }),
    });
  },

  getTadInsulation(params: {
    dataset_id: string;
    cohort: CohortSpec;
    locus: GenomicInterval;
    resolution_bp: ResolutionBp;
  }): Promise<TadInsulationResponse> {
    const windowBp = Math.max(250_000, params.resolution_bp * 5);
    return fetchJson(`${API_BASE}/tools/tad-insulation`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...params,
        window_bp: windowBp,
        min_boundary_strength: 0.15,
      }),
    });
  },

  emitViewportSync(params: {
    dataset_id: string;
    locus: GenomicInterval;
    resolution_bp: ResolutionBp;
    active_cohorts: CohortSpec[];
    highlighted_bins: HighlightBinSpec[];
    camera_focus_bin?: number | null;
  }): Promise<ViewportSyncResponse> {
    return fetchJson(`${API_BASE}/tools/viewport-sync`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });
  },

  getBookmark(bookmarkId: string): Promise<Partial<ViewportState>> {
    return fetchJson(`${API_BASE}/data/bookmarks/${bookmarkId}`);
  },

  planQuery(req: AgentPlanRequest): Promise<AgentPlanResponse> {
    return fetchJson(`${API_BASE}/agent/plan`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
  },

  executePlan(req: AgentExecuteRequest): Promise<CopilotResponse> {
    return fetchJson(`${API_BASE}/agent/execute`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
  },

  getProvenance(toolCallId: string): Promise<ProvenanceRecord> {
    return fetchJson(`${API_BASE}/agent/provenance/${toolCallId}`);
  },
};
