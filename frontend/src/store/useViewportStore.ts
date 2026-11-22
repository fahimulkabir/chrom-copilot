// Single source of truth for the locked ViewportState contract (schema_version: "1.0.0"), atomic applyViewportPatch (Query-to-View), interactive 3D/2D bin selection (Point-to-Query), and bookmark restore.

import { create } from "zustand";
import { api } from "../services/api";
import type {
  Bead3D,
  ContactMapResponse,
  DatasetMetadata,
  GeneFeature,
  TadInsulationResponse,
} from "../services/api";
import type {
  CohortSpec,
  GenomicInterval,
  ResolutionBp,
  ViewportState,
} from "../types/viewport";

const INITIAL_VIEWPORT: ViewportState = {
  schema_version: "1.0.0",
  dataset_id: "rao2014_gm12878_k562",
  dataset_mode: "bulk_condition",
  genome_build: "hg19",
  locus: { chrom: "chr8", start: 127_500_000, end: 130_000_000 },
  resolution_bp: 50_000,
  active_cohorts: [
    {
      cohort_id: "GM12878",
      label: "GM12878 (B-Lymphoblastoid)",
      entity_ids: ["GM12878"],
      color_hex: "#2563EB",
    },
    {
      cohort_id: "K562",
      label: "K562 (Erythroleukemia)",
      entity_ids: ["K562"],
      color_hex: "#DC2626",
    },
  ],
  active_1d_tracks: ["gene_annotations", "rna_expression", "tad_insulation"],
  highlighted_bins: [],
  spatial_selection_3d: { selected_bins: [], radius_nm: null },
  camera_3d: { position: [12, 9, 16], target: [0, 0, 0] },
  bookmark_id: null,
};

interface ViewportStore {
  viewport: ViewportState;
  metadata: DatasetMetadata | null;
  primary2DCohortId: string;
  hoveredBin: number | null;
  cameraVersion: number;

  // Live fetched multi-scale data for current viewport
  loading: boolean;
  error: string | null;
  genes: GeneFeature[];
  contactMaps: Record<string, ContactMapResponse>;
  insulation: Record<string, TadInsulationResponse>;
  structures3D: Record<string, Bead3D[]>;
  savedBookmarks: { id: string; label: string }[];

  // Actions
  initialize: () => Promise<void>;
  refreshData: () => Promise<void>;
  selectLocusPreset: (locusName: string) => Promise<void>;
  setLocus: (locus: GenomicInterval) => Promise<void>;
  setResolution: (res: ResolutionBp) => Promise<void>;
  toggleCohort: (cohortId: string) => Promise<void>;
  setPrimary2DCohortId: (cohortId: string) => void;
  setHoveredBin: (binIndex: number | null) => void;
  toggleSelectedBin: (binIndex: number) => void;
  clearSelectedBins: () => void;
  setRadiusNm: (radius: number | null) => void;
  applyViewportPatch: (patch: Partial<ViewportState>) => Promise<void>;
  restoreBookmark: (bookmarkId: string) => Promise<void>;
  addSavedBookmark: (id: string, label: string) => void;
}

export const useViewportStore = create<ViewportStore>((set, get) => ({
  viewport: INITIAL_VIEWPORT,
  metadata: null,
  primary2DCohortId: "GM12878",
  hoveredBin: null,
  cameraVersion: 0,

  loading: false,
  error: null,
  genes: [],
  contactMaps: {},
  insulation: {},
  structures3D: {},
  savedBookmarks: [],

  async initialize() {
    try {
      set({ loading: true, error: null });
      const meta = await api.getMetadata();
      const allCohorts = Object.values(meta.cohorts);
      const myc = meta.target_loci["MYC"] ?? Object.values(meta.target_loci)[0];
      const mycGene = meta.gene_lookup["MYC"];

      set((s) => ({
        metadata: meta,
        primary2DCohortId: allCohorts[0]?.cohort_id ?? "GM12878",
        viewport: {
          ...s.viewport,
          dataset_id: meta.dataset_id,
          dataset_mode: meta.dataset_mode,
          genome_build: meta.genome_build,
          locus: { chrom: myc.chrom, start: myc.start, end: myc.end },
          active_cohorts: allCohorts,
          highlighted_bins: mycGene
            ? [
                {
                  chrom: mycGene.chrom,
                  bin_index: mycGene.bin_50kb,
                  label: "MYC",
                  role: "promoter",
                },
              ]
            : [],
        },
      }));
      await get().refreshData();
    } catch (err) {
      set({ loading: false, error: (err as Error).message });
    }
  },

  async refreshData() {
    const { viewport } = get();
    set({ loading: true, error: null });
    try {
      const genes = await api.getGenes(viewport.locus);

      const contactMaps: Record<string, ContactMapResponse> = {};
      const insulation: Record<string, TadInsulationResponse> = {};
      const structures3D: Record<string, Bead3D[]> = {};

      await Promise.all(
        viewport.active_cohorts.map(async (cohort: CohortSpec) => {
          const cid = cohort.cohort_id;
          const entityId = cohort.entity_ids[0];
          const [cm, ins, s3d] = await Promise.all([
            api.getContactMap({
              dataset_id: viewport.dataset_id,
              cohort,
              locus: viewport.locus,
              resolution_bp: viewport.resolution_bp,
            }),
            viewport.resolution_bp <= 100_000
              ? api.getTadInsulation({
                  dataset_id: viewport.dataset_id,
                  cohort,
                  locus: viewport.locus,
                  resolution_bp: viewport.resolution_bp,
                })
              : Promise.resolve(null),
            api.getStructure3D(
              entityId,
              viewport.locus,
              viewport.resolution_bp,
            ),
          ]);
          contactMaps[cid] = cm;
          if (ins) insulation[cid] = ins;
          structures3D[cid] = s3d;
        }),
      );

      set({ genes, contactMaps, insulation, structures3D, loading: false });
    } catch (err) {
      set({ loading: false, error: (err as Error).message });
    }
  },

  async selectLocusPreset(locusName: string) {
    const { metadata, viewport } = get();
    if (!metadata || !metadata.target_loci[locusName]) return;
    const loc = metadata.target_loci[locusName];

    const binKey =
      viewport.resolution_bp === 50_000
        ? "bin_50kb"
        : viewport.resolution_bp === 100_000
          ? "bin_100kb"
          : "bin_250kb";

    const highlights = [...loc.key_genes, ...loc.key_enhancers]
      .map((sym) => {
        const g = metadata.gene_lookup[sym];
        if (!g) return null;
        return {
          chrom: g.chrom,
          bin_index: g[binKey],
          label: sym,
          role: (g.feature_type === "enhancer" ? "enhancer" : "promoter") as
            | "enhancer"
            | "promoter",
        };
      })
      .filter((x): x is NonNullable<typeof x> => x !== null);

    set((s) => ({
      viewport: {
        ...s.viewport,
        locus: { chrom: loc.chrom, start: loc.start, end: loc.end },
        highlighted_bins: highlights,
        spatial_selection_3d: { selected_bins: [], radius_nm: null },
        bookmark_id: null,
      },
    }));
    await get().refreshData();
  },

  async setLocus(locus: GenomicInterval) {
    set((s) => ({
      viewport: { ...s.viewport, locus, bookmark_id: null },
    }));
    await get().refreshData();
  },

  async setResolution(res: ResolutionBp) {
    set((s) => ({
      viewport: {
        ...s.viewport,
        resolution_bp: res,
        spatial_selection_3d: { selected_bins: [], radius_nm: null },
        bookmark_id: null,
      },
    }));
    await get().refreshData();
  },

  async toggleCohort(cohortId: string) {
    const { metadata, viewport, primary2DCohortId } = get();
    if (!metadata || !metadata.cohorts[cohortId]) return;
    const exists = viewport.active_cohorts.some(
      (c) => c.cohort_id === cohortId,
    );

    // Always keep at least 1 active cohort (per ViewportState minItems=1 invariant)
    if (exists && viewport.active_cohorts.length === 1) return;

    const nextCohorts = exists
      ? viewport.active_cohorts.filter((c) => c.cohort_id !== cohortId)
      : [...viewport.active_cohorts, metadata.cohorts[cohortId]];

    const nextPrimary = nextCohorts.some(
      (c) => c.cohort_id === primary2DCohortId,
    )
      ? primary2DCohortId
      : nextCohorts[0].cohort_id;

    set((s) => ({
      primary2DCohortId: nextPrimary,
      viewport: { ...s.viewport, active_cohorts: nextCohorts },
    }));
    await get().refreshData();
  },

  setPrimary2DCohortId(cohortId: string) {
    set({ primary2DCohortId: cohortId });
  },

  setHoveredBin(binIndex: number | null) {
    set({ hoveredBin: binIndex });
  },

  toggleSelectedBin(binIndex: number) {
    set((s) => {
      const current = s.viewport.spatial_selection_3d.selected_bins;
      const next = current.includes(binIndex)
        ? current.filter((b) => b !== binIndex)
        : [...current, binIndex];
      return {
        viewport: {
          ...s.viewport,
          spatial_selection_3d: {
            ...s.viewport.spatial_selection_3d,
            selected_bins: next,
          },
        },
      };
    });
  },

  clearSelectedBins() {
    set((s) => ({
      viewport: {
        ...s.viewport,
        spatial_selection_3d: { selected_bins: [], radius_nm: null },
      },
    }));
  },

  setRadiusNm(radius: number | null) {
    set((s) => ({
      viewport: {
        ...s.viewport,
        spatial_selection_3d: {
          ...s.viewport.spatial_selection_3d,
          radius_nm: radius,
        },
      },
    }));
  },

  async applyViewportPatch(patch: Partial<ViewportState>) {
    const hasCamera = Boolean(patch.camera_3d);
    set((s) => ({
      cameraVersion: hasCamera ? s.cameraVersion + 1 : s.cameraVersion,
      primary2DCohortId:
        patch.active_cohorts && patch.active_cohorts.length > 0
          ? patch.active_cohorts[0].cohort_id
          : s.primary2DCohortId,
      viewport: {
        ...s.viewport,
        ...patch,
      },
    }));
    await get().refreshData();
  },

  async restoreBookmark(bookmarkId: string) {
    const patch = await api.getBookmark(bookmarkId);
    await get().applyViewportPatch(patch);
  },

  addSavedBookmark(id: string, label: string) {
    set((s) => ({
      savedBookmarks: [
        { id, label },
        ...s.savedBookmarks.filter((b) => b.id !== id),
      ].slice(0, 6),
    }));
  },
}));
