// The bottom-right slot holds our Part 3 Live ViewportState & Query-to-View Patch Tester (verifying Tool 8 camera flight + bookmark restore right now; in Part 5 this exact panel swaps in the full LangGraph Copilot Chat + Plan Verification Card).

import { useState } from "react";
import { api } from "../services/api";
import { useViewportStore } from "../store/useViewportStore";
import { Chromatin3DPanel } from "./Chromatin3DPanel";
import { CohortStatsPanel } from "./CohortStatsPanel";
import { Genome2D1DPanel } from "./Genome2D1DPanel";

function ViewportStateBridgePanel() {
  const {
    viewport,
    metadata,
    savedBookmarks,
    applyViewportPatch,
    restoreBookmark,
    addSavedBookmark,
  } = useViewportStore();
  const [testing, setTesting] = useState(false);

  // Tests Tool 8 (Query-to-View) atomically without any LLM yet
  const handleTestQueryToView = async () => {
    if (!metadata) return;
    setTesting(true);
    try {
      const hbbLoc = metadata.target_loci["HBB"];
      const hbbGene = metadata.gene_lookup["HBB"];
      const lcr = metadata.gene_lookup["HBB_LCR"];
      const res = await api.emitViewportSync({
        dataset_id: viewport.dataset_id,
        locus: { chrom: hbbLoc.chrom, start: hbbLoc.start, end: hbbLoc.end },
        resolution_bp: 50_000,
        active_cohorts: Object.values(metadata.cohorts),
        highlighted_bins: [
          {
            chrom: hbbGene.chrom,
            bin_index: hbbGene.bin_50kb,
            label: "HBB",
            role: "promoter",
          },
          {
            chrom: lcr.chrom,
            bin_index: lcr.bin_50kb,
            label: "HBB_LCR",
            role: "enhancer",
          },
        ],
        camera_focus_bin: hbbGene.bin_50kb,
      });
      addSavedBookmark(res.bookmark_id, "HBB / LCR Focus (50kb)");
      await applyViewportPatch(res.viewport_patch);
    } finally {
      setTesting(false);
    }
  };

  return (
    <div className="flex h-full flex-col rounded-lg border border-slate-700 bg-slate-900 p-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <div>
          <h2 className="text-sm font-semibold text-slate-100">
            Copilot State Contract (Part 3 Bridge)
          </h2>
          <p className="text-xs text-slate-400">
            Live serialized ViewportState JSON · Agent UI mounts here in Part 5
          </p>
        </div>

        <button
          onClick={handleTestQueryToView}
          disabled={testing || !metadata}
          className="rounded bg-indigo-600 px-2.5 py-1 text-xs font-semibold text-white hover:bg-indigo-500 disabled:opacity-50"
        >
          {testing ? "Syncing..." : "Test Tool 8 Patch → HBB"}
        </button>
      </div>

      {savedBookmarks.length > 0 && (
        <div className="mb-2 flex flex-wrap items-center gap-1.5">
          <span className="text-[11px] text-slate-400">Restore View:</span>
          {savedBookmarks.map((bk) => (
            <button
              key={bk.id}
              onClick={() => restoreBookmark(bk.id)}
              className="rounded border border-indigo-500/50 bg-indigo-950/60 px-2 py-0.5 text-[11px] text-indigo-200 hover:bg-indigo-900"
            >
              {bk.id} ({bk.label})
            </button>
          ))}
        </div>
      )}

      <pre className="min-h-0 flex-1 overflow-auto rounded border border-slate-800 bg-slate-950 p-2.5 font-mono text-[11px] text-emerald-300">
        {JSON.stringify(viewport, null, 2)}
      </pre>
    </div>
  );
}

export function WorkspaceLayout() {
  const { loading, error, viewport } = useViewportStore();

  return (
    <div className="flex h-screen flex-col bg-slate-950 text-slate-100">
      <header className="flex items-center justify-between border-b border-slate-800 bg-slate-900/90 px-4 py-2">
        <div className="flex items-center gap-3">
          <span className="text-base font-bold tracking-tight text-white">
            ChromCopilot
          </span>
          <span className="rounded bg-slate-800 px-2 py-0.5 text-xs text-sky-300">
            {viewport.dataset_id} · {viewport.genome_build}
          </span>
        </div>
        <div className="text-xs">
          {loading && (
            <span className="text-amber-300">Syncing 1D/2D/3D...</span>
          )}
          {error && <span className="text-rose-400">Error: {error}</span>}
          {!loading && !error && (
            <span className="text-emerald-400">1D / 2D / 3D Synchronized</span>
          )}
        </div>
      </header>

      <main className="grid min-h-0 flex-1 grid-cols-1 gap-3 p-3 lg:grid-cols-2 lg:grid-rows-2">
        <Genome2D1DPanel />
        <Chromatin3DPanel />
        <CohortStatsPanel />
        <ViewportStateBridgePanel />
      </main>
    </div>
  );
}
