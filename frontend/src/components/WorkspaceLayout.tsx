import { useState } from "react";
import { useViewportStore } from "../store/useViewportStore";
import { Chromatin3DPanel } from "./Chromatin3DPanel";
import { CohortStatsPanel } from "./CohortStatsPanel";
import { CopilotChatPanel } from "./CopilotChatPanel";
import { Genome2D1DPanel } from "./Genome2D1DPanel";
import { ProvenanceDrawer } from "./ProvenanceDrawer";

export function WorkspaceLayout() {
  const { loading, error, viewport } = useViewportStore();
  const [activeProvenanceId, setActiveProvenanceId] = useState<string | null>(
    null,
  );

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
        <CopilotChatPanel onInspectProvenance={setActiveProvenanceId} />
      </main>

      <ProvenanceDrawer
        toolCallId={activeProvenanceId}
        onClose={() => setActiveProvenanceId(null)}
      />
    </div>
  );
}
