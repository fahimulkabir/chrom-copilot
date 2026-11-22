// Provides interactive controls for target loci (MYC, HBB, BCL2, GATA2), resolution (50 kb, 100 kb, 250 kb), cohort toggles (GM12878, K562), and renders a live Plotly chart comparing 1D gene expression (TPM) across the active locus alongside cached 3D Radius of Gyration (Rg)

import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";
import { useViewportStore } from "../store/useViewportStore";
import type { ResolutionBp } from "../types/viewport";

export function CohortStatsPanel() {
  const {
    viewport,
    metadata,
    genes,
    structures3D,
    selectLocusPreset,
    setResolution,
    toggleCohort,
  } = useViewportStore();

  const chartRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = chartRef.current;
    if (!el) return;

    const exprGenes = genes.filter(
      (g) =>
        g.feature_type === "gene" &&
        g.GM12878_tpm !== null &&
        g.K562_tpm !== null,
    );

    const symbols = exprGenes.map((g) => g.symbol);
    const traces: Record<string, unknown>[] = [];

    if (viewport.active_cohorts.some((c) => c.cohort_id === "GM12878")) {
      traces.push({
        type: "bar",
        name: "GM12878 (TPM)",
        x: symbols,
        y: exprGenes.map((g) => g.GM12878_tpm),
        marker: { color: "#2563EB" },
      });
    }
    if (viewport.active_cohorts.some((c) => c.cohort_id === "K562")) {
      traces.push({
        type: "bar",
        name: "K562 (TPM)",
        x: symbols,
        y: exprGenes.map((g) => g.K562_tpm),
        marker: { color: "#DC2626" },
      });
    }

    Plotly.react(
      el,
      traces,
      {
        barmode: "group",
        margin: { l: 46, r: 12, t: 18, b: 36 },
        paper_bgcolor: "transparent",
        plot_bgcolor: "#0f172a",
        font: { color: "#cbd5e1", size: 11 },
        legend: { orientation: "h", y: 1.18, x: 0 },
        yaxis: {
          title: { text: "Expression (TPM)", font: { size: 10 } },
          type: "log",
          gridcolor: "#1e293b",
        },
        xaxis: { gridcolor: "#1e293b" },
      },
      { displayModeBar: false, responsive: true },
    );
  }, [genes, viewport.active_cohorts]);

  const locusNames = metadata ? Object.keys(metadata.target_loci) : [];
  const resolutions: ResolutionBp[] = metadata?.available_resolutions_bp ?? [
    50_000, 100_000, 250_000,
  ];
  const allCohorts = metadata ? Object.values(metadata.cohorts) : [];

  return (
    <div className="flex h-full flex-col rounded-lg border border-slate-700 bg-slate-900 p-3">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-sm font-semibold text-slate-100">
          Cohorts, Locus Controls & 1D Expression
        </h2>
        <div className="flex items-center gap-1.5 text-[11px] text-slate-400">
          {viewport.active_cohorts.map((c) => {
            const rg = structures3D[c.cohort_id]?.[0]?.rg;
            return (
              <span
                key={c.cohort_id}
                className="rounded border border-slate-700 bg-slate-950 px-2 py-0.5"
              >
                {c.cohort_id} R<sub>g</sub>: {rg ? rg.toFixed(2) : "—"}
              </span>
            );
          })}
        </div>
      </div>

      {/* Controls Row */}
      <div className="mb-2 flex flex-wrap items-center gap-3 text-xs">
        <div className="flex items-center gap-1">
          <span className="text-slate-400">Locus:</span>
          {locusNames.map((name) => {
            const loc = metadata?.target_loci[name];
            const isActive =
              loc &&
              loc.chrom === viewport.locus.chrom &&
              loc.start === viewport.locus.start;
            return (
              <button
                key={name}
                onClick={() => selectLocusPreset(name)}
                className={`rounded px-2 py-1 font-medium transition ${
                  isActive
                    ? "bg-sky-600 text-white"
                    : "bg-slate-800 text-slate-300 hover:bg-slate-700"
                }`}
              >
                {name}
              </button>
            );
          })}
        </div>

        <div className="flex items-center gap-1">
          <span className="text-slate-400">Res:</span>
          {resolutions.map((r) => (
            <button
              key={r}
              onClick={() => setResolution(r)}
              className={`rounded px-2 py-1 font-medium transition ${
                viewport.resolution_bp === r
                  ? "bg-emerald-600 text-white"
                  : "bg-slate-800 text-slate-300 hover:bg-slate-700"
              }`}
            >
              {r / 1000}kb
            </button>
          ))}
        </div>

        <div className="flex items-center gap-1">
          <span className="text-slate-400">Cohorts:</span>
          {allCohorts.map((c) => {
            const active = viewport.active_cohorts.some(
              (ac) => ac.cohort_id === c.cohort_id,
            );
            return (
              <button
                key={c.cohort_id}
                onClick={() => toggleCohort(c.cohort_id)}
                className={`rounded border px-2 py-1 font-medium transition ${
                  active
                    ? "border-slate-400 bg-slate-800 text-white"
                    : "border-slate-800 bg-slate-950 text-slate-500"
                }`}
              >
                <span
                  className="mr-1.5 inline-block h-2 w-2 rounded-full"
                  style={{ backgroundColor: c.color_hex }}
                />
                {c.cohort_id}
              </button>
            );
          })}
        </div>
      </div>

      {/* Live 1D Expression Bar Chart */}
      <div className="min-h-0 flex-1 rounded border border-slate-800 bg-slate-950 p-1">
        <div ref={chartRef} className="h-full w-full" />
      </div>
    </div>
  );
}
