// Stacks three linked visual layers on the exact same genomic axis:
// 1D Gene & Regulatory Track (highlighting curated key genes/enhancers and selected 3D bins)
// 1D TAD Insulation Track (live cooltools score overlay comparing GM12878 vs. K562)
// 2D Hi-C Contact Heatmap (log-scaled ICE-balanced matrix with hover crosshairs and click-to-select bin syncing directly into spatial_selection_3d.selected_bins)

import { useEffect, useMemo, useRef } from "react";
import Plotly from "plotly.js-dist-min";
import { useViewportStore } from "../store/useViewportStore";

export function Genome2D1DPanel() {
  const {
    viewport,
    genes,
    contactMaps,
    insulation,
    primary2DCohortId,
    setPrimary2DCohortId,
    hoveredBin,
    setHoveredBin,
    toggleSelectedBin,
  } = useViewportStore();

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const insulationRef = useRef<HTMLDivElement | null>(null);

  const activeMap = contactMaps[primary2DCohortId];
  const startBin = Math.floor(viewport.locus.start / viewport.resolution_bp);
  const nBins = activeMap?.n_bins ?? 0;

  // Build dense symmetric log1p matrix for crisp 2D canvas rendering
  const { matrix, maxVal } = useMemo(() => {
    if (!activeMap || nBins === 0) {
      return { matrix: new Float32Array(0), maxVal: 1 };
    }
    const mat = new Float32Array(nBins * nBins);
    let maxV = 0;
    for (const px of activeMap.sparse_pixels) {
      const i = px.bin_i - startBin;
      const j = px.bin_j - startBin;
      if (i < 0 || i >= nBins || j < 0 || j >= nBins) continue;
      const v = Math.log1p(px.value * 100);
      mat[i * nBins + j] = v;
      mat[j * nBins + i] = v;
      if (i !== j && v > maxV) maxV = v;
    }
    return { matrix: mat, maxVal: maxV || 1 };
  }, [activeMap, nBins, startBin]);

  // Render 2D Hi-C contact heatmap + highlights + hover/selection overlays
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || nBins === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const size = canvas.width;
    const cell = size / nBins;
    ctx.clearRect(0, 0, size, size);
    ctx.fillStyle = "#0b1120";
    ctx.fillRect(0, 0, size, size);

    for (let i = 0; i < nBins; i++) {
      for (let j = 0; j < nBins; j++) {
        const val = matrix[i * nBins + j];
        if (val <= 0) continue;
        const t = Math.min(1, val / maxVal);
        // White -> Amber -> Crimson scientific Hi-C palette
        const r = Math.round(255);
        const g = Math.round(245 * (1 - t * 0.88));
        const b = Math.round(230 * (1 - t * 0.96));
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(j * cell, i * cell, Math.ceil(cell), Math.ceil(cell));
      }
    }

    // Highlight annotated promoter/enhancer/boundary bins
    for (const hb of viewport.highlighted_bins) {
      const idx = hb.bin_index - startBin;
      if (idx < 0 || idx >= nBins) continue;
      ctx.strokeStyle = hb.role === "enhancer" ? "#facc15" : "#38bdf8";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(idx * cell, 0, cell, size);
      ctx.strokeRect(0, idx * cell, size, cell);
    }

    // Highlight bins selected in 3D or 2D (Point-to-Query)
    for (const selBin of viewport.spatial_selection_3d.selected_bins) {
      const idx = selBin - startBin;
      if (idx < 0 || idx >= nBins) continue;
      ctx.fillStyle = "rgba(34, 211, 238, 0.25)";
      ctx.fillRect(idx * cell, 0, cell, size);
      ctx.fillRect(0, idx * cell, size, cell);
      ctx.strokeStyle = "#22d3ee";
      ctx.lineWidth = 2;
      ctx.strokeRect(idx * cell, idx * cell, cell, cell);
    }

    // Hover crosshair synced from 3D viewer or 2D cursor
    if (hoveredBin !== null) {
      const idx = hoveredBin - startBin;
      if (idx >= 0 && idx < nBins) {
        ctx.strokeStyle = "rgba(255, 255, 255, 0.7)";
        ctx.lineWidth = 1;
        ctx.strokeRect(idx * cell, 0, cell, size);
        ctx.strokeRect(0, idx * cell, size, cell);
      }
    }
  }, [
    matrix,
    maxVal,
    nBins,
    startBin,
    viewport.highlighted_bins,
    viewport.spatial_selection_3d.selected_bins,
    hoveredBin,
  ]);

  // Render synchronized 1D TAD Insulation track via Plotly
  useEffect(() => {
    const el = insulationRef.current;
    if (!el) return;

    const traces: Record<string, unknown>[] = [];
    for (const cohort of viewport.active_cohorts) {
      const ins = insulation[cohort.cohort_id];
      if (!ins) continue;
      traces.push({
        type: "scatter",
        mode: "lines",
        name: cohort.cohort_id,
        x: ins.bin_positions.map((p) => p / 1e6),
        y: ins.insulation_scores,
        line: { color: cohort.color_hex, width: 2 },
      });
    }

    const shapes: Record<string, unknown>[] =
      viewport.spatial_selection_3d.selected_bins.map((b) => ({
        type: "line",
        x0: (b * viewport.resolution_bp) / 1e6,
        x1: (b * viewport.resolution_bp) / 1e6,
        y0: 0,
        y1: 1,
        yref: "paper",
        line: { color: "#22d3ee", width: 2, dash: "dot" },
      }));

    Plotly.react(
      el,
      traces,
      {
        margin: { l: 42, r: 10, t: 6, b: 24 },
        paper_bgcolor: "transparent",
        plot_bgcolor: "#0f172a",
        font: { color: "#cbd5e1", size: 10 },
        showlegend: true,
        legend: { orientation: "h", y: 1.25, x: 0 },
        xaxis: {
          title: { text: `${viewport.locus.chrom} (Mb)`, font: { size: 10 } },
          range: [viewport.locus.start / 1e6, viewport.locus.end / 1e6],
          gridcolor: "#1e293b",
        },
        yaxis: {
          title: { text: "TAD Ins.", font: { size: 10 } },
          gridcolor: "#1e293b",
        },
        shapes,
      },
      { displayModeBar: false, responsive: true },
    );
  }, [insulation, viewport]);

  const handleCanvasPointer = (
    e: React.MouseEvent<HTMLCanvasElement>,
    isClick: boolean,
  ) => {
    if (!canvasRef.current || nBins === 0) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const xRatio = (e.clientX - rect.left) / rect.width;
    const localIdx = Math.min(
      nBins - 1,
      Math.max(0, Math.floor(xRatio * nBins)),
    );
    const binIdx = startBin + localIdx;
    if (isClick) {
      toggleSelectedBin(binIdx);
    } else {
      setHoveredBin(binIdx);
    }
  };

  // Curated/expressed genes for clean 1D gene ribbon
  const featuredGenes = useMemo(
    () =>
      genes
        .filter((g) => g.GM12878_tpm !== null || g.feature_type === "enhancer")
        .slice(0, 12),
    [genes],
  );

  const locusSpan = Math.max(1, viewport.locus.end - viewport.locus.start);

  return (
    <div className="flex h-full flex-col rounded-lg border border-slate-700 bg-slate-900 p-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <div>
          <h2 className="text-sm font-semibold text-slate-100">
            1D Tracks + 2D Hi-C Contact Map
          </h2>
          <p className="text-xs text-slate-400">
            {viewport.locus.chrom}:{(viewport.locus.start / 1e6).toFixed(2)}–
            {(viewport.locus.end / 1e6).toFixed(2)} Mb ({viewport.genome_build})
            · {viewport.resolution_bp / 1000} kb
            {activeMap && (
              <span>
                {" "}
                · Sparsity:{" "}
                {(activeMap.contact_sparsity_fraction * 100).toFixed(1)}%
              </span>
            )}
          </p>
        </div>

        <div className="flex items-center gap-1">
          {viewport.active_cohorts.map((c) => (
            <button
              key={c.cohort_id}
              onClick={() => setPrimary2DCohortId(c.cohort_id)}
              className={`rounded px-2 py-1 text-xs font-medium transition ${
                primary2DCohortId === c.cohort_id
                  ? "bg-slate-100 text-slate-900"
                  : "bg-slate-800 text-slate-300 hover:bg-slate-700"
              }`}
            >
              {c.cohort_id}
            </button>
          ))}
        </div>
      </div>

      {/* 1D Gene & Enhancer Ribbon */}
      <div className="relative mb-2 h-8 w-full overflow-hidden rounded border border-slate-800 bg-slate-950">
        {featuredGenes.map((g) => {
          const leftPct = Math.max(
            0,
            Math.min(96, ((g.start - viewport.locus.start) / locusSpan) * 100),
          );
          const widthPct = Math.max(
            2,
            Math.min(100 - leftPct, ((g.end - g.start) / locusSpan) * 100),
          );
          const midBin = Math.floor(
            (g.start + g.end) / 2 / viewport.resolution_bp,
          );
          const isEnh = g.feature_type === "enhancer";
          return (
            <button
              key={g.symbol}
              onClick={() => toggleSelectedBin(midBin)}
              onMouseEnter={() => setHoveredBin(midBin)}
              onMouseLeave={() => setHoveredBin(null)}
              title={`${g.symbol} (${g.feature_type}) — click to select bin ${midBin}`}
              style={{ left: `${leftPct}%`, width: `${widthPct}%` }}
              className={`absolute top-1.5 h-5 truncate rounded px-1 text-[10px] font-semibold leading-5 ${
                isEnh
                  ? "bg-amber-400/90 text-slate-950"
                  : "bg-sky-500/85 text-white"
              }`}
            >
              {g.symbol}
            </button>
          );
        })}
      </div>

      {/* 1D TAD Insulation Track */}
      <div className="mb-2 h-24 w-full shrink-0">
        {viewport.resolution_bp <= 100_000 ? (
          <div ref={insulationRef} className="h-full w-full" />
        ) : (
          <div className="flex h-full items-center justify-center rounded border border-slate-800 bg-slate-950 text-xs text-slate-500">
            Switch to 50 kb or 100 kb to view TAD insulation track
          </div>
        )}
      </div>

      {/* 2D Hi-C Heatmap Canvas */}
      <div className="relative flex min-h-0 flex-1 items-center justify-center overflow-hidden rounded border border-slate-800 bg-slate-950 p-1">
        <canvas
          ref={canvasRef}
          width={360}
          height={360}
          onMouseMove={(e) => handleCanvasPointer(e, false)}
          onMouseLeave={() => setHoveredBin(null)}
          onClick={(e) => handleCanvasPointer(e, true)}
          className="aspect-square max-h-full max-w-full cursor-crosshair"
        />
        {hoveredBin !== null && (
          <div className="pointer-events-none absolute bottom-2 left-2 rounded bg-slate-900/90 px-2 py-1 text-[11px] text-cyan-300">
            Bin {hoveredBin} · {viewport.locus.chrom}:
            {((hoveredBin * viewport.resolution_bp) / 1e6).toFixed(2)} Mb (click
            to pin)
          </div>
        )}
      </div>
    </div>
  );
}
