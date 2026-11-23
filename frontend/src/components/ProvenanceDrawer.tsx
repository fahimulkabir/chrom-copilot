// Clicking any provenance badge (tc_...) in the copilot response opens this drawer, fetches the exact persisted record from ProvenanceLedger, summarizes huge arrays (like sparse_pixels) cleanly so the UI stays snappy, and lets the researcher audit exact arguments, numbers, latency, and warnings.

import { useEffect, useMemo, useState } from "react";
import { api } from "../services/api";
import type { ProvenanceRecord } from "../types/agent";

interface ProvenanceDrawerProps {
  toolCallId: string | null;
  onClose: () => void;
}

function summarizeLargeArrays(
  obj: Record<string, unknown>,
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(obj)) {
    if (Array.isArray(v) && v.length > 25) {
      out[k] = `[Array(${v.length}) — showing first 10 items]`;
      out[`${k}_preview`] = v.slice(0, 10);
    } else {
      out[k] = v;
    }
  }
  return out;
}

export function ProvenanceDrawer({
  toolCallId,
  onClose,
}: ProvenanceDrawerProps) {
  const [record, setRecord] = useState<ProvenanceRecord | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!toolCallId) {
      setRecord(null);
      setError(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .getProvenance(toolCallId)
      .then((rec) => {
        if (!cancelled) {
          setRecord(rec);
          setLoading(false);
        }
      })
      .catch((err: Error) => {
        if (!cancelled) {
          setError(err.message);
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [toolCallId]);

  useEffect(() => {
    if (!toolCallId) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toolCallId, onClose]);

  const compactOutput = useMemo(
    () => (record ? summarizeLargeArrays(record.output) : null),
    [record],
  );

  if (!toolCallId) return null;

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-slate-950/70 backdrop-blur-xs">
      <div className="flex h-full w-full max-w-lg flex-col border-l border-slate-700 bg-slate-900 shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-800 px-4 py-3">
          <div>
            <h3 className="text-sm font-semibold text-white">
              Deterministic Provenance Inspector
            </h3>
            <p className="font-mono text-xs text-cyan-300">{toolCallId}</p>
          </div>
          <button
            onClick={onClose}
            className="rounded bg-slate-800 px-2.5 py-1 text-xs text-slate-300 hover:bg-slate-700 hover:text-white"
          >
            Close (Esc)
          </button>
        </div>

        {/* Body */}
        <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4 text-xs">
          {loading && (
            <p className="text-amber-300">Loading provenance record...</p>
          )}
          {error && (
            <div className="rounded border border-rose-500/40 bg-rose-950/40 p-3 text-rose-200">
              {error}
            </div>
          )}

          {record && (
            <>
              {/* Audit Metadata Strip */}
              <div className="grid grid-cols-2 gap-2 rounded border border-slate-800 bg-slate-950 p-3">
                <div>
                  <span className="text-slate-400">Tool:</span>{" "}
                  <span className="font-mono font-semibold text-sky-300">
                    {record.tool_name}
                  </span>
                </div>
                <div>
                  <span className="text-slate-400">Step ID:</span>{" "}
                  <span className="font-mono text-slate-200">
                    {record.step_id}
                  </span>
                </div>
                <div>
                  <span className="text-slate-400">Compute Latency:</span>{" "}
                  <span className="font-mono text-emerald-300">
                    {record.latency_ms.toFixed(1)} ms
                  </span>
                </div>
                <div>
                  <span className="text-slate-400">Contact Sparsity:</span>{" "}
                  <span className="font-mono text-slate-200">
                    {record.contact_sparsity_fraction !== null
                      ? `${(record.contact_sparsity_fraction * 100).toFixed(1)}%`
                      : "N/A"}
                  </span>
                </div>
              </div>

              {/* Warnings */}
              {record.warnings.length > 0 && (
                <div className="rounded border border-amber-500/40 bg-amber-950/40 p-3">
                  <p className="mb-1 font-semibold text-amber-300">
                    Engine Warnings ({record.warnings.length})
                  </p>
                  <ul className="list-disc space-y-1 pl-4 text-amber-100">
                    {record.warnings.map((w, i) => (
                      <li key={i}>{w}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Validated Tool Arguments */}
              <div>
                <p className="mb-1 font-semibold text-slate-200">
                  Validated Input Arguments
                </p>
                <pre className="max-h-56 overflow-auto rounded border border-slate-800 bg-slate-950 p-2.5 font-mono text-[11px] text-sky-200">
                  {JSON.stringify(record.arguments, null, 2)}
                </pre>
              </div>

              {/* Deterministic Tool Output */}
              <div>
                <p className="mb-1 font-semibold text-slate-200">
                  Deterministic Output Payload
                </p>
                <pre className="max-h-80 overflow-auto rounded border border-slate-800 bg-slate-950 p-2.5 font-mono text-[11px] text-emerald-300">
                  {JSON.stringify(compactOutput, null, 2)}
                </pre>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
