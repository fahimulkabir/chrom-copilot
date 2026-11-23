import { useEffect, useRef, useState } from "react";
import { api } from "../services/api";
import { useViewportStore } from "../store/useViewportStore";
import type {
  AgentPlanResponse,
  CopilotResponse,
  PlannedToolCall,
} from "../types/agent";
import type { ResolutionBp } from "../types/viewport";

interface CopilotChatPanelProps {
  onInspectProvenance: (toolCallId: string) => void;
}

type ChatEntry =
  | { id: string; kind: "user"; text: string }
  | {
      id: string;
      kind: "plan";
      userQuery: string;
      planResp: AgentPlanResponse;
      editedPlan: PlannedToolCall[];
      state: "pending" | "executed" | "cancelled";
    }
  | { id: string; kind: "response"; data: CopilotResponse }
  | { id: string; kind: "error"; message: string };

const QUICK_PROMPTS = [
  {
    label: "MYC TAD Insulation",
    query: "Compare MYC TAD insulation between GM12878 and K562 at 50kb",
  },
  {
    label: "HBB–LCR 3D Distance",
    query: "Measure the 3D distance between HBB and HBB_LCR",
  },
  {
    label: "chr18 A/B Switches",
    query: "Detect A/B compartment switches on chr18 between GM12878 and K562",
  },
  {
    label: "3D Neighbors (Clicked Bead)",
    query: "What genes are physically near this bead in 3D for K562?",
  },
];

function getPlanResolutionOptions(
  plan: PlannedToolCall[],
): { res: ResolutionBp; disabledReason: string | null }[] {
  const hasTad = plan.some((s) => s.tool_name === "compute_tad_insulation");
  const hasAb = plan.some((s) => s.tool_name === "compute_ab_compartments");

  return ([50_000, 100_000, 250_000] as ResolutionBp[]).map((res) => {
    if (hasTad && res > 100_000) {
      return {
        res,
        disabledReason: "TAD insulation requires resolution ≤ 100 kb",
      };
    }
    if (hasAb && res < 100_000) {
      return {
        res,
        disabledReason: "A/B compartments require resolution ≥ 100 kb",
      };
    }
    return { res, disabledReason: null };
  });
}

export function CopilotChatPanel({
  onInspectProvenance,
}: CopilotChatPanelProps) {
  const {
    viewport,
    applyViewportPatch,
    restoreBookmark,
    addSavedBookmark,
    clearSelectedBins,
  } = useViewportStore();

  const [sessionId] = useState(
    () => `sess_${Math.random().toString(36).slice(2, 8)}`,
  );
  const [input, setInput] = useState("");
  const [useLlm, setUseLlm] = useState(true);
  const [showRawState, setShowRawState] = useState(false);
  const [busy, setBusy] = useState(false);
  const [entries, setEntries] = useState<ChatEntry[]>([]);

  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [entries, busy]);

  const submitQuery = async (rawText: string) => {
    const query = rawText.trim();
    if (!query || busy) return;

    setInput("");
    setBusy(true);
    const uid = `u_${Date.now()}`;
    setEntries((prev) => [...prev, { id: uid, kind: "user", text: query }]);

    try {
      const planResp = await api.planQuery({
        session_id: sessionId,
        user_query: query,
        viewport,
        use_llm: useLlm,
      });

      setEntries((prev) => [
        ...prev,
        {
          id: `p_${Date.now()}`,
          kind: "plan",
          userQuery: query,
          planResp,
          editedPlan: planResp.verification.plan,
          state: "pending",
        },
      ]);
    } catch (err) {
      setEntries((prev) => [
        ...prev,
        {
          id: `e_${Date.now()}`,
          kind: "error",
          message: (err as Error).message,
        },
      ]);
    } finally {
      setBusy(false);
    }
  };

  const handlePlanResolutionChange = (
    entryId: string,
    newRes: ResolutionBp,
  ) => {
    setEntries((prev) =>
      prev.map((item) => {
        if (item.id !== entryId || item.kind !== "plan") return item;
        const allowed = getPlanResolutionOptions(item.editedPlan).find(
          (o) => o.res === newRes,
        );
        if (!allowed || allowed.disabledReason) return item;

        const updatedPlan = item.editedPlan.map((step) => {
          if (!("resolution_bp" in step.arguments)) return step;
          const oldRes = Number(step.arguments.resolution_bp) || newRes;
          const nextArgs: Record<string, unknown> = {
            ...step.arguments,
            resolution_bp: newRes,
          };

          // Keep TAD window_bp synchronized (>= 5 bins, min 250 kb)
          if (step.tool_name === "compute_tad_insulation") {
            nextArgs.window_bp = Math.max(250_000, newRes * 5);
          }

          // Rescale bin indices atomically if resolution changed
          if (oldRes !== newRes) {
            if (typeof nextArgs.anchor_bin_index === "number") {
              nextArgs.anchor_bin_index = Math.floor(
                (nextArgs.anchor_bin_index * oldRes) / newRes,
              );
            }
            if (typeof nextArgs.camera_focus_bin === "number") {
              nextArgs.camera_focus_bin = Math.floor(
                (nextArgs.camera_focus_bin * oldRes) / newRes,
              );
            }
            if (Array.isArray(nextArgs.highlighted_bins)) {
              nextArgs.highlighted_bins = nextArgs.highlighted_bins.map(
                (hb) => {
                  if (hb && typeof hb === "object" && "bin_index" in hb) {
                    const rec = hb as Record<string, unknown>;
                    return {
                      ...rec,
                      bin_index: Math.floor(
                        (Number(rec.bin_index) * oldRes) / newRes,
                      ),
                    };
                  }
                  return hb;
                },
              );
            }
          }

          const updatedRationale = step.rationale.replace(
            /\b(50|100|250)\s*kb\b/gi,
            `${newRes / 1000} kb`,
          );

          return {
            ...step,
            arguments: nextArgs,
            rationale: updatedRationale,
          };
        });

        return { ...item, editedPlan: updatedPlan };
      }),
    );
  };

  const handleApproveAndRun = async (entryId: string) => {
    const target = entries.find((e) => e.id === entryId);
    if (!target || target.kind !== "plan" || busy) return;

    setBusy(true);
    try {
      const res = await api.executePlan({
        session_id: sessionId,
        plan_id: target.planResp.plan_id,
        user_query: target.userQuery,
        viewport,
        approved_plan: target.editedPlan,
        use_llm: useLlm,
        allow_confirmed_warnings: true,
      });

      setEntries((prev) => [
        ...prev.map((item) =>
          item.id === entryId && item.kind === "plan"
            ? { ...item, state: "executed" as const }
            : item,
        ),
        { id: `r_${Date.now()}`, kind: "response", data: res },
      ]);

      // Atomic Query-to-View synchronization across 1D, 2D, and 3D panels
      if (res.viewport_patch) {
        const bk = res.viewport_patch.bookmark_id;
        if (bk) {
          addSavedBookmark(bk, target.userQuery.slice(0, 32));
        }
        await applyViewportPatch(res.viewport_patch);
      }
    } catch (err) {
      setEntries((prev) => [
        ...prev,
        {
          id: `e_${Date.now()}`,
          kind: "error",
          message: (err as Error).message,
        },
      ]);
    } finally {
      setBusy(false);
    }
  };

  const handleCancelPlan = (entryId: string) => {
    setEntries((prev) =>
      prev.map((item) =>
        item.id === entryId && item.kind === "plan"
          ? { ...item, state: "cancelled" as const }
          : item,
      ),
    );
  };

  const selectedBins = viewport.spatial_selection_3d.selected_bins;

  return (
    <div className="flex h-full flex-col rounded-lg border border-slate-700 bg-slate-900 p-3">
      {/* Header */}
      <div className="mb-2 flex items-center justify-between gap-2">
        <div>
          <h2 className="text-sm font-semibold text-slate-100">
            Grounded Multi-Scale Copilot
          </h2>
          <p className="text-xs text-slate-400">
            Plan Verification Gate · 8 Deterministic Tools · Claim-to-Evidence
            Provenance
          </p>
        </div>

        <div className="flex items-center gap-2 text-[11px]">
          <label className="flex cursor-pointer items-center gap-1 text-slate-300">
            <input
              type="checkbox"
              checked={useLlm}
              onChange={(e) => setUseLlm(e.target.checked)}
              className="accent-sky-500"
            />
            Gemini LLM
          </label>
          <button
            onClick={() => setShowRawState((v) => !v)}
            className="rounded border border-slate-700 bg-slate-800 px-2 py-0.5 text-slate-300 hover:bg-slate-700"
          >
            {showRawState ? "Hide State" : "State JSON"}
          </button>
        </div>
      </div>

      {/* Active Visual Context Strip (Point-to-Query Grounding) */}
      <div className="mb-2 flex flex-wrap items-center gap-1.5 rounded border border-slate-800 bg-slate-950 px-2.5 py-1.5 text-[11px]">
        <span className="font-semibold text-slate-400">Context:</span>
        <span className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-sky-300">
          {viewport.locus.chrom}:{(viewport.locus.start / 1e6).toFixed(2)}–
          {(viewport.locus.end / 1e6).toFixed(2)}Mb
        </span>
        <span className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-emerald-300">
          {viewport.resolution_bp / 1000}kb
        </span>
        <span className="rounded bg-slate-800 px-1.5 py-0.5 text-slate-200">
          {viewport.active_cohorts.map((c) => c.cohort_id).join(" + ")}
        </span>
        {selectedBins.length > 0 ? (
          <span className="flex items-center gap-1 rounded border border-cyan-500/40 bg-cyan-950/60 px-1.5 py-0.5 font-mono text-cyan-300">
            3D Bins: [{selectedBins.join(", ")}]
            <button
              onClick={clearSelectedBins}
              title="Clear selected bins"
              className="ml-0.5 text-cyan-200 hover:text-white"
            >
              ×
            </button>
          </span>
        ) : (
          <span className="text-slate-500">No 3D bead pinned</span>
        )}
      </div>

      {/* Optional Collapsible Raw ViewportState Inspector */}
      {showRawState && (
        <pre className="mb-2 max-h-32 overflow-auto rounded border border-slate-800 bg-slate-950 p-2 font-mono text-[10px] text-emerald-300">
          {JSON.stringify(viewport, null, 2)}
        </pre>
      )}

      {/* Message & Verification Stream */}
      <div
        ref={scrollRef}
        className="min-h-0 flex-1 space-y-2.5 overflow-y-auto rounded border border-slate-800 bg-slate-950 p-2.5 text-xs"
      >
        {entries.length === 0 && (
          <div className="space-y-2 text-slate-400">
            <p>
              Ask a multi-scale chromatin question or click a 3D bead above for{" "}
              <span className="text-cyan-300">Point-to-Query</span> exploration.
              Every plan passes the deterministic Verification Gate before
              execution.
            </p>
            <div className="flex flex-wrap gap-1.5">
              {QUICK_PROMPTS.map((qp) => (
                <button
                  key={qp.label}
                  onClick={() => void submitQuery(qp.query)}
                  className="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-[11px] text-sky-300 hover:border-sky-500 hover:bg-slate-800"
                >
                  {qp.label}
                </button>
              ))}
            </div>
          </div>
        )}

        {entries.map((entry) => {
          if (entry.kind === "user") {
            return (
              <div key={entry.id} className="flex justify-end">
                <div className="max-w-[85%] rounded-lg bg-sky-700/80 px-3 py-1.5 text-white">
                  {entry.text}
                </div>
              </div>
            );
          }

          if (entry.kind === "error") {
            return (
              <div
                key={entry.id}
                className="rounded border border-rose-500/40 bg-rose-950/50 p-2 text-rose-200"
              >
                {entry.message}
              </div>
            );
          }

          if (entry.kind === "plan") {
            const ver = entry.planResp.verification;
            const statusStyle =
              ver.status === "approved"
                ? "border-emerald-500/40 bg-emerald-950/30 text-emerald-300"
                : ver.status === "needs_confirmation"
                  ? "border-amber-500/50 bg-amber-950/30 text-amber-300"
                  : "border-rose-500/50 bg-rose-950/30 text-rose-300";

            const hasResolutionStep = entry.editedPlan.some(
              (s) => "resolution_bp" in s.arguments,
            );

            return (
              <div
                key={entry.id}
                className="rounded-lg border border-slate-700 bg-slate-900 p-2.5"
              >
                {/* Plan Gate Status Row */}
                <div className="mb-1.5 flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <span
                      className={`rounded border px-1.5 py-0.5 text-[10px] font-bold uppercase ${statusStyle}`}
                    >
                      Gate: {ver.status.replace("_", " ")}
                    </span>
                    <span className="font-mono text-[10px] text-slate-400">
                      {entry.planResp.planner_mode}
                    </span>
                  </div>
                  {entry.state !== "pending" && (
                    <span className="text-[10px] font-semibold uppercase text-slate-400">
                      {entry.state}
                    </span>
                  )}
                </div>

                {/* Passed Checks */}
                {ver.checks_passed.length > 0 && (
                  <div className="mb-1.5 flex flex-wrap gap-1">
                    {ver.checks_passed.map((chk) => (
                      <span
                        key={chk}
                        className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-[10px] text-emerald-300"
                      >
                        ✓ {chk}
                      </span>
                    ))}
                  </div>
                )}

                {/* Warnings & Clarification Prompt */}
                {ver.warnings.length > 0 && (
                  <ul className="mb-1.5 list-disc space-y-0.5 pl-4 text-[11px] text-amber-200">
                    {ver.warnings.map((w, i) => (
                      <li key={i}>{w}</li>
                    ))}
                  </ul>
                )}
                {ver.clarification_prompt && (
                  <p className="mb-2 rounded border border-amber-500/30 bg-amber-950/40 p-2 text-[11px] text-amber-100">
                    {ver.clarification_prompt}
                  </p>
                )}

                {/* Planned Deterministic Steps */}
                {entry.editedPlan.length > 0 && (
                  <ol className="mb-2 space-y-1 border-l-2 border-slate-700 pl-2.5">
                    {entry.editedPlan.map((step) => (
                      <li key={step.step_id} className="text-[11px]">
                        <span className="font-mono font-semibold text-sky-300">
                          {step.step_id}: {step.tool_name}
                        </span>
                        {"resolution_bp" in step.arguments && (
                          <span className="ml-1.5 rounded bg-slate-800 px-1 py-0.2 font-mono text-[10px] text-emerald-300">
                            {Number(step.arguments.resolution_bp) / 1000}kb
                          </span>
                        )}
                        <p className="text-slate-400">{step.rationale}</p>
                      </li>
                    ))}
                  </ol>
                )}

                {/* Interactive Steering & Approval Controls */}
                {entry.state === "pending" && (
                  <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
                    {hasResolutionStep && ver.status !== "blocked" ? (
                      <div className="flex items-center gap-1 text-[10px] text-slate-400">
                        <span>Adjust res:</span>
                        {getPlanResolutionOptions(entry.editedPlan).map(
                          ({ res, disabledReason }) => {
                            const activeRes = Number(
                              entry.editedPlan.find(
                                (s) => "resolution_bp" in s.arguments,
                              )?.arguments.resolution_bp,
                            );
                            const isCurrent = activeRes === res;
                            return (
                              <button
                                key={res}
                                disabled={Boolean(disabledReason)}
                                title={
                                  disabledReason ??
                                  `Switch all plan steps to ${res / 1000} kb`
                                }
                                onClick={() =>
                                  handlePlanResolutionChange(entry.id, res)
                                }
                                className={`rounded px-1.5 py-0.5 transition ${
                                  isCurrent
                                    ? "bg-emerald-600 font-semibold text-white"
                                    : disabledReason
                                      ? "cursor-not-allowed bg-slate-900 text-slate-600 line-through"
                                      : "bg-slate-800 text-slate-200 hover:bg-slate-700"
                                }`}
                              >
                                {res / 1000}kb
                              </button>
                            );
                          },
                        )}
                      </div>
                    ) : (
                      <span />
                    )}

                    <div className="flex items-center gap-1.5">
                      <button
                        onClick={() => handleCancelPlan(entry.id)}
                        disabled={busy}
                        className="rounded border border-slate-700 bg-slate-800 px-2.5 py-1 text-[11px] text-slate-300 hover:bg-slate-700"
                      >
                        Cancel
                      </button>
                      <button
                        onClick={() => void handleApproveAndRun(entry.id)}
                        disabled={
                          busy ||
                          ver.status === "blocked" ||
                          entry.editedPlan.length === 0
                        }
                        className="rounded bg-emerald-600 px-2.5 py-1 text-[11px] font-semibold text-white hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-40"
                      >
                        {busy ? "Running..." : "Approve & Run"}
                      </button>
                    </div>
                  </div>
                )}
              </div>
            );
          }

          // Verified Provenance Response
          return (
            <div
              key={entry.id}
              className="space-y-2 rounded-lg border border-sky-500/30 bg-slate-900/90 p-2.5"
            >
              {entry.data.claims.map((claim) => (
                <div
                  key={claim.claim_id}
                  className="rounded border border-slate-800 bg-slate-950/70 p-2"
                >
                  <div className="mb-1 flex items-start justify-between gap-2">
                    <p className="leading-relaxed text-slate-100">
                      {claim.text}
                    </p>
                    <span
                      className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] font-semibold ${
                        claim.is_verified
                          ? "bg-emerald-950 text-emerald-300 border border-emerald-500/40"
                          : "bg-amber-950 text-amber-300 border border-amber-500/40"
                      }`}
                    >
                      {claim.is_verified ? "Verified" : "Unverified"}
                    </span>
                  </div>

                  {/* Clickable Provenance Chips + One-Click Restore View */}
                  <div className="flex flex-wrap items-center gap-1.5 pt-1">
                    {claim.cited_tool_call_ids.map((tid) => (
                      <button
                        key={tid}
                        onClick={() => onInspectProvenance(tid)}
                        className="rounded border border-cyan-500/40 bg-cyan-950/50 px-1.5 py-0.5 font-mono text-[10px] text-cyan-200 hover:bg-cyan-900"
                        title="Inspect deterministic tool input, output, and warnings"
                      >
                        {tid}
                      </button>
                    ))}
                    {claim.bookmark_id && (
                      <button
                        onClick={() => void restoreBookmark(claim.bookmark_id!)}
                        className="rounded border border-indigo-500/40 bg-indigo-950/60 px-1.5 py-0.5 text-[10px] font-medium text-indigo-200 hover:bg-indigo-900"
                        title="Snap 1D, 2D, and 3D views back to this evidence state"
                      >
                        ↺ Restore View ({claim.bookmark_id})
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          );
        })}
      </div>

      {/* Prompt Input Bar */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          void submitQuery(input);
        }}
        className="mt-2 flex items-center gap-2"
      >
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about TAD insulation, A/B switches, 3D distance, or clicked 3D beads..."
          disabled={busy}
          className="min-w-0 flex-1 rounded border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-xs text-slate-100 placeholder-slate-500 focus:border-sky-500 focus:outline-none"
        />
        <button
          type="submit"
          disabled={busy || !input.trim()}
          className="rounded bg-sky-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-sky-500 disabled:opacity-40"
        >
          {busy ? "..." : "Plan"}
        </button>
      </form>
    </div>
  );
}
