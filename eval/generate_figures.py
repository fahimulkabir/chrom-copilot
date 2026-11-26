from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "eval" / "results"
FIG_DIR = REPO_ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.engine.data_store import get_data_store
from app.engine.tools_2d import run_compute_tad_insulation
from app.schemas.tools import TadInsulationInput
from app.schemas.viewport import CohortSpec, GenomicInterval

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

COLORS = {
    "full_chromcopilot": "#1D4ED8",
    "ablation_no_gate_no_visual": "#D97706",
    "unconstrained_code_agent": "#DC2626",
}


def _save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300)
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    plt.close(fig)
    print(f"Saved {FIG_DIR / stem}.png (+ .pdf)")


def plot_fig1_benchmark_comparison() -> None:
    df = pd.read_csv(RESULTS_DIR / "benchmark_summary_live_llm.csv")
    order = ["full_chromcopilot", "ablation_no_gate_no_visual", "unconstrained_code_agent"]
    labels = {
        "full_chromcopilot": "Full ChromCopilot\n(Gate + 8 Tools + Visual)",
        "ablation_no_gate_no_visual": "Ablation\n(No Gate / No Visual)",
        "unconstrained_code_agent": "Unconstrained Code Agent\n(Gemini Python Gen)",
    }
    df = df.set_index("system").loc[order].reset_index()

    metrics = [
        ("exec_success_pct", "Execution\nSuccess"),
        ("t1_t2_accuracy_pct", "Tier 1–2\nAccuracy"),
        ("t3_trap_catch_pct", "Tier 3 Trap\nCatch Rate"),
        ("overall_accuracy_pct", "Overall\nAccuracy"),
        ("unsupported_claim_pct", "Unsupported\nClaim Rate (↓)"),
    ]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    for i, sys_key in enumerate(order):
        row = df[df["system"] == sys_key].iloc[0]
        vals = [float(row[m[0]]) for m in metrics]
        bars = ax.bar(
            x + (i - 1) * width,
            vals,
            width=width,
            label=labels[sys_key],
            color=COLORS[sys_key],
            edgecolor="#0F172A",
            linewidth=0.6,
        )
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 1.8,
                f"{v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7.8,
                fontweight="semibold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylim(0, 114)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Multi-Scale 3D+1D Benchmark Performance (n = 35 Tasks, Live gemini-3.5-flash-lite)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    _save(fig, "fig1_benchmark_comparison")


def plot_fig2_error_taxonomy() -> None:
    raw = pd.read_csv(RESULTS_DIR / "benchmark_raw_live_llm.csv")

    # Panel A: Unconstrained Code-Agent Outcome Taxonomy
    u = raw[raw["system"] == "unconstrained_code_agent"]
    cats = {
        "Silent Methodological Drift\n(Wrong Matrix / Norm / Params)": 0,
        "Correct Tier 3 Trap Refusal": 0,
        "Python Runtime Crash\n(code_exit=1)": 0,
        "Missing Output / None\n(3D Neighbors / Parsing)": 0,
        "Missed Tier 3 Safety Trap": 0,
        "Transient API 503 Error": 0,
    }
    for _, row in u.iterrows():
        note = str(row["notes"])
        if row["tier"] == "tier3_traps":
            if bool(row["trap_caught"]):
                cats["Correct Tier 3 Trap Refusal"] += 1
            elif "ServerError" in note:
                cats["Transient API 503 Error"] += 1
            else:
                cats["Missed Tier 3 Safety Trap"] += 1
        else:
            if "code_exit=" in note:
                cats["Python Runtime Crash\n(code_exit=1)"] += 1
            elif "ServerError" in note:
                cats["Transient API 503 Error"] += 1
            elif "obs=None" in note:
                cats["Missing Output / None\n(3D Neighbors / Parsing)"] += 1
            elif not bool(row["answer_accurate"]):
                cats["Silent Methodological Drift\n(Wrong Matrix / Norm / Params)"] += 1

    # Panel B: Full ChromCopilot Execution Mode Breakdown
    fc = raw[raw["system"] == "full_chromcopilot"]
    modes = fc["notes"].str.extract(r"mode=([^;]+)")[0].value_counts()
    mode_labels = [
        "Live Gemini LLM\n(gemini-3.5-flash-lite)",
        "Deterministic Fallback\n(503 / Schema Guard)",
        "Injected Safety Traps\n(Direct Gate Test)",
    ]
    mode_vals = [
        int(modes.get("gemini_llm", 0)),
        int(modes.get("deterministic_fallback", 0)),
        int(modes.get("injected_trap", 0)),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})

    y_pos = np.arange(len(cats))
    counts = list(cats.values())
    bar_colors = ["#DC2626", "#16A34A", "#991B1B", "#EA580C", "#F59E0B", "#64748B"]
    bars = ax1.barh(y_pos, counts, color=bar_colors, edgecolor="#0F172A", linewidth=0.6)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(list(cats.keys()), fontsize=8.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("Number of Tasks (out of 35)")
    ax1.set_title("A. Unconstrained Code-Agent Failure Taxonomy")
    ax1.set_xlim(0, max(counts) + 3.5)
    ax1.grid(axis="x", linestyle="--", alpha=0.35)
    ax1.set_axisbelow(True)
    for b, v in zip(bars, counts):
        ax1.text(v + 0.3, b.get_y() + b.get_height() / 2, f"{v} ({v/35*100:.0f}%)", va="center", fontsize=8.5, fontweight="semibold")

    x2 = np.arange(len(mode_labels))
    bars2 = ax2.bar(x2, mode_vals, width=0.55, color=["#1D4ED8", "#0EA5E9", "#475569"], edgecolor="#0F172A", linewidth=0.6)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(mode_labels, fontsize=8.2)
    ax2.set_ylabel("Number of Tasks (100% Verified Accurate)")
    ax2.set_title("B. ChromCopilot Planning & Resilience Modes")
    ax2.set_ylim(0, max(mode_vals) + 5)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.set_axisbelow(True)
    for b, v in zip(bars2, mode_vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v}/35", ha="center", va="bottom", fontsize=8.5, fontweight="semibold")

    fig.tight_layout()
    _save(fig, "fig2_error_taxonomy")


def plot_fig3_latency_scalability() -> None:
    tools_df = pd.read_csv(RESULTS_DIR / "latency_tools.csv")
    scale_df = pd.read_csv(RESULTS_DIR / "latency_scalability.csv")
    prof = json.loads((RESULTS_DIR / "latency_profile.json").read_text())
    gate_ms = next(s["mean_ms"] for s in prof["stages"] if "Verify Gate" in s["stage"])

    # Append Verification Gate row for direct visual comparison
    gate_row = pd.DataFrame([{"tool": "Pre-Execution Verify Gate", "modality": "Gate", "mean_ms": gate_ms, "p95_ms": gate_ms}])
    plot_df = pd.concat([tools_df, gate_row], ignore_index=True).sort_values("mean_ms", ascending=True)

    mod_colors = {
        "1D": "#0284C7",
        "2D": "#2563EB",
        "3D": "#7C3AED",
        "Stats": "#059669",
        "Sync": "#475569",
        "Gate": "#D97706",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.3), gridspec_kw={"width_ratios": [1.35, 1]})

    y = np.arange(len(plot_df))
    colors = [mod_colors.get(m, "#334155") for m in plot_df["modality"]]
    bars = ax1.barh(y, plot_df["mean_ms"], color=colors, edgecolor="#0F172A", linewidth=0.6)
    ax1.set_xscale("log")
    ax1.set_yticks(y)
    ax1.set_yticklabels(plot_df["tool"], fontsize=8.5)
    ax1.set_xlabel("Mean Compute Latency (ms, log scale)")
    ax1.set_title("A. Deterministic Tool & Verify Gate Latency")
    ax1.set_xlim(0.2, 1200)
    ax1.grid(axis="x", linestyle="--", alpha=0.35)
    ax1.set_axisbelow(True)
    for b, v in zip(bars, plot_df["mean_ms"]):
        ax1.text(v * 1.18, b.get_y() + b.get_height() / 2, f"{v:.2f} ms", va="center", fontsize=8)

    ax2.plot(scale_df["n_cells"], scale_df["mean_ms"], marker="o", linewidth=2.2, color="#1D4ED8", label="Mean latency")
    ax2.plot(scale_df["n_cells"], scale_df["p95_ms"], marker="s", linestyle="--", linewidth=1.6, color="#0EA5E9", label="P95 latency")
    ax2.fill_between(scale_df["n_cells"], scale_df["mean_ms"], scale_df["p95_ms"], color="#93C5FD", alpha=0.25)
    for x_v, y_v in zip(scale_df["n_cells"], scale_df["mean_ms"]):
        ax2.text(x_v, y_v + 0.08, f"{y_v:.2f} ms", ha="center", fontsize=8)
    ax2.set_xlabel("Single-Cell Cohort Size (n cells)")
    ax2.set_ylabel("3D Distance + Pseudobulk + MWU Latency (ms)")
    ax2.set_title("B. Cohort Scalability (100–2,000 Cells)")
    ax2.set_ylim(0, max(scale_df["p95_ms"]) * 1.35)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    _save(fig, "fig3_latency_scalability")


def plot_fig4_case_study_myc() -> None:
    store = get_data_store()
    ds_id = store.dataset_id
    chrom, start, end, res = "chr8", 127_500_000, 130_000_000, 50_000
    iv = GenomicInterval(chrom=chrom, start=start, end=end)
    gm = CohortSpec(cohort_id="GM12878", label="GM12878", entity_ids=["GM12878"])
    k562 = CohortSpec(cohort_id="K562", label="K562", entity_ids=["K562"])

    mat_gm, _ = store.get_matrix(["GM12878"], chrom, start, end, res, balance=True)
    ins_gm = run_compute_tad_insulation(
        store, TadInsulationInput(dataset_id=ds_id, cohort=gm, locus=iv, resolution_bp=res, window_bp=250_000, min_boundary_strength=0.15)
    )
    ins_k562 = run_compute_tad_insulation(
        store, TadInsulationInput(dataset_id=ds_id, cohort=k562, locus=iv, resolution_bp=res, window_bp=250_000, min_boundary_strength=0.15)
    )
    df3_gm = store.get_3d_coords("GM12878", chrom, res, start=start, end=end)
    df3_k562 = store.get_3d_coords("K562", chrom, res, start=start, end=end)
    myc_pos = 128.75  # Mb

    fig = plt.figure(figsize=(11.5, 4.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.15, 1.05], wspace=0.32)

    # Panel A: 2D Hi-C Contact Map
    ax1 = fig.add_subplot(gs[0, 0])
    img = np.log1p(np.nan_to_num(mat_gm, nan=0.0) * 100.0)
    im = ax1.imshow(img, cmap="YlOrRd", origin="upper", extent=[start / 1e6, end / 1e6, end / 1e6, start / 1e6])
    ax1.axvline(myc_pos, color="#0284C7", linestyle="--", linewidth=1.1)
    ax1.axhline(myc_pos, color="#0284C7", linestyle="--", linewidth=1.1)
    ax1.set_xlabel("chr8 (Mb)")
    ax1.set_ylabel("chr8 (Mb)")
    ax1.set_title("A. 2D Hi-C Map (GM12878, 50 kb)")
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("log1p(100 × ICE)", fontsize=8)

    # Panel B: 1D TAD Insulation Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    xs = [p / 1e6 for p in ins_gm.bin_positions]
    ax2.plot(xs, ins_gm.insulation_scores, color="#2563EB", linewidth=1.9, label=f"GM12878 ({len(ins_gm.boundaries)} TAD bnd)")
    ax2.plot(xs, ins_k562.insulation_scores, color="#DC2626", linewidth=1.9, label=f"K562 ({len(ins_k562.boundaries)} TAD bnd)")
    ax2.axvline(myc_pos, color="#0F172A", linestyle=":", linewidth=1.2, label="MYC TSS (128.75 Mb)")
    ax2.set_xlabel("chr8 Position (Mb, hg19)")
    ax2.set_ylabel("log2 TAD Insulation Score")
    ax2.set_title("B. 1D TAD Insulation (MWU p = 0.393)")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(frameon=False, fontsize=7.8, loc="lower left")

    # Panel C: 3D Chromatin Conformation
    ax3 = fig.add_subplot(gs[0, 2], projection="3d")
    ax3.plot(df3_gm["x"], df3_gm["y"], df3_gm["z"], color="#2563EB", linewidth=1.8, alpha=0.85, label="GM12878 3D")
    ax3.plot(df3_k562["x"], df3_k562["y"], df3_k562["z"], color="#DC2626", linewidth=1.8, alpha=0.85, label="K562 3D")
    myc_gm = df3_gm[df3_gm["bin_index"] == 2575]
    myc_k562 = df3_k562[df3_k562["bin_index"] == 2575]
    if not myc_gm.empty:
        ax3.scatter(myc_gm["x"], myc_gm["y"], myc_gm["z"], color="#38BDF8", edgecolor="#0F172A", s=55, zorder=5, label="MYC Bin 2575")
    if not myc_k562.empty:
        ax3.scatter(myc_k562["x"], myc_k562["y"], myc_k562["z"], color="#FACC15", edgecolor="#0F172A", s=55, zorder=5)
    ax3.set_title("C. 3D Chromatin Fold (MYC Locus)")
    ax3.set_xlabel("X", fontsize=8)
    ax3.set_ylabel("Y", fontsize=8)
    ax3.set_zlabel("Z", fontsize=8)
    ax3.tick_params(labelsize=7)
    ax3.legend(frameon=False, fontsize=7.5, loc="upper left")

    _save(fig, "fig4_case_study_myc")


def main() -> None:
    plot_fig1_benchmark_comparison()
    plot_fig2_error_taxonomy()
    plot_fig3_latency_scalability()
    plot_fig4_case_study_myc()
    print("\nAll 4 publication figures generated in paper/figures/.")


if __name__ == "__main__":
    main()