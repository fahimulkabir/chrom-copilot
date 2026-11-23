# Translates natural-language questions + live ViewportState JSON (including clicked 3D beads for Point-to-Query) into a typed list[PlannedToolCall] using only the 8 deterministic tools. Uses Gemini structured JSON output when GEMINI_API_KEY is configured, and falls back seamlessly to a multi-scale rule-based planner offline.

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.config import get_settings
from app.engine.data_store import DataStore
from app.schemas.agent import PlannedToolCall
from app.schemas.viewport import ViewportState


class _LLMPlanEnvelope(BaseModel):
    plan: list[PlannedToolCall] = Field(default_factory=list)
    clarification_prompt: str | None = None


def _has_live_gemini_key() -> bool:
    key = get_settings().gemini_api_key.strip()
    return bool(key) and key != "your_api_key_here"


def _build_planner_system_prompt(store: DataStore) -> str:
    meta = store.metadata
    compact_meta = {
        "dataset_id": meta["dataset_id"],
        "dataset_mode": meta["dataset_mode"],
        "genome_build": meta["genome_build"],
        "available_resolutions_bp": meta["available_resolutions_bp"],
        "chromosomes": meta["chromosomes"],
        "cohorts": {
            k: {
                "cohort_id": v["cohort_id"],
                "label": v["label"],
                "entity_ids": v["entity_ids"],
                "color_hex": v["color_hex"],
            }
            for k, v in meta["cohorts"].items()
        },
        "target_loci": meta["target_loci"],
        "gene_lookup": meta["gene_lookup"],
    }
    return (
        "You are ChromCopilot's Multi-Scale Bioinformatics Planner.\n"
        "Convert the user's question and live ViewportState JSON into a deterministic DAG of tool calls.\n"
        "RULES:\n"
        "1. ONLY use these 8 tools: stratify_by_1d_feature, get_contact_map, compute_tad_insulation, "
        "compute_ab_compartments, measure_3d_pairwise_distance, find_3d_spatial_neighbors, "
        "run_differential_test, emit_viewport_sync.\n"
        "2. NEVER compute math yourself. Always include emit_viewport_sync as the final step when moving "
        "to a locus or highlighting bins so 1D/2D/3D views synchronize.\n"
        "3. You may reference earlier step outputs using '$step_1.field_name' (e.g., '$step_1.insulation_scores').\n"
        "4. If the user asks a Point-to-Query question ('this bead', 'selected region') and "
        "viewport.spatial_selection_3d.selected_bins is empty, return plan=[] and a clear clarification_prompt.\n"
        "5. If the query is underspecified or asks for a nonexistent chromosome/gene, return plan=[] "
        "with a helpful clarification_prompt.\n\n"
        f"DATASET MANIFEST:\n{json.dumps(compact_meta, indent=2)}"
    )


def _plan_with_gemini(
    store: DataStore, user_query: str, viewport: ViewportState
) -> tuple[list[PlannedToolCall], str | None]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=get_settings().gemini_api_key)
    user_payload = (
        f"USER QUERY:\n{user_query}\n\n"
        f"LIVE VIEWPORT STATE:\n{viewport.model_dump_json(indent=2)}"
    )

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_payload,
        config=types.GenerateContentConfig(
            system_instruction=_build_planner_system_prompt(store),
            response_mime_type="application/json",
            response_schema=_LLMPlanEnvelope,
            temperature=0.0,
        ),
    )
    envelope = _LLMPlanEnvelope.model_validate_json(resp.text)
    return envelope.plan, envelope.clarification_prompt


# ---------------------------------------------------------------------------
# Deterministic Multi-Scale Fallback Planner (Offline / Unit Tests / Benchmarks)
# ---------------------------------------------------------------------------

def _detect_locus(store: DataStore, query: str, viewport: ViewportState) -> tuple[str | None, dict[str, Any]]:
    q_up = query.upper()
    meta = store.metadata
    for lname, linfo in meta["target_loci"].items():
        if re.search(rf"\b{re.escape(lname)}\b", q_up):
            return lname, {"chrom": linfo["chrom"], "start": linfo["start"], "end": linfo["end"]}
    for gsym, ginfo in meta["gene_lookup"].items():
        if re.search(rf"\b{re.escape(gsym)}\b", q_up):
            for lname, linfo in meta["target_loci"].items():
                if linfo["chrom"] == ginfo["chrom"]:
                    return lname, {"chrom": linfo["chrom"], "start": linfo["start"], "end": linfo["end"]}
    # Fallback to current viewport locus
    for lname, linfo in meta["target_loci"].items():
        if linfo["chrom"] == viewport.locus.chrom and linfo["start"] == viewport.locus.start:
            return lname, viewport.locus.model_dump()
    return None, viewport.locus.model_dump()


def _detect_chrom(query: str, fallback_chrom: str) -> str:
    m = re.search(r"\b(chr(?:[0-9]+|X|Y))\b", query, flags=re.IGNORECASE)
    if m:
        raw = m.group(1).lower()
        return "chr" + raw[3:].upper() if raw[3:] in ("x", "y") else raw
    return fallback_chrom


def _detect_resolution(query: str, fallback_res: int) -> int:
    m = re.search(r"\b(25|50|100|250|500|1000)\s*kb\b", query, flags=re.IGNORECASE)
    if m:
        return int(m.group(1)) * 1000
    return fallback_res


def _cohort_dict(store: DataStore, cid: str) -> dict[str, Any]:
    c = store.metadata["cohorts"][cid]
    return {
        "cohort_id": c["cohort_id"],
        "label": c["label"],
        "entity_ids": c["entity_ids"],
        "color_hex": c["color_hex"],
    }


def _plan_deterministic(
    store: DataStore, user_query: str, viewport: ViewportState
) -> tuple[list[PlannedToolCall], str | None]:
    q = user_query.lower()
    meta = store.metadata
    ds_id = store.dataset_id
    res = _detect_resolution(user_query, viewport.resolution_bp)
    locus_name, locus_dict = _detect_locus(store, user_query, viewport)
    chrom = _detect_chrom(user_query, locus_dict["chrom"])

    gm = _cohort_dict(store, "GM12878")
    k562 = _cohort_dict(store, "K562")
    both = {
        "cohort_id": "all_cohorts",
        "label": "GM12878 + K562",
        "entity_ids": ["GM12878", "K562"],
        "color_hex": "#6366F1",
    }

    # Explicit Point-to-Query (3D spatial neighbors around clicked bead)
    if any(kw in q for kw in ("neighbor", "nearby", "physically near", "3d neighborhood", "this bead", "selected bead")):
        sel_bins = viewport.spatial_selection_3d.selected_bins
        if not sel_bins:
            return (
                [],
                "No 3D chromatin bead is currently selected. Click a bead in the 3D viewer (or specify a gene) to query its 3D spatial neighborhood.",
            )
        anchor_bin = sel_bins[0]
        target_cohort = k562 if "k562" in q else (gm if "gm12878" in q else viewport.active_cohorts[0].model_dump())
        radius = viewport.spatial_selection_3d.radius_nm or 4.5
        return (
            [
                PlannedToolCall(
                    step_id="step_1",
                    tool_name="find_3d_spatial_neighbors",
                    arguments={
                        "dataset_id": ds_id,
                        "cohort": target_cohort,
                        "anchor_chrom": viewport.locus.chrom,
                        "anchor_bin_index": anchor_bin,
                        "resolution_bp": res,
                        "radius_threshold": radius,
                        "min_linear_separation_bins": 3,
                    },
                    rationale=f"Find non-adjacent 3D spatial neighbors and overlapping genes around clicked bin {anchor_bin}.",
                ),
                PlannedToolCall(
                    step_id="step_2",
                    tool_name="emit_viewport_sync",
                    arguments={
                        "dataset_id": ds_id,
                        "locus": viewport.locus.model_dump(),
                        "resolution_bp": res,
                        "active_cohorts": [target_cohort],
                        "highlighted_bins": [
                            {
                                "chrom": viewport.locus.chrom,
                                "bin_index": anchor_bin,
                                "label": f"Anchor Bin {anchor_bin}",
                                "role": "anchor",
                            }
                        ],
                        "camera_focus_bin": anchor_bin,
                    },
                    rationale="Highlight the clicked anchor bin and focus the 3D camera.",
                ),
            ],
            None,
        )

    # A/B Compartments & Switches
    if any(kw in q for kw in ("compartment", "a/b", "eigenvector", "switch")):
        ab_res = res if res >= 100_000 else 250_000
        return (
            [
                PlannedToolCall(
                    step_id="step_1",
                    tool_name="compute_ab_compartments",
                    arguments={
                        "dataset_id": ds_id,
                        "cohort": gm,
                        "compare_cohort": k562,
                        "chrom": chrom,
                        "resolution_bp": ab_res,
                    },
                    rationale=f"Compute gene-density-phased E1 A/B compartments and GM12878 vs K562 switches on {chrom}.",
                ),
                PlannedToolCall(
                    step_id="step_2",
                    tool_name="emit_viewport_sync",
                    arguments={
                        "dataset_id": ds_id,
                        "locus": locus_dict if locus_dict["chrom"] == chrom else {
                            "chrom": chrom,
                            "start": 0,
                            "end": min(2_500_000, int(meta["chromosomes"].get(chrom, 2_500_000))),
                        },
                        "resolution_bp": ab_res,
                        "active_cohorts": [gm, k562],
                        "highlighted_bins": [],
                        "camera_focus_bin": None,
                    },
                    rationale=f"Synchronize workspace to {chrom} at {ab_res // 1000} kb.",
                ),
            ],
            None,
        )

    # 3D Pairwise Distance (Enhancer-Promoter / Gene-Enhancer)
    if any(kw in q for kw in ("3d distance", "euclidean", "spatial distance", "closer in 3d", "distance between")):
        # Match two named features in gene_lookup
        found_syms = [
            sym
            for sym in meta["gene_lookup"].keys()
            if re.search(rf"\b{re.escape(sym)}\b", user_query, flags=re.IGNORECASE)
        ]
        # Smart default enhancer pairing if only the gene is named
        if len(found_syms) == 1 and locus_name and meta["target_loci"][locus_name]["key_enhancers"]:
            found_syms.append(meta["target_loci"][locus_name]["key_enhancers"][0])
        elif len(found_syms) < 2 and locus_name == "HBB":
            found_syms = ["HBB", "HBB_LCR"]
        elif len(found_syms) < 2 and locus_name == "GATA2":
            found_syms = ["GATA2", "GATA2_ENH"]
        elif len(found_syms) < 2 and locus_name == "MYC":
            found_syms = ["MYC", "MYC_E1"]

        if len(found_syms) < 2:
            return (
                [],
                "Specify two genomic features (e.g., HBB and HBB_LCR, or GATA2 and GATA2_ENH) to measure 3D pairwise distance.",
            )

        g_a = meta["gene_lookup"][found_syms[0]]
        g_b = meta["gene_lookup"][found_syms[1]]
        bin_key = "bin_50kb" if res == 50_000 else ("bin_100kb" if res == 100_000 else "bin_250kb")

        return (
            [
                PlannedToolCall(
                    step_id="step_1",
                    tool_name="measure_3d_pairwise_distance",
                    arguments={
                        "dataset_id": ds_id,
                        "cohort": both,
                        "locus_a": {"chrom": g_a["chrom"], "start": g_a["start"], "end": g_a["end"]},
                        "locus_b": {"chrom": g_b["chrom"], "start": g_b["start"], "end": g_b["end"]},
                        "resolution_bp": res,
                        "normalize_by_radius_of_gyration": True,
                    },
                    rationale=f"Measure Rg-normalized 3D Euclidean distance between {found_syms[0]} and {found_syms[1]} in GM12878 vs K562.",
                ),
                PlannedToolCall(
                    step_id="step_2",
                    tool_name="emit_viewport_sync",
                    arguments={
                        "dataset_id": ds_id,
                        "locus": locus_dict,
                        "resolution_bp": res,
                        "active_cohorts": [gm, k562],
                        "highlighted_bins": [
                            {
                                "chrom": g_a["chrom"],
                                "bin_index": g_a[bin_key],
                                "label": found_syms[0],
                                "role": "enhancer" if g_a["feature_type"] == "enhancer" else "promoter",
                            },
                            {
                                "chrom": g_b["chrom"],
                                "bin_index": g_b[bin_key],
                                "label": found_syms[1],
                                "role": "enhancer" if g_b["feature_type"] == "enhancer" else "promoter",
                            },
                        ],
                        "camera_focus_bin": g_a[bin_key],
                    },
                    rationale=f"Highlight {found_syms[0]} and {found_syms[1]} across 1D/2D/3D views and focus camera.",
                ),
            ],
            None,
        )

    # 1D Expression Stratification (+ optional TAD insulation or contact map)
    if any(kw in q for kw in ("stratify", "expression", "high vs low", "tpm")):
        gene_sym = next(
            (
                sym
                for sym, info in meta["gene_lookup"].items()
                if info["feature_type"] == "gene"
                and re.search(rf"\b{re.escape(sym)}\b", user_query, flags=re.IGNORECASE)
            ),
            None,
        )
        if not gene_sym:
            return (
                [],
                "Which gene should be used to stratify cohorts (e.g., HBB, BCL2, MYC, or GATA2)?",
            )
        steps = [
            PlannedToolCall(
                step_id="step_1",
                tool_name="stratify_by_1d_feature",
                arguments={
                    "dataset_id": ds_id,
                    "feature_name": gene_sym,
                    "source_cohort": both,
                    "split_method": "condition_group",
                },
                rationale=f"Stratify GM12878 and K562 by 1D {gene_sym} expression (TPM).",
            )
        ]
        if "insulation" in q or "tad" in q:
            tad_res = res if res <= 100_000 else 50_000
            steps.append(
                PlannedToolCall(
                    step_id="step_2",
                    tool_name="compute_tad_insulation",
                    arguments={
                        "dataset_id": ds_id,
                        "cohort": "$step_1.high_cohort",
                        "locus": locus_dict,
                        "resolution_bp": tad_res,
                        "window_bp": max(250_000, tad_res * 5),
                        "min_boundary_strength": 0.15,
                    },
                    rationale=f"Compute TAD insulation in the {gene_sym}-high cohort.",
                )
            )
        else:
            steps.append(
                PlannedToolCall(
                    step_id="step_2",
                    tool_name="get_contact_map",
                    arguments={
                        "dataset_id": ds_id,
                        "cohort": "$step_1.high_cohort",
                        "locus": locus_dict,
                        "resolution_bp": res,
                        "normalization": "ice_balanced",
                    },
                    rationale=f"Fetch 2D Hi-C contact map for the {gene_sym}-high cohort at {locus_dict['chrom']}.",
                )
            )
        g_info = meta["gene_lookup"][gene_sym]
        bin_key = "bin_50kb" if res == 50_000 else ("bin_100kb" if res == 100_000 else "bin_250kb")
        steps.append(
            PlannedToolCall(
                step_id="step_3",
                tool_name="emit_viewport_sync",
                arguments={
                    "dataset_id": ds_id,
                    "locus": locus_dict,
                    "resolution_bp": res,
                    "active_cohorts": [gm, k562],
                    "highlighted_bins": [
                        {
                            "chrom": g_info["chrom"],
                            "bin_index": g_info[bin_key],
                            "label": gene_sym,
                            "role": "promoter",
                        }
                    ],
                    "camera_focus_bin": g_info[bin_key],
                },
                rationale=f"Synchronize 1D/2D/3D workspace onto {gene_sym}.",
            )
        )
        return steps, None

    # TAD Insulation Comparison (+ deterministic Mann-Whitney U across locus bins)
    if any(kw in q for kw in ("tad", "insulation", "boundary", "boundaries")):
        tad_res = res if res <= 100_000 else 50_000
        win_bp = max(250_000, tad_res * 5)
        return (
            [
                PlannedToolCall(
                    step_id="step_1",
                    tool_name="compute_tad_insulation",
                    arguments={
                        "dataset_id": ds_id,
                        "cohort": gm,
                        "locus": locus_dict,
                        "resolution_bp": tad_res,
                        "window_bp": win_bp,
                        "min_boundary_strength": 0.15,
                    },
                    rationale=f"Compute GM12878 TAD insulation profile at {locus_dict['chrom']} ({tad_res // 1000} kb).",
                ),
                PlannedToolCall(
                    step_id="step_2",
                    tool_name="compute_tad_insulation",
                    arguments={
                        "dataset_id": ds_id,
                        "cohort": k562,
                        "locus": locus_dict,
                        "resolution_bp": tad_res,
                        "window_bp": win_bp,
                        "min_boundary_strength": 0.15,
                    },
                    rationale=f"Compute K562 TAD insulation profile at {locus_dict['chrom']} ({tad_res // 1000} kb).",
                ),
                PlannedToolCall(
                    step_id="step_3",
                    tool_name="run_differential_test",
                    arguments={
                        "dataset_id": ds_id,
                        "metric_name": f"tad_insulation_{locus_name or locus_dict['chrom']}",
                        "group_a_label": "GM12878",
                        "group_b_label": "K562",
                        "group_a_values": "$step_1.insulation_scores",
                        "group_b_values": "$step_2.insulation_scores",
                        "test_type": "mann_whitney_u",
                    },
                    rationale="Test whether bin-wise TAD insulation differs significantly between GM12878 and K562.",
                ),
                PlannedToolCall(
                    step_id="step_4",
                    tool_name="emit_viewport_sync",
                    arguments={
                        "dataset_id": ds_id,
                        "locus": locus_dict,
                        "resolution_bp": tad_res,
                        "active_cohorts": [gm, k562],
                        "highlighted_bins": [],
                        "camera_focus_bin": None,
                    },
                    rationale="Synchronize 1D insulation track, 2D heatmap, and 3D viewer to this locus.",
                ),
            ],
            None,
        )

    # 2D Contact Map Inspection
    if any(kw in q for kw in ("contact", "heatmap", "hi-c", "hic", "obs/exp", "matrix")):
        norm = "obs_over_exp" if ("obs" in q or "expected" in q) else "ice_balanced"
        target_cohort = k562 if "k562" in q else gm
        return (
            [
                PlannedToolCall(
                    step_id="step_1",
                    tool_name="get_contact_map",
                    arguments={
                        "dataset_id": ds_id,
                        "cohort": target_cohort,
                        "locus": locus_dict,
                        "resolution_bp": res,
                        "normalization": norm,
                    },
                    rationale=f"Fetch {norm} 2D Hi-C contact matrix for {target_cohort['cohort_id']}.",
                ),
                PlannedToolCall(
                    step_id="step_2",
                    tool_name="emit_viewport_sync",
                    arguments={
                        "dataset_id": ds_id,
                        "locus": locus_dict,
                        "resolution_bp": res,
                        "active_cohorts": [target_cohort],
                        "highlighted_bins": [],
                        "camera_focus_bin": None,
                    },
                    rationale="Sync 1D/2D/3D workspace to the queried contact map locus.",
                ),
            ],
            None,
        )

    # Ambiguous / unsupported prompt -> trigger HMI clarification card
    return (
        [],
        "Could not map your request to a specific multi-scale chromatin action. "
        "Try asking about TAD insulation, A/B compartment switches, 3D enhancer–promoter distance, "
        "1D gene expression stratification, or click a 3D bead to inspect its spatial neighbors.",
    )


def generate_plan(
    store: DataStore,
    user_query: str,
    viewport: ViewportState,
    *,
    use_llm: bool = True,
) -> tuple[list[PlannedToolCall], str | None, Literal["gemini_llm", "deterministic_fallback"]]:
    if use_llm and _has_live_gemini_key():
        try:
            plan, clarification = _plan_with_gemini(store, user_query, viewport)
            return plan, clarification, "gemini_llm"
        except Exception:
            # Graceful fallback if network/quota fails
            pass

    plan, clarification = _plan_deterministic(store, user_query, viewport)
    return plan, clarification, "deterministic_fallback"