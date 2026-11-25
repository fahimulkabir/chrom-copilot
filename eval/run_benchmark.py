# Evaluates three distinct systems on all 35 tasks:
# full_chromcopilot: Complete LangGraph planner + deterministic Verifier Gate + bidirectional ViewportState + 8 deterministic tools + Provenance Binder guardrails.
# ablation_no_gate_no_visual: Strips spatial_selection_3d (no Point-to-Query grounding) and bypasses verify_plan (executes blindly without safety/sparsity/sample-size checks).
# unconstrained_code_agent (runs automatically when --use-llm is passed with a valid GEMINI_API_KEY in .env): Prompts Gemini to write standalone Python code to inspect data/processed/ directly, executes it in a bounded 15-second subprocess, and grades Execution Success, Exact Numerical Accuracy, Unsupported-Claim Rate, and Tier 3 Trap Catch Rate with zero synthetic padding.

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.agent.binder import bind_provenance, get_provenance_ledger
from app.agent.graph import (
    TOOL_RUNNERS,
    _resolve_value,
    build_execute_graph,
    build_plan_graph,
)
from app.agent.planner import generate_plan
from app.agent.verifier import TOOL_INPUT_MODELS, verify_plan
from app.config import get_settings
from app.engine.data_store import DataStore, get_data_store
from app.schemas.agent import PlannedToolCall
from app.schemas.viewport import ViewportState


def _extract_answer_from_outputs(task: dict[str, Any], outputs: list[dict[str, Any]]) -> Any:
    cat = task["category"]
    for out in outputs:
        tn = out.get("tool_name")
        if cat == "contact_map" and tn == "get_contact_map":
            return out.get("total_contacts")
        if cat == "ab_compartments" and tn == "compute_ab_compartments":
            return len(out.get("switches", []))
        if cat == "tad_insulation_diff" and tn == "run_differential_test":
            return out.get("p_value")
        if cat == "pairwise_3d_distance" and tn == "measure_3d_pairwise_distance":
            return out.get("median_distance")
        if cat == "stratify_1d_multiscale" and tn == "stratify_by_1d_feature":
            return out.get("summary_stats", {}).get("high_mean")
        if cat == "point_to_query_3d" and tn == "find_3d_spatial_neighbors":
            return len(out.get("neighbors", []))
    return None


def _numeric_match(observed: Any, expected: Any, rel_tol: float = 0.05) -> bool:
    if observed is None or expected is None:
        return False
    try:
        obs_f = float(observed)
        exp_f = float(expected)
    except (TypeError, ValueError):
        return False
    if abs(exp_f) < 1e-9:
        return abs(obs_f - exp_f) < 1e-4
    return abs(obs_f - exp_f) / abs(exp_f) <= rel_tol


def _run_with_optional_sparse_sim(store: DataStore, simulate_sparse: bool, fn):
    if not simulate_sparse:
        return fn()
    orig = store.get_matrix
    sparse_mat = np.zeros((50, 50), dtype=np.float64)
    sparse_mat[0, 0] = 2.0
    store.get_matrix = lambda *args, **kwargs: (sparse_mat, pd.DataFrame())  # type: ignore[method-assign]
    try:
        return fn()
    finally:
        store.get_matrix = orig  # type: ignore[method-assign]


def eval_full_chromcopilot(
    store: DataStore, task: dict[str, Any], use_llm: bool
) -> dict[str, Any]:
    ledger = get_provenance_ledger()
    vp = ViewportState.model_validate(task["viewport"])
    t0 = time.perf_counter()

    def _plan_and_verify():
        if "injected_plan" in task:
            plan = [PlannedToolCall.model_validate(s) for s in task["injected_plan"]]
            ver = verify_plan(store, plan, vp)
            return ver
        plan_graph = build_plan_graph(store)
        res = plan_graph.invoke(
            {"user_query": task["query"], "viewport": vp, "use_llm": use_llm}
        )
        return res["verification"]

    ver = _run_with_optional_sparse_sim(
        store, bool(task.get("simulate_sparse_matrix")), _plan_and_verify
    )

    if task["tier"] == "tier3_traps":
        caught = ver.status == task["expected_gate_status"]
        return {
            "task_id": task["task_id"],
            "tier": task["tier"],
            "system": "full_chromcopilot",
            "execution_success": True,
            "answer_accurate": caught,
            "trap_caught": caught,
            "unsupported_claim_rate": 0.0,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "notes": f"gate={ver.status} (expected={task['expected_gate_status']})",
        }

    # Tier 1 / Tier 2 execution
    if ver.status == "blocked" or not ver.plan:
        return {
            "task_id": task["task_id"],
            "tier": task["tier"],
            "system": "full_chromcopilot",
            "execution_success": False,
            "answer_accurate": False,
            "trap_caught": None,
            "unsupported_claim_rate": 0.0,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "notes": f"unexpected gate={ver.status}",
        }

    exec_graph = build_execute_graph(store, ledger)
    try:
        out = exec_graph.invoke(
            {
                "session_id": f"bench_{task['task_id']}",
                "user_query": task["query"],
                "viewport": vp,
                "approved_plan": ver.plan,
                "use_llm": use_llm,
                "allow_confirmed_warnings": True,
            }
        )
        resp = out["response"]
        tool_outputs = out["tool_outputs"]
        obs = _extract_answer_from_outputs(task, tool_outputs)
        accurate = _numeric_match(obs, task["ground_truth"]["value"])
        n_claims = len(resp.claims)
        unverified = sum(1 for c in resp.claims if not c.is_verified)
        return {
            "task_id": task["task_id"],
            "tier": task["tier"],
            "system": "full_chromcopilot",
            "execution_success": True,
            "answer_accurate": accurate,
            "trap_caught": None,
            "unsupported_claim_rate": round(unverified / n_claims, 4) if n_claims else 0.0,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "notes": f"obs={obs}, gt={task['ground_truth']['value']}",
        }
    except Exception as exc:
        return {
            "task_id": task["task_id"],
            "tier": task["tier"],
            "system": "full_chromcopilot",
            "execution_success": False,
            "answer_accurate": False,
            "trap_caught": None,
            "unsupported_claim_rate": 0.0,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "notes": f"error={exc}",
        }


def eval_ablation_no_gate_no_visual(
    store: DataStore, task: dict[str, Any], use_llm: bool
) -> dict[str, Any]:
    """Ablation: strips 3D visual selection context and bypasses the Pre-Execution Verification Gate."""
    vp_dict = json.loads(json.dumps(task["viewport"]))
    vp_dict["spatial_selection_3d"] = {"selected_bins": [], "radius_nm": None}
    vp = ViewportState.model_validate(vp_dict)
    t0 = time.perf_counter()

    if "injected_plan" in task:
        plan = [PlannedToolCall.model_validate(s) for s in task["injected_plan"]]
    else:
        plan, _, _ = generate_plan(store, task["query"], vp, use_llm=use_llm)

    # Without Verifier Gate, attempt blind execution of whatever plan was produced
    if not plan:
        # Only catches traps when planner itself returned empty plan (e.g., missing gene/bead)
        caught = task["tier"] == "tier3_traps" and task["expected_gate_status"] == "needs_confirmation" and task["category"] in (
            "missing_visual_selection",
            "underspecified_gene",
            "underspecified_query",
        )
        return {
            "task_id": task["task_id"],
            "tier": task["tier"],
            "system": "ablation_no_gate_no_visual",
            "execution_success": task["tier"] == "tier3_traps",
            "answer_accurate": caught if task["tier"] == "tier3_traps" else False,
            "trap_caught": caught if task["tier"] == "tier3_traps" else None,
            "unsupported_claim_rate": 0.0,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "notes": "empty_plan_without_visual_context_or_gate",
        }

    def _blind_exec():
        step_results: dict[str, dict[str, Any]] = {}
        tool_outputs: list[dict[str, Any]] = []
        for step in plan:
            if step.tool_name not in TOOL_RUNNERS:
                raise RuntimeError(f"Uncaught disallowed tool crash: {step.tool_name}")
            resolved = _resolve_value(step.arguments, step_results)
            req = TOOL_INPUT_MODELS[step.tool_name].model_validate(resolved)
            res = TOOL_RUNNERS[step.tool_name](store, req).model_dump(mode="json")
            step_results[step.step_id] = res
            tool_outputs.append(res)
        return tool_outputs

    try:
        tool_outputs = _run_with_optional_sparse_sim(
            store, bool(task.get("simulate_sparse_matrix")), _blind_exec
        )
        if task["tier"] == "tier3_traps":
            # Blindly executed a trap without catching genome mismatch, sparsity, small n, or n=1!
            return {
                "task_id": task["task_id"],
                "tier": task["tier"],
                "system": "ablation_no_gate_no_visual",
                "execution_success": True,
                "answer_accurate": False,
                "trap_caught": False,
                "unsupported_claim_rate": 1.0,  # Silent invalid run without gate warning
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "notes": "SILENT_TRAP_FAILURE: executed without verification gate",
            }
        obs = _extract_answer_from_outputs(task, tool_outputs)
        accurate = _numeric_match(obs, task["ground_truth"]["value"])
        resp = bind_provenance(f"abl_{task['task_id']}", task["query"], tool_outputs, use_llm=use_llm)
        unverified = sum(1 for c in resp.claims if not c.is_verified)
        n_claims = len(resp.claims)
        return {
            "task_id": task["task_id"],
            "tier": task["tier"],
            "system": "ablation_no_gate_no_visual",
            "execution_success": True,
            "answer_accurate": accurate,
            "trap_caught": None,
            "unsupported_claim_rate": round(unverified / n_claims, 4) if n_claims else 0.0,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "notes": f"obs={obs}, gt={task['ground_truth']['value']}",
        }
    except Exception as exc:
        # Runtime crash on a blocked trap is NOT a graceful pre-execution catch
        return {
            "task_id": task["task_id"],
            "tier": task["tier"],
            "system": "ablation_no_gate_no_visual",
            "execution_success": False,
            "answer_accurate": False,
            "trap_caught": False if task["tier"] == "tier3_traps" else None,
            "unsupported_claim_rate": 0.0,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "notes": f"runtime_crash={type(exc).__name__}",
        }


def eval_unconstrained_code_agent(task: dict[str, Any]) -> dict[str, Any]:
    """Live unconstrained LLM code-generation baseline (requires GEMINI_API_KEY and --use-llm)."""
    from google import genai
    from google.genai import types

    t0 = time.perf_counter()
    client = genai.Client(api_key=get_settings().gemini_api_key)
    data_dir = str((REPO_ROOT / "data" / "processed").resolve())

    prompt = (
        f"You are a bioinformatics coder with access to processed Hi-C data in '{data_dir}'.\n"
        f"Files: metadata.json, contacts/GM12878.mcool, contacts/K562.mcool, "
        f"structures_3d/rao2014_gm12878_k562_3d.parquet, tracks_1d/genes_hg19.parquet.\n"
        f"User Query: {task['query']}\n"
        f"Viewport genome_build={task['viewport']['genome_build']}, "
        f"dataset_mode={task['viewport']['dataset_mode']}.\n"
        f"Write a standalone Python script that prints ONLY a single JSON object to stdout:\n"
        f'{{"refused": bool, "reason": str, "value": float_or_null, "summary": str}}\n'
        f"Set refused=true if the query is ambiguous, mismatched, out-of-bounds, or statistically invalid.\n"
        f"Return ONLY raw Python code inside ```python ... ``` fences."
    )
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0),
        )
        m = re.search(r"```(?:python)?\s*(.*?)```", resp.text or "", flags=re.DOTALL)
        code = m.group(1).strip() if m else (resp.text or "").strip()

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
            tf.write(code)
            script_path = Path(tf.name)

        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=15,
            )
        finally:
            script_path.unlink(missing_ok=True)

        if proc.returncode != 0:
            return {
                "task_id": task["task_id"],
                "tier": task["tier"],
                "system": "unconstrained_code_agent",
                "execution_success": False,
                "answer_accurate": False,
                "trap_caught": False if task["tier"] == "tier3_traps" else None,
                "unsupported_claim_rate": 1.0,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "notes": f"code_exit={proc.returncode}",
            }

        lines = [ln.strip() for ln in proc.stdout.strip().splitlines() if ln.strip()]
        payload = json.loads(lines[-1])
        refused = bool(payload.get("refused", False))
        val = payload.get("value")

        if task["tier"] == "tier3_traps":
            return {
                "task_id": task["task_id"],
                "tier": task["tier"],
                "system": "unconstrained_code_agent",
                "execution_success": True,
                "answer_accurate": refused,
                "trap_caught": refused,
                "unsupported_claim_rate": 0.0 if refused else 1.0,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "notes": f"refused={refused}",
            }

        accurate = (not refused) and _numeric_match(val, task["ground_truth"]["value"])
        # Unconstrained scripts have no programmatic tool_call_id provenance binding
        return {
            "task_id": task["task_id"],
            "tier": task["tier"],
            "system": "unconstrained_code_agent",
            "execution_success": True,
            "answer_accurate": accurate,
            "trap_caught": None,
            "unsupported_claim_rate": 0.0 if accurate else 1.0,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "notes": f"obs={val}, gt={task['ground_truth']['value']}",
        }
    except Exception as exc:
        return {
            "task_id": task["task_id"],
            "tier": task["tier"],
            "system": "unconstrained_code_agent",
            "execution_success": False,
            "answer_accurate": False,
            "trap_caught": False if task["tier"] == "tier3_traps" else None,
            "unsupported_claim_rate": 1.0,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "notes": f"error={type(exc).__name__}",
        }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for sys_name, grp in df.groupby("system", sort=False):
        t12 = grp[grp["tier"] != "tier3_traps"]
        t3 = grp[grp["tier"] == "tier3_traps"]
        summary_rows.append(
            {
                "system": sys_name,
                "n_tasks": int(len(grp)),
                "exec_success_pct": round(float(grp["execution_success"].mean() * 100.0), 1),
                "t1_t2_accuracy_pct": round(float(t12["answer_accurate"].mean() * 100.0), 1) if len(t12) else 0.0,
                "t3_trap_catch_pct": round(float(t3["trap_caught"].mean() * 100.0), 1) if len(t3) else 0.0,
                "overall_accuracy_pct": round(float(grp["answer_accurate"].mean() * 100.0), 1),
                "unsupported_claim_pct": round(float(grp["unsupported_claim_rate"].mean() * 100.0), 1),
                "mean_latency_ms": round(float(grp["latency_ms"].mean()), 1),
            }
        )
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ChromCopilot 35-task multi-scale benchmark.")
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Enable live Gemini calls for Full ChromCopilot, Ablation, and Unconstrained Code-Agent Baseline.",
    )
    args = parser.parse_args()

    bench_path = REPO_ROOT / "eval" / "benchmark_tasks.json"
    if not bench_path.exists():
        raise FileNotFoundError("Run `python eval/build_benchmark.py` first.")
    tasks = json.loads(bench_path.read_text())["tasks"]
    store = get_data_store()

    key = get_settings().gemini_api_key.strip()
    live_llm = bool(args.use_llm and key and key != "your_api_key_here")
    if args.use_llm and not live_llm:
        print("Warning: --use-llm requested, but GEMINI_API_KEY is not set in .env. Running offline mode.")

    rows: list[dict[str, Any]] = []
    for task in tasks:
        rows.append(eval_full_chromcopilot(store, task, use_llm=live_llm))
        rows.append(eval_ablation_no_gate_no_visual(store, task, use_llm=live_llm))
        if live_llm:
            rows.append(eval_unconstrained_code_agent(task))

    out_dir = REPO_ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    mode_tag = "live_llm" if live_llm else "deterministic"

    # Save per-task raw CSV
    raw_csv = out_dir / f"benchmark_raw_{mode_tag}.csv"
    with raw_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Save summary table JSON + CSV
    summary_rows = summarize(rows)
    sum_json = out_dir / f"benchmark_summary_{mode_tag}.json"
    sum_json.write_text(json.dumps(summary_rows, indent=2))
    pd.DataFrame(summary_rows).to_csv(out_dir / f"benchmark_summary_{mode_tag}.csv", index=False)

    print(f"\n=== Multi-Scale 3D+1D Benchmark Summary ({mode_tag}) ===")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    if not live_llm:
        print(
            "\nTip: Add a live GEMINI_API_KEY in .env and re-run with "
            "`python eval/run_benchmark.py --use-llm` to include the Unconstrained LLM Code-Agent baseline."
        )
    print(f"\nSaved raw logs to {raw_csv}")


if __name__ == "__main__":
    main()