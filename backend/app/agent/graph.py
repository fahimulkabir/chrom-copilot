from __future__ import annotations

import time
from typing import Any, Callable, Literal, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from app.agent.binder import ProvenanceLedger, bind_provenance
from app.agent.planner import generate_plan
from app.agent.verifier import STEP_REF_PATTERN, TOOL_INPUT_MODELS, verify_plan
from app.engine.data_store import DataStore
from app.engine.tools_1d import run_stratify_by_1d_feature
from app.engine.tools_2d import (
    run_compute_ab_compartments,
    run_compute_tad_insulation,
    run_get_contact_map,
)
from app.engine.tools_3d import (
    run_find_3d_spatial_neighbors,
    run_measure_3d_pairwise_distance,
)
from app.engine.tools_stats import run_differential_test, run_emit_viewport_sync
from app.schemas.agent import (
    CopilotResponse,
    PlannedToolCall,
    PlanVerificationResult,
    ProvenanceRecord,
)
from app.schemas.viewport import ViewportState

TOOL_RUNNERS: dict[str, Callable[[DataStore, Any], BaseModel]] = {
    "stratify_by_1d_feature": run_stratify_by_1d_feature,
    "get_contact_map": run_get_contact_map,
    "compute_tad_insulation": run_compute_tad_insulation,
    "compute_ab_compartments": run_compute_ab_compartments,
    "measure_3d_pairwise_distance": run_measure_3d_pairwise_distance,
    "find_3d_spatial_neighbors": run_find_3d_spatial_neighbors,
    "run_differential_test": run_differential_test,
    "emit_viewport_sync": run_emit_viewport_sync,
}


def _resolve_value(val: Any, step_results: dict[str, dict[str, Any]], target_key: str | None = None) -> Any:
    if isinstance(val, str):
        m = STEP_REF_PATTERN.match(val)
        if m:
            step_id, field_path = m.group(1), m.group(2)
            if step_id not in step_results:
                raise ValueError(f"Step reference '${step_id}.{field_path}' not found in executed results.")
            cur: Any = step_results[step_id]
            for part in field_path.split("."):
                cur = cur[part]
            # Clean None values or dict values automatically for numeric list arguments
            if target_key in ("group_a_values", "group_b_values"):
                if isinstance(cur, dict):
                    return [float(v) for v in cur.values() if v is not None]
                if isinstance(cur, list):
                    return [float(v) for v in cur if v is not None]
            return cur
        return val
    if isinstance(val, list):
        return [_resolve_value(x, step_results) for x in val]
    if isinstance(val, dict):
        return {k: _resolve_value(v, step_results, target_key=k) for k, v in val.items()}
    return val


# ---------------------------------------------------------------------------
# Phase 1 LangGraph: Plan -> Verify Gate (No Execution Yet)
# ---------------------------------------------------------------------------

class PlanGraphState(TypedDict, total=False):
    user_query: str
    viewport: ViewportState
    use_llm: bool
    sparsity_threshold: float
    plan: list[PlannedToolCall]
    planner_clarification: str | None
    planner_mode: Literal["gemini_llm", "deterministic_fallback"]
    verification: PlanVerificationResult


def build_plan_graph(store: DataStore):
    def plan_node(state: PlanGraphState) -> PlanGraphState:
        plan, clarification, mode = generate_plan(
            store,
            state["user_query"],
            state["viewport"],
            use_llm=state.get("use_llm", True),
        )
        return {
            "plan": plan,
            "planner_clarification": clarification,
            "planner_mode": mode,
        }

    def verify_node(state: PlanGraphState) -> PlanGraphState:
        verification = verify_plan(
            store,
            state["plan"],
            state["viewport"],
            sparsity_threshold=state.get("sparsity_threshold", 0.95),
            planner_clarification=state.get("planner_clarification"),
        )
        return {"verification": verification}

    wf = StateGraph(PlanGraphState)
    wf.add_node("plan_node", plan_node)
    wf.add_node("verify_node", verify_node)
    wf.set_entry_point("plan_node")
    wf.add_edge("plan_node", "verify_node")
    wf.add_edge("verify_node", END)
    return wf.compile()


# ---------------------------------------------------------------------------
# Phase 2 LangGraph: Re-Verify -> Execute 8 Deterministic Tools -> Bind Provenance
# ---------------------------------------------------------------------------

class ExecuteGraphState(TypedDict, total=False):
    session_id: str
    user_query: str
    viewport: ViewportState
    approved_plan: list[PlannedToolCall]
    use_llm: bool
    allow_confirmed_warnings: bool
    verification: PlanVerificationResult
    tool_outputs: list[dict[str, Any]]
    response: CopilotResponse


def build_execute_graph(store: DataStore, ledger: ProvenanceLedger):
    def reverify_node(state: ExecuteGraphState) -> ExecuteGraphState:
        verification = verify_plan(
            store,
            state["approved_plan"],
            state["viewport"],
        )
        if verification.status == "blocked":
            raise ValueError(
                f"Execution blocked by Plan Verification Gate: {'; '.join(verification.warnings)}"
            )
        if verification.status == "needs_confirmation" and not state.get(
            "allow_confirmed_warnings", True
        ):
            raise ValueError(
                f"Plan requires explicit user confirmation: {verification.clarification_prompt}"
            )
        return {"verification": verification}

    def execute_tools_node(state: ExecuteGraphState) -> ExecuteGraphState:
        step_results: dict[str, dict[str, Any]] = {}
        tool_outputs: list[dict[str, Any]] = []

        for step in state["approved_plan"]:
            resolved_args = _resolve_value(step.arguments, step_results)
            input_cls = TOOL_INPUT_MODELS[step.tool_name]
            runner = TOOL_RUNNERS[step.tool_name]

            validated_req = input_cls.model_validate(resolved_args)
            t0 = time.perf_counter()
            result_model = runner(store, validated_req)
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            out_dict = result_model.model_dump(mode="json")
            step_results[step.step_id] = out_dict
            tool_outputs.append(out_dict)

            # Persist full provenance record for Part 5 ProvenanceDrawer
            ledger.record(
                ProvenanceRecord(
                    tool_call_id=out_dict["tool_call_id"],
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    session_id=state["session_id"],
                    latency_ms=latency_ms,
                    arguments=resolved_args,
                    output=out_dict,
                    warnings=out_dict.get("warnings", []),
                    contact_sparsity_fraction=out_dict.get("contact_sparsity_fraction"),
                )
            )

        return {"tool_outputs": tool_outputs}

    def bind_node(state: ExecuteGraphState) -> ExecuteGraphState:
        response = bind_provenance(
            session_id=state["session_id"],
            user_query=state["user_query"],
            tool_outputs=state["tool_outputs"],
            use_llm=state.get("use_llm", True),
        )
        return {"response": response}

    wf = StateGraph(ExecuteGraphState)
    wf.add_node("reverify_node", reverify_node)
    wf.add_node("execute_tools_node", execute_tools_node)
    wf.add_node("bind_node", bind_node)
    wf.set_entry_point("reverify_node")
    wf.add_edge("reverify_node", "execute_tools_node")
    wf.add_edge("execute_tools_node", "bind_node")
    wf.add_edge("bind_node", END)
    return wf.compile()