from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException

from app.agent.binder import ProvenanceLedger, get_provenance_ledger
from app.agent.graph import build_execute_graph, build_plan_graph
from app.engine.data_store import DataStore, get_data_store
from app.schemas.agent import (
    AgentExecuteRequest,
    AgentPlanRequest,
    AgentPlanResponse,
    CopilotResponse,
    ProvenanceRecord,
)

router = APIRouter(prefix="/agent", tags=["Grounded LangGraph Copilot"])


@router.post("/plan", response_model=AgentPlanResponse)
def plan_query(
    req: AgentPlanRequest,
    store: DataStore = Depends(get_data_store),
) -> AgentPlanResponse:
    graph = build_plan_graph(store)
    result = graph.invoke(
        {
            "user_query": req.user_query,
            "viewport": req.viewport,
            "use_llm": req.use_llm,
            "sparsity_threshold": req.sparsity_threshold,
        }
    )
    return AgentPlanResponse(
        session_id=req.session_id,
        plan_id=f"pl_{uuid4().hex[:8]}",
        planner_mode=result["planner_mode"],
        verification=result["verification"],
    )


@router.post("/execute", response_model=CopilotResponse)
def execute_plan(
    req: AgentExecuteRequest,
    store: DataStore = Depends(get_data_store),
    ledger: ProvenanceLedger = Depends(get_provenance_ledger),
) -> CopilotResponse:
    graph = build_execute_graph(store, ledger)
    try:
        result = graph.invoke(
            {
                "session_id": req.session_id,
                "user_query": req.user_query,
                "viewport": req.viewport,
                "approved_plan": req.approved_plan,
                "use_llm": req.use_llm,
                "allow_confirmed_warnings": req.allow_confirmed_warnings,
            }
        )
        return result["response"]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/provenance/{tool_call_id}", response_model=ProvenanceRecord)
def get_provenance(
    tool_call_id: str,
    ledger: ProvenanceLedger = Depends(get_provenance_ledger),
) -> ProvenanceRecord:
    rec = ledger.get(tool_call_id)
    if rec is None:
        raise HTTPException(
            status_code=404, detail=f"Provenance record '{tool_call_id}' not found."
        )
    return rec