from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

from app.schemas.viewport import ViewportState


class PlannedToolCall(BaseModel):
    step_id: str
    tool_name: str
    arguments: dict[str, Any]
    rationale: str


class PlanVerificationResult(BaseModel):
    status: Literal["approved", "needs_confirmation", "blocked"]
    plan: list[PlannedToolCall]
    checks_passed: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    clarification_prompt: Optional[str] = Field(
        default=None,
        description="Displayed in the UI Plan Confirmation Card when user input or confirmation is required.",
    )


class ProvenanceClaim(BaseModel):
    claim_id: str
    text: str
    cited_tool_call_ids: list[str] = Field(default_factory=list)
    is_verified: bool = Field(
        ...,
        description="True only if backed by at least one deterministic tool_call_id; False marks [Hypothesis / Unverified].",
    )
    bookmark_id: Optional[str] = None


class CopilotResponse(BaseModel):
    session_id: str
    claims: list[ProvenanceClaim]
    viewport_patch: Optional[dict[str, Any]] = None
    executed_tool_call_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Part 4 Agent API & Provenance Ledger Contracts
# ---------------------------------------------------------------------------

class AgentPlanRequest(BaseModel):
    session_id: str = "default"
    user_query: str = Field(..., min_length=1)
    viewport: ViewportState
    use_llm: bool = Field(
        default=True,
        description="Uses Gemini when GEMINI_API_KEY is configured; falls back to deterministic planner otherwise.",
    )
    sparsity_threshold: float = Field(default=0.95, ge=0.0, le=1.0)


class AgentPlanResponse(BaseModel):
    session_id: str
    plan_id: str
    planner_mode: Literal["gemini_llm", "deterministic_fallback"]
    verification: PlanVerificationResult


class AgentExecuteRequest(BaseModel):
    session_id: str = "default"
    plan_id: Optional[str] = None
    user_query: str = Field(..., min_length=1)
    viewport: ViewportState
    approved_plan: list[PlannedToolCall] = Field(..., min_length=1)
    use_llm: bool = True
    allow_confirmed_warnings: bool = Field(
        default=True,
        description="True once the researcher explicitly clicks 'Approve & Run' on a needs_confirmation card.",
    )


class ProvenanceRecord(BaseModel):
    tool_call_id: str
    step_id: str
    tool_name: str
    session_id: str
    executed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    latency_ms: float
    arguments: dict[str, Any]
    output: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)
    contact_sparsity_fraction: Optional[float] = None