from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


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