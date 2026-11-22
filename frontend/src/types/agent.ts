export interface PlannedToolCall {
  step_id: string;
  tool_name: string;
  arguments: Record<string, unknown>;
  rationale: string;
}

export interface PlanVerificationResult {
  status: "approved" | "needs_confirmation" | "blocked";
  plan: PlannedToolCall[];
  checks_passed: string[];
  warnings: string[];
  clarification_prompt: string | null;
}

export interface ProvenanceClaim {
  claim_id: string;
  text: string;
  cited_tool_call_ids: string[];
  is_verified: boolean;
  bookmark_id: string | null;
}

export interface CopilotResponse {
  session_id: string;
  claims: ProvenanceClaim[];
  viewport_patch: Record<string, unknown> | null;
  executed_tool_call_ids: string[];
}
