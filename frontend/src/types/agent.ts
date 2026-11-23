import type { ViewportState } from "./viewport";

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
  viewport_patch: Partial<ViewportState> | null;
  executed_tool_call_ids: string[];
}

export interface AgentPlanRequest {
  session_id: string;
  user_query: string;
  viewport: ViewportState;
  use_llm: boolean;
  sparsity_threshold?: number;
}

export interface AgentPlanResponse {
  session_id: string;
  plan_id: string;
  planner_mode: "gemini_llm" | "deterministic_fallback";
  verification: PlanVerificationResult;
}

export interface AgentExecuteRequest {
  session_id: string;
  plan_id?: string | null;
  user_query: string;
  viewport: ViewportState;
  approved_plan: PlannedToolCall[];
  use_llm: boolean;
  allow_confirmed_warnings: boolean;
}

export interface ProvenanceRecord {
  tool_call_id: string;
  step_id: string;
  tool_name: string;
  session_id: string;
  executed_at: string;
  latency_ms: number;
  arguments: Record<string, unknown>;
  output: Record<string, unknown>;
  warnings: string[];
  contact_sparsity_fraction: number | null;
}
