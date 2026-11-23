from __future__ import annotations

import re
from typing import Any

import numpy as np
from pydantic import BaseModel, ValidationError

from app.engine.data_store import DataStore
from app.schemas.agent import PlannedToolCall, PlanVerificationResult
from app.schemas.tools import (
    ABCompartmentInput,
    ContactMapInput,
    DifferentialTestInput,
    EmitViewportSyncInput,
    Pairwise3DDistanceInput,
    SpatialNeighbors3DInput,
    StratifyBy1DInput,
    TadInsulationInput,
)
from app.schemas.viewport import ViewportState

TOOL_INPUT_MODELS: dict[str, type[BaseModel]] = {
    "stratify_by_1d_feature": StratifyBy1DInput,
    "get_contact_map": ContactMapInput,
    "compute_tad_insulation": TadInsulationInput,
    "compute_ab_compartments": ABCompartmentInput,
    "measure_3d_pairwise_distance": Pairwise3DDistanceInput,
    "find_3d_spatial_neighbors": SpatialNeighbors3DInput,
    "run_differential_test": DifferentialTestInput,
    "emit_viewport_sync": EmitViewportSyncInput,
}

STEP_REF_PATTERN = re.compile(r"^\$([A-Za-z0-9_]+)\.([A-Za-z0-9_.]+)$")


def _contains_step_ref(obj: Any) -> bool:
    if isinstance(obj, str):
        return bool(STEP_REF_PATTERN.match(obj))
    if isinstance(obj, list):
        return any(_contains_step_ref(x) for x in obj)
    if isinstance(obj, dict):
        return any(_contains_step_ref(v) for v in obj.values())
    return False


def _extract_cohorts(args: dict[str, Any]) -> list[dict[str, Any]]:
    cohorts: list[dict[str, Any]] = []
    for key in ("cohort", "compare_cohort", "source_cohort"):
        val = args.get(key)
        if isinstance(val, dict) and "entity_ids" in val:
            cohorts.append(val)
    active = args.get("active_cohorts")
    if isinstance(active, list):
        for item in active:
            if isinstance(item, dict) and "entity_ids" in item:
                cohorts.append(item)
    return cohorts


def _extract_intervals(args: dict[str, Any]) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    for key in ("locus", "locus_a", "locus_b"):
        val = args.get(key)
        if isinstance(val, dict) and {"chrom", "start", "end"}.issubset(val.keys()):
            intervals.append(val)
    return intervals


def _quick_sparsity(
    store: DataStore, entity_ids: list[str], chrom: str, start: int, end: int, res: int
) -> float:
    raw_mat, _ = store.get_matrix(entity_ids, chrom, start, end, res, balance=False)
    n = raw_mat.shape[0]
    if n == 0:
        return 1.0
    vals = raw_mat[np.triu_indices(n, k=0)]
    valid_pos = np.isfinite(vals) & (vals > 0)
    return round(float(1.0 - np.mean(valid_pos)), 4)


def verify_plan(
    store: DataStore,
    plan: list[PlannedToolCall],
    viewport: ViewportState,
    *,
    sparsity_threshold: float = 0.95,
    min_sc_cells: int = 20,
    planner_clarification: str | None = None,
) -> PlanVerificationResult:
    """Deterministic pre-execution safety, bounds, sample-size, and sparsity gate."""
    checks_passed: list[str] = []
    warnings: list[str] = []
    block_reasons: list[str] = []
    confirm_reasons: list[str] = []

    meta = store.metadata
    avail_res = set(int(r) for r in meta["available_resolutions_bp"])
    chrom_sizes: dict[str, int] = {k: int(v) for k, v in meta["chromosomes"].items()}
    known_entities: set[str] = set()
    for cinfo in meta["cohorts"].values():
        known_entities.update(cinfo["entity_ids"])

    # 1. Genome build & dataset identity match
    if viewport.dataset_id != store.dataset_id:
        block_reasons.append(
            f"Viewport dataset '{viewport.dataset_id}' does not match loaded dataset '{store.dataset_id}'."
        )
    elif viewport.genome_build != meta["genome_build"]:
        block_reasons.append(
            f"Genome build mismatch: viewport requested '{viewport.genome_build}', "
            f"but dataset '{store.dataset_id}' is aligned to '{meta['genome_build']}'."
        )
    else:
        checks_passed.append(f"genome_build_match:{meta['genome_build']}")

    # 2. Empty or explicitly ambiguous plan
    if not plan:
        msg = (
            planner_clarification
            or "Query is underspecified. Specify a target gene/locus, cohort, or select a 3D bead."
        )
        return PlanVerificationResult(
            status="blocked" if block_reasons else "needs_confirmation",
            plan=[],
            checks_passed=checks_passed,
            warnings=block_reasons + warnings,
            clarification_prompt=msg,
        )

    seen_step_ids: set[str] = set()

    for step in plan:
        # 3. Strict 8-tool allowlist (no arbitrary code execution)
        if step.tool_name not in TOOL_INPUT_MODELS:
            block_reasons.append(
                f"Step '{step.step_id}' requested disallowed tool '{step.tool_name}'. "
                f"Only the 8 verified bioinformatics tools are permitted."
            )
            continue

        args = step.arguments
        has_refs = _contains_step_ref(args)

        # Validate forward references point only to earlier steps
        if has_refs:
            for val in args.values():
                if isinstance(val, str):
                    m = STEP_REF_PATTERN.match(val)
                    if m and m.group(1) not in seen_step_ids:
                        block_reasons.append(
                            f"Step '{step.step_id}' references unknown or future step '${m.group(1)}'."
                        )
        else:
            # Strict Pydantic schema validation on concrete arguments
            model_cls = TOOL_INPUT_MODELS[step.tool_name]
            try:
                model_cls.model_validate(args)
            except ValidationError as exc:
                err_msg = exc.errors()[0]["msg"]
                block_reasons.append(
                    f"Step '{step.step_id}' ({step.tool_name}) failed schema validation: {err_msg}"
                )
                continue

        # 4. Dataset ID check per step
        if args.get("dataset_id") != store.dataset_id:
            block_reasons.append(
                f"Step '{step.step_id}' targets unknown dataset_id '{args.get('dataset_id')}'."
            )

        # 5. Resolution availability & tool-specific resolution guardrails
        if "resolution_bp" in args and isinstance(args["resolution_bp"], int):
            res = args["resolution_bp"]
            if res not in avail_res:
                block_reasons.append(
                    f"Step '{step.step_id}' requested unavailable resolution {res} bp "
                    f"(available: {sorted(avail_res)})."
                )
            elif step.tool_name == "compute_tad_insulation" and res > 100_000:
                block_reasons.append(
                    f"Step '{step.step_id}': TAD insulation requires resolution <= 100 kb (got {res // 1000} kb)."
                )
            elif step.tool_name == "compute_ab_compartments" and res < 100_000:
                block_reasons.append(
                    f"Step '{step.step_id}': A/B compartments require resolution >= 100 kb (got {res // 1000} kb)."
                )

        # 6. Chromosome & genomic coordinate bounds checks
        chrom_arg = args.get("chrom") or args.get("anchor_chrom")
        if isinstance(chrom_arg, str) and chrom_arg not in chrom_sizes:
            block_reasons.append(
                f"Step '{step.step_id}' references chromosome '{chrom_arg}' not in dataset "
                f"(available: {sorted(chrom_sizes.keys())})."
            )

        for iv in _extract_intervals(args):
            c = iv["chrom"]
            if c not in chrom_sizes:
                block_reasons.append(
                    f"Step '{step.step_id}' locus chromosome '{c}' not in dataset."
                )
            elif iv["start"] < 0 or iv["end"] <= iv["start"] or iv["end"] > chrom_sizes[c]:
                block_reasons.append(
                    f"Step '{step.step_id}' locus {c}:{iv['start']}-{iv['end']} exceeds "
                    f"chromosome bounds (0–{chrom_sizes[c]})."
                )

        # 7. Cohort entity validation & single-cell sample-size guardrail
        for coh in _extract_cohorts(args):
            ent_ids = coh.get("entity_ids", [])
            unknown = [e for e in ent_ids if e not in known_entities]
            if unknown and viewport.dataset_mode == "bulk_condition":
                block_reasons.append(
                    f"Step '{step.step_id}' references unknown cohort entity {unknown} "
                    f"(available: {sorted(known_entities)})."
                )
            if viewport.dataset_mode == "single_cell" and len(ent_ids) < min_sc_cells:
                msg = (
                    f"Step '{step.step_id}' cohort '{coh.get('label', coh.get('cohort_id'))}' "
                    f"contains only {len(ent_ids)} single cells (below recommended minimum of {min_sc_cells})."
                )
                warnings.append(msg)
                confirm_reasons.append(
                    f"Proceed with small single-cell cohort (n={len(ent_ids)} < {min_sc_cells})?"
                )

        # 8. 3D coordinate availability check
        if step.tool_name == "find_3d_spatial_neighbors":
            a_chrom = args.get("anchor_chrom")
            a_bin = args.get("anchor_bin_index")
            a_res = args.get("resolution_bp", 50_000)
            coh = args.get("cohort", {})
            ent_ids = coh.get("entity_ids", []) if isinstance(coh, dict) else []
            if ent_ids and ent_ids[0] in known_entities and a_chrom in chrom_sizes and a_res in avail_res:
                df_3d = store.get_3d_coords(ent_ids[0], a_chrom, a_res)
                if df_3d[df_3d["bin_index"] == a_bin].empty:
                    block_reasons.append(
                        f"Step '{step.step_id}': 3D anchor bin {a_bin} on {a_chrom} ({a_res // 1000} kb) "
                        f"is outside cached 3D target windows."
                    )

        # 9. Statistical validity pre-check (Tool 7)
        if step.tool_name == "run_differential_test":
            t_type = args.get("test_type", "mann_whitney_u")
            ga = args.get("group_a_values")
            gb = args.get("group_b_values")
            small_concrete = (
                isinstance(ga, list)
                and isinstance(gb, list)
                and (len(ga) < 3 or len(gb) < 3)
            )
            refs_bulk_dist = (
                viewport.dataset_mode == "bulk_condition"
                and (
                    (isinstance(ga, str) and "entity_distances" in ga)
                    or (isinstance(gb, str) and "entity_distances" in gb)
                )
            )
            if t_type != "delta_only" and (small_concrete or refs_bulk_dist):
                msg = (
                    f"Step '{step.step_id}' requests '{t_type}' on fewer than 3 observations per group. "
                    f"Deterministic engine will downgrade to 'delta_only' (statistically_valid=False)."
                )
                warnings.append(msg)
                confirm_reasons.append(
                    "Sample size per group is under 3 (bulk condition mode). Run as descriptive delta_only?"
                )

        # 10. Live 2D Contact Sparsity Pre-Check
        if (
            step.tool_name in ("get_contact_map", "compute_tad_insulation")
            and not block_reasons
            and not has_refs
        ):
            coh = args.get("cohort", {})
            loc = args.get("locus", {})
            res = args.get("resolution_bp", 50_000)
            if coh and loc and res in avail_res and loc["chrom"] in chrom_sizes:
                try:
                    sp = _quick_sparsity(
                        store, coh["entity_ids"], loc["chrom"], loc["start"], loc["end"], res
                    )
                    if sp > sparsity_threshold:
                        msg = (
                            f"Step '{step.step_id}' contact sparsity is {sp:.1%} at {res // 1000} kb "
                            f"(exceeds threshold {sparsity_threshold:.0%})."
                        )
                        warnings.append(msg)
                        confirm_reasons.append(
                            f"High contact sparsity ({sp:.1%}) detected at {res // 1000} kb. "
                            f"Confirm execution or switch to a coarser resolution (100 kb / 250 kb)?"
                        )
                except Exception as exc:
                    block_reasons.append(f"Step '{step.step_id}' sparsity pre-check failed: {exc}")

        seen_step_ids.add(step.step_id)

    if not block_reasons:
        checks_passed.extend(
            [
                "strict_8_tool_allowlist",
                "pydantic_schema_valid",
                "genomic_bounds_valid",
                "resolution_available",
                "cohorts_verified",
            ]
        )

    if block_reasons:
        return PlanVerificationResult(
            status="blocked",
            plan=plan,
            checks_passed=checks_passed,
            warnings=block_reasons + warnings,
            clarification_prompt="Plan blocked due to safety/bounds violations. Review warnings and revise query.",
        )

    if confirm_reasons or planner_clarification:
        prompt = planner_clarification or " ".join(dict.fromkeys(confirm_reasons))
        return PlanVerificationResult(
            status="needs_confirmation",
            plan=plan,
            checks_passed=checks_passed,
            warnings=warnings,
            clarification_prompt=prompt,
        )

    checks_passed.append("sparsity_within_threshold")
    return PlanVerificationResult(
        status="approved",
        plan=plan,
        checks_passed=checks_passed,
        warnings=warnings,
        clarification_prompt=None,
    )