from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.engine.data_store import DataStore, get_data_store
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
from app.schemas.tools import (
    ABCompartmentInput,
    ABCompartmentOutput,
    ContactMapInput,
    ContactMapOutput,
    DifferentialTestInput,
    DifferentialTestOutput,
    EmitViewportSyncInput,
    EmitViewportSyncOutput,
    Pairwise3DDistanceInput,
    Pairwise3DDistanceOutput,
    SpatialNeighbors3DInput,
    SpatialNeighbors3DOutput,
    StratifyBy1DInput,
    StratifyBy1DOutput,
    TadInsulationInput,
    TadInsulationOutput,
)

router = APIRouter(prefix="/tools", tags=["Deterministic Bioinformatics Tools"])


def _exec(fn, store: DataStore, req):
    try:
        return fn(store, req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/stratify-1d", response_model=StratifyBy1DOutput)
def stratify_1d(req: StratifyBy1DInput, store: DataStore = Depends(get_data_store)):
    return _exec(run_stratify_by_1d_feature, store, req)


@router.post("/contact-map", response_model=ContactMapOutput)
def contact_map(req: ContactMapInput, store: DataStore = Depends(get_data_store)):
    return _exec(run_get_contact_map, store, req)


@router.post("/tad-insulation", response_model=TadInsulationOutput)
def tad_insulation(req: TadInsulationInput, store: DataStore = Depends(get_data_store)):
    return _exec(run_compute_tad_insulation, store, req)


@router.post("/ab-compartments", response_model=ABCompartmentOutput)
def ab_compartments(req: ABCompartmentInput, store: DataStore = Depends(get_data_store)):
    return _exec(run_compute_ab_compartments, store, req)


@router.post("/3d-distance", response_model=Pairwise3DDistanceOutput)
def pairwise_3d_distance(
    req: Pairwise3DDistanceInput, store: DataStore = Depends(get_data_store)
):
    return _exec(run_measure_3d_pairwise_distance, store, req)


@router.post("/3d-neighbors", response_model=SpatialNeighbors3DOutput)
def spatial_3d_neighbors(
    req: SpatialNeighbors3DInput, store: DataStore = Depends(get_data_store)
):
    return _exec(run_find_3d_spatial_neighbors, store, req)


@router.post("/differential-test", response_model=DifferentialTestOutput)
def differential_test(
    req: DifferentialTestInput, store: DataStore = Depends(get_data_store)
):
    return _exec(run_differential_test, store, req)


@router.post("/viewport-sync", response_model=EmitViewportSyncOutput)
def viewport_sync(
    req: EmitViewportSyncInput, store: DataStore = Depends(get_data_store)
):
    return _exec(run_emit_viewport_sync, store, req)