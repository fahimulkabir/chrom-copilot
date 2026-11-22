from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query

from app.engine.data_store import DataStore, get_data_store

router = APIRouter(prefix="/data", tags=["UI Data Streaming"])


@router.get("/metadata")
def get_metadata(store: DataStore = Depends(get_data_store)) -> dict[str, Any]:
    return store.metadata


@router.get("/genes")
def get_genes(
    chrom: str = Query(..., pattern=r"^chr([0-9]+|X|Y)$"),
    start: int = Query(..., ge=0),
    end: int = Query(..., gt=0),
    store: DataStore = Depends(get_data_store),
) -> list[dict[str, Any]]:
    df = store.genes_1d
    sub = df[(df["chrom"] == chrom) & (df["start"] < end) & (df["end"] > start)].copy()
    sub = sub.replace({np.nan: None})
    return sub.to_dict(orient="records")


@router.get("/tracks-1d")
def get_tracks_1d(
    chrom: str = Query(..., pattern=r"^chr([0-9]+|X|Y)$"),
    start: int = Query(..., ge=0),
    end: int = Query(..., gt=0),
    resolution_bp: int = Query(50_000),
    store: DataStore = Depends(get_data_store),
) -> list[dict[str, Any]]:
    df = store.bin_signals_1d
    sub = df[
        (df["chrom"] == chrom)
        & (df["resolution_bp"] == resolution_bp)
        & (df["end"] > start)
        & (df["start"] < end)
    ].copy()
    sub = sub.replace({np.nan: None})
    return sub.to_dict(orient="records")


@router.get("/structure-3d")
def get_structure_3d(
    entity_id: str = Query(...),
    chrom: str = Query(..., pattern=r"^chr([0-9]+|X|Y)$"),
    start: int = Query(..., ge=0),
    end: int = Query(..., gt=0),
    resolution_bp: int = Query(50_000),
    store: DataStore = Depends(get_data_store),
) -> list[dict[str, Any]]:
    sub = store.get_3d_coords(entity_id, chrom, resolution_bp, start=start, end=end)
    if sub.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No 3D coordinates for {entity_id} on {chrom}:{start}-{end} @ {resolution_bp}",
        )
    return sub[["bin_index", "start", "end", "x", "y", "z", "rg"]].to_dict(orient="records")


@router.get("/bookmarks/{bookmark_id}")
def get_bookmark(
    bookmark_id: str, store: DataStore = Depends(get_data_store)
) -> dict[str, Any]:
    patch = store.get_bookmark(bookmark_id)
    if patch is None:
        raise HTTPException(status_code=404, detail=f"Bookmark '{bookmark_id}' not found.")
    return patch