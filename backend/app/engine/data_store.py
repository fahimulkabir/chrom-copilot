# Loads metadata.json, .mcool handles, 3D Parquet folds, and 1D tracks once into memory, caches expensive chromosome-wide computations (like cooltools.eigs_cis), and persists viewport bookmarks.

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import cooler
import numpy as np
import pandas as pd

from app.config import get_settings


class DataStore:
    """In-memory data access layer for bulk .mcool, 3D Parquet folds, and 1D tracks."""

    def __init__(self, processed_dir: Path, cache_dir: Path) -> None:
        self.processed_dir = processed_dir
        self.cache_dir = cache_dir
        meta_path = self.processed_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Dataset manifest not found at {meta_path}. Run Part 1 scripts first."
            )

        self.metadata: dict[str, Any] = json.loads(meta_path.read_text())
        self.dataset_id: str = self.metadata["dataset_id"]

        # Load Parquet tables once into memory for sub-200ms queries
        self.structures_3d: pd.DataFrame = pd.read_parquet(
            self.processed_dir / self.metadata["structures_3d_path"]
        )
        self.genes_1d: pd.DataFrame = pd.read_parquet(
            self.processed_dir / self.metadata["tracks_1d"]["genes_path"]
        )
        self.bin_signals_1d: pd.DataFrame = pd.read_parquet(
            self.processed_dir / self.metadata["tracks_1d"]["bin_signals_path"]
        )

        # Open cooler handles keyed by (entity_id, resolution_bp)
        self._coolers: dict[tuple[str, int], cooler.Cooler] = {}
        for cid, cinfo in self.metadata["cohorts"].items():
            mcool_path = self.processed_dir / cinfo["mcool_path"]
            for res in self.metadata["available_resolutions_bp"]:
                self._coolers[(cid, int(res))] = cooler.Cooler(
                    f"{mcool_path}::/resolutions/{res}"
                )

        # Caches for expensive chromosome-wide cooltools calls & bookmarks
        self._eigs_cache: dict[tuple[str, str, int], pd.DataFrame] = {}
        self._bookmarks_path = self.cache_dir / "bookmarks.json"
        self._bookmarks: dict[str, dict[str, Any]] = self._load_bookmarks()

    def validate_dataset(self, dataset_id: str) -> None:
        if dataset_id != self.dataset_id:
            raise ValueError(
                f"Unknown dataset_id '{dataset_id}'. Available: '{self.dataset_id}'."
            )

    def get_cooler(self, entity_id: str, resolution_bp: int) -> cooler.Cooler:
        key = (entity_id, int(resolution_bp))
        if key not in self._coolers:
            raise ValueError(
                f"No cooler matrix for entity='{entity_id}' at resolution={resolution_bp}."
            )
        return self._coolers[key]

    def get_matrix(
        self,
        entity_ids: list[str],
        chrom: str,
        start: int,
        end: int,
        resolution_bp: int,
        balance: bool = True,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """Fetch and aggregate (mean/sum) 2D contact matrix for a CohortSpec."""
        region = f"{chrom}:{start}-{end}"
        matrices: list[np.ndarray] = []
        bins_df: pd.DataFrame | None = None

        for ent in entity_ids:
            clr = self.get_cooler(ent, resolution_bp)
            if bins_df is None:
                bins_df = clr.bins().fetch(region).reset_index(drop=True)
            mat = clr.matrix(balance=balance).fetch(region).astype(np.float64)
            matrices.append(mat)

        assert bins_df is not None
        if len(matrices) == 1:
            return matrices[0], bins_df
        # Pseudobulk aggregation for multi-entity / single-cell cohorts
        stacked = np.stack([np.nan_to_num(m, nan=0.0) for m in matrices], axis=0)
        return np.mean(stacked, axis=0), bins_df

    def get_3d_coords(
        self,
        entity_id: str,
        chrom: str,
        resolution_bp: int,
        start: int | None = None,
        end: int | None = None,
    ) -> pd.DataFrame:
        df = self.structures_3d
        mask = (
            (df["entity_id"] == entity_id)
            & (df["chrom"] == chrom)
            & (df["resolution_bp"] == int(resolution_bp))
        )
        if start is not None:
            mask &= df["end"] > start
        if end is not None:
            mask &= df["start"] < end
        return df.loc[mask].sort_values("bin_index").reset_index(drop=True)

    def get_1d_phasing_track(self, chrom: str, resolution_bp: int) -> pd.DataFrame:
        df = self.bin_signals_1d
        mask = (df["chrom"] == chrom) & (df["resolution_bp"] == int(resolution_bp))
        return (
            df.loc[mask, ["chrom", "start", "end", "gene_density"]]
            .sort_values("start")
            .reset_index(drop=True)
        )

    def get_overlapping_genes(self, chrom: str, start: int, end: int) -> list[str]:
        df = self.genes_1d
        mask = (
            (df["chrom"] == chrom)
            & (df["start"] < end)
            & (df["end"] > start)
            & (df["feature_type"] == "gene")
        )
        return sorted(df.loc[mask, "symbol"].dropna().unique().tolist())

    def save_bookmark(self, bookmark_id: str, patch: dict[str, Any]) -> None:
        self._bookmarks[bookmark_id] = patch
        self._bookmarks_path.write_text(json.dumps(self._bookmarks, indent=2))

    def get_bookmark(self, bookmark_id: str) -> dict[str, Any] | None:
        return self._bookmarks.get(bookmark_id)

    def _load_bookmarks(self) -> dict[str, dict[str, Any]]:
        if self._bookmarks_path.exists():
            try:
                return json.loads(self._bookmarks_path.read_text())
            except json.JSONDecodeError:
                return {}
        return {}


@lru_cache(maxsize=1)
def get_data_store() -> DataStore:
    settings = get_settings()
    return DataStore(settings.processed_dir, settings.cache_dir)