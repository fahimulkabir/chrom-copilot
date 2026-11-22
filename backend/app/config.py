# Resolves data/processed and data/cache reliably whether you run uvicorn/pytest from the repo root or from backend/.

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    data_root: str = "./data/processed"
    cache_root: str = "./data/cache"
    default_dataset_id: str = "rao2014_gm12878_k562"
    llm_provider: str = "gemini"
    gemini_api_key: str = ""

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def processed_dir(self) -> Path:
        p = Path(self.data_root)
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()

    @property
    def cache_dir(self) -> Path:
        p = Path(self.cache_root)
        path = p if p.is_absolute() else (REPO_ROOT / p).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()