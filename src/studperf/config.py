"""Configuração do projeto, lida do arquivo .env (B, semente e alfa)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

#: Raiz do repositório (duas pastas acima deste arquivo: src/studperf/config.py).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """Parâmetros de execução: caminho do CSV, número B de reamostras, semente e alfa."""

    raw_csv: Path
    n_resamples: int
    seed: int
    alpha: float


def get_settings() -> Settings:
    """Lê o .env e devolve Settings, com valores padrão seguros."""
    return Settings(
        raw_csv=PROJECT_ROOT / os.getenv("RAW_CSV", "data/raw/StudentsPerformance.csv"),
        n_resamples=int(os.getenv("N_RESAMPLES", "10000")),
        seed=int(os.getenv("SEED", "2025")),
        alpha=float(os.getenv("ALPHA", "0.05")),
    )