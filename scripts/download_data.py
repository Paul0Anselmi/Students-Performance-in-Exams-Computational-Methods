"""Baixa o CSV do Kaggle uma única vez para data/raw (rodar só se o arquivo não existir)."""

import shutil
from pathlib import Path

import kagglehub

DATASET = "spscientist/students-performance-in-exams"
DEST = Path(__file__).resolve().parents[1] / "data" / "raw" / "StudentsPerformance.csv"


def main() -> None:
    if DEST.exists():
        print(f"Já existe: {DEST}")
        return
    pasta = Path(kagglehub.dataset_download(DATASET))
    DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(pasta / "StudentsPerformance.csv", DEST)
    print(f"CSV salvo em {DEST}")


if __name__ == "__main__":
    main()