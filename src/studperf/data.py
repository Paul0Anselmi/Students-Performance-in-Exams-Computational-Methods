from __future__ import annotations
 
from dataclasses import dataclass
from functools import lru_cache
from typing import Final
 
import numpy as np
import pandas as pd
from numpy.typing import NDArray
 
from studperf.config import get_settings


COLUMN_MAP: Final[dict[str, str]] = {
    "gender": "gender",
    "race/ethnicity": "race_ethnicity",
    "parental level of education": "parental_education",
    "lunch": "lunch",
    "test preparation course": "test_prep",
    "math score": "math_score",
    "reading score": "reading_score",
    "writing score": "writing_score",
}

SCORES: Final[tuple[str, ...]] = ("math_score", "reading_score", "writing_score")
FACTORS: Final[tuple[str, ...]] = ("gender", "race_ethnicity", "parental_education", "lunch", "test_prep")
EDUCATION_ORDER: Final[tuple[str, ...]] = (
    "some high school",
    "high school",
    "some college",
    "associate's degree",
    "bachelor's degree",
    "master's degree",
)

@lru_cache(maxsize=1)
def load_students() -> pd.DataFrame:
    """Lê o CSV uma única vez (cache), renomeia colunas e cria ``student_id``.
 
    Também valida o que os métodos supõem: 1000 linhas, nenhuma nota ausente
    e notas inteiras entre 0 e 100.
    """
    df = pd.read_csv(get_settings().raw_csv).rename(columns=COLUMN_MAP)
    df.insert(0, "student_id", np.arange(1, len(df) + 1))
    if df[list(SCORES)].isna().any().any():
        raise ValueError("há notas ausentes no CSV")
    if not df[list(SCORES)].apply(lambda s: s.between(0, 100)).all().all():
        raise ValueError("há notas fora do intervalo [0, 100]")
    return df
 
 
def _check(name: str, allowed: tuple[str, ...]) -> str:
    """Garante que o nome de coluna pedido existe (erro claro em vez de KeyError)."""
    if name not in allowed:
        raise ValueError(f"coluna inválida: {name!r}; use uma de {allowed}")
    return name
 
 
@dataclass(frozen=True)
class TwoSamples:
    """Duas amostras independentes: ``x`` (grupo 1) e ``y`` (grupo 2)."""
 
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    label_x: str
    label_y: str
 
 
def two_independent(score: str, factor: str, level_x: str, level_y: str) -> TwoSamples:
    """Nota ``score`` separada em dois níveis de ``factor``.
 
    Exemplo: ``two_independent("math_score", "test_prep", "completed", "none")``.
    Cada aluno cai em um único grupo, o que garante a independência entre amostras.
    """
    df = load_students()
    score, factor = _check(score, SCORES), _check(factor, FACTORS)
    x = df.loc[df[factor] == level_x, score].to_numpy(np.float64)
    y = df.loc[df[factor] == level_y, score].to_numpy(np.float64)
    return TwoSamples(x, y, level_x, level_y)
 
 
@dataclass(frozen=True)
class PairedSample:
    """Pares ``(x_i, y_i)`` medidos no mesmo aluno ``i``."""
 
    x: NDArray[np.float64]
    y: NDArray[np.float64]
 
    @property
    def diff(self) -> NDArray[np.float64]:
        """Diferenças ``Z_i = X_i - Y_i``: o teste pareado vira um problema de uma amostra."""
        return self.x - self.y
 
 
def paired(score_x: str, score_y: str) -> PairedSample:
    """Duas notas do mesmo aluno, lidas da mesma linha para nunca quebrar o par."""
    df = load_students()
    sx, sy = _check(score_x, SCORES), _check(score_y, SCORES)
    return PairedSample(df[sx].to_numpy(np.float64), df[sy].to_numpy(np.float64))
 
 
@dataclass(frozen=True)
class KSamples:
    """Resposta ``y`` e código inteiro de grupo ``g`` em ``0, ..., K-1``."""
 
    y: NDArray[np.float64]
    g: NDArray[np.int64]
    levels: tuple[str, ...]
 
 
def k_groups(score: str, factor: str, levels: tuple[str, ...] | None = None) -> KSamples:
    """Nota ``score`` com o rótulo de grupo de ``factor`` codificado em inteiros.
 
    Os códigos inteiros permitem montar a matriz indicadora (one-hot) e
    calcular as somas por grupo de todas as permutações com um produto matricial.
    """
    df = load_students()
    score, factor = _check(score, SCORES), _check(factor, FACTORS)
    levels = levels or tuple(sorted(df[factor].unique()))
    codes = pd.Categorical(df[factor], categories=list(levels), ordered=True).codes
    return KSamples(df[score].to_numpy(np.float64), codes.astype(np.int64), levels)
 
 
@dataclass(frozen=True)
class RegressionData:
    """Matriz de planejamento ``X`` (com intercepto) e resposta ``y``."""
 
    X: NDArray[np.float64]
    y: NDArray[np.float64]
    columns: tuple[str, ...]
 
 
def regression(response: str, factors: tuple[str, ...], baselines: dict[str, str]) -> RegressionData:
    """Monta ``X`` com variáveis indicadoras (dummies) e coluna de intercepto.
 
    O nível de referência de cada fator (``baselines``) fica sem dummy; cada
    coeficiente estimado é então a diferença média em relação a esse nível,
    mantidos os demais fatores fixos.
    """
    df = load_students()
    response = _check(response, SCORES)
    blocks = []
    for f in factors:
        dummies = pd.get_dummies(df[_check(f, FACTORS)], prefix=f, dtype=np.float64)
        blocks.append(dummies.drop(columns=f"{f}_{baselines[f]}"))
    design = pd.concat(blocks, axis=1)
    design.insert(0, "intercepto", 1.0)
    return RegressionData(design.to_numpy(np.float64), df[response].to_numpy(np.float64), tuple(design.columns))