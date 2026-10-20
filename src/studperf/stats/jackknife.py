r"""Jackknife (Quenouille, 1947; Tukey, 1958) e validação cruzada k-fold.

Jackknife: a :math:`i`-ésima amostra :math:`y_{-i}` remove a observação
:math:`i`. Com :math:`\hat\theta_{(i)}` e
:math:`\hat\theta_{(\cdot)}=\tfrac1n\sum_i\hat\theta_{(i)}`:

.. math::
    \widehat{\text{vício}}_{jack} = (n-1)(\hat\theta_{(\cdot)}-\hat\theta),\qquad
    \widehat{EP}_{jack} = \sqrt{\tfrac{n-1}{n}\sum_i(\hat\theta_{(i)}-\hat\theta_{(\cdot)})^2}.

O fator :math:`(n-1)/n` compensa a semelhança entre as amostras jackknife e a
original. Observação: o jackknife é inconsistente para estatísticas não
suaves (ex.: mediana), por isso aqui é aplicado a coeficientes de MQO.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def leave_one_out_index(n: int) -> NDArray[np.int64]:
    """Matriz ``(n, n-1)`` cuja linha ``i`` lista todos os índices exceto ``i``."""
    j = np.arange(n - 1)[None, :]
    return j + (j >= np.arange(n)[:, None])


@dataclass(frozen=True)
class JackknifeResult:
    """Estimativa original, réplicas leave-one-out, vício e EP jackknife."""

    estimate: FloatArray
    replicates: FloatArray = field(repr=False)
    bias: FloatArray
    se: FloatArray


def jackknife_ols(X: FloatArray, y: FloatArray, chunk: int = 200) -> JackknifeResult:
    r"""Jackknife dos coeficientes de MQO.

    Usa a atualização de posto um
    :math:`X_{-i}^TX_{-i} = X^TX - x_ix_i^T` e
    :math:`X_{-i}^Ty_{-i} = X^Ty - x_iy_i`, resolvendo os :math:`n` sistemas em
    lote (equivalente a reajustar o modelo :math:`n` vezes).
    """
    n = y.size
    xtx = X.T @ X
    xty = X.T @ y
    beta = np.linalg.solve(xtx, xty)
    reps: list[FloatArray] = []
    for start in range(0, n, chunk):
        xi = X[start : start + chunk]
        a = xtx[None] - np.einsum("bi,bj->bij", xi, xi)
        b = xty[None] - xi * y[start : start + chunk, None]
        reps.append(np.linalg.solve(a, b[..., None])[..., 0])
    rep = np.concatenate(reps)
    mean_rep = rep.mean(axis=0)
    se = np.sqrt((n - 1) / n * ((rep - mean_rep) ** 2).sum(axis=0))
    return JackknifeResult(beta, rep, (n - 1) * (mean_rep - beta), se)


def kfold_rmse(
    X: FloatArray, y: FloatArray, *, k: int = 10, rng: np.random.Generator | None = None
) -> FloatArray:
    r"""Raiz do erro quadrático médio de predição por validação cruzada k-fold.

    Os dados são particionados aleatoriamente em :math:`k` partes; cada parte
    é predita pelo modelo ajustado nas outras :math:`k-1`. Objetivo
    preditivo: estimar o erro em observações novas (sinaliza *overfitting*).
    Retorna o RMSE de cada fold.
    """
    rng = rng or np.random.default_rng()
    folds = rng.permutation(y.size) % k
    out = np.empty(k)
    for f in range(k):
        test = folds == f
        beta = np.linalg.lstsq(X[~test], y[~test], rcond=None)[0]
        out[f] = np.sqrt(np.mean((y[test] - X[test] @ beta) ** 2))
    return out
