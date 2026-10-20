r"""Métodos paramétricos clássicos, usados como ponto de comparação.

Todos supõem normalidade (dos dados ou, para n grande, da média pelo TCL):

* duas populações independentes -> teste t de Welch (não supõe variâncias iguais);
* duas populações pareadas      -> teste t pareado sobre :math:`Z_i = X_i - Y_i`;
* K > 2 populações              -> ANOVA de um fator (supõe também homocedasticidade).

Cada função devolve a estimativa, a estatística, o valor-p e o IC, para que
os resultados possam ser postos lado a lado com permutação e bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats

Alternative = Literal["two-sided", "less", "greater"]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ParametricResult:
    """Resultado de um teste paramétrico.

    Attributes:
        estimate: estimativa pontual (diferença de médias; ``nan`` na ANOVA).
        statistic: estatística t ou F observada.
        df: graus de liberdade (na ANOVA, o do numerador e o do denominador).
        p_value: valor-p pela distribuição t ou F.
        ci: intervalo de confiança para ``estimate`` (``nan`` na ANOVA).
    """

    estimate: float
    statistic: float
    df: tuple[float, ...]
    p_value: float
    ci: tuple[float, float]


def welch_t_test(
    x: FloatArray, y: FloatArray, *, alternative: Alternative = "two-sided", alpha: float = 0.05
) -> ParametricResult:
    r"""Teste t de Welch para :math:`H_0: \mu_X = \mu_Y`.

    .. math:: t = \frac{\bar X - \bar Y}{\sqrt{S_X^2/n + S_Y^2/m}},

    com graus de liberdade de Welch–Satterthwaite. O IC é bilateral de nível
    :math:`1-\alpha` para :math:`\mu_X - \mu_Y`.
    """
    res = stats.ttest_ind(x, y, equal_var=False, alternative=alternative)
    ci = stats.ttest_ind(x, y, equal_var=False).confidence_interval(1 - alpha)
    return ParametricResult(
        float(np.mean(x) - np.mean(y)), float(res.statistic), (float(res.df),), float(res.pvalue), (float(ci.low), float(ci.high))
    )


def paired_t_test(diff: FloatArray, *, alternative: Alternative = "two-sided", alpha: float = 0.05) -> ParametricResult:
    r"""Teste t pareado para :math:`H_0: \mu_Z = 0`, com :math:`Z_i = X_i - Y_i`.

    .. math:: t = \frac{\bar Z}{S_Z/\sqrt n}, \qquad t \sim t_{n-1} \text{ sob } H_0.
    """
    res = stats.ttest_1samp(diff, 0.0, alternative=alternative)
    ci = stats.ttest_1samp(diff, 0.0).confidence_interval(1 - alpha)
    return ParametricResult(
        float(np.mean(diff)), float(res.statistic), (float(res.df),), float(res.pvalue), (float(ci.low), float(ci.high))
    )


def one_way_anova(y: FloatArray, groups: NDArray[np.int64]) -> ParametricResult:
    r"""ANOVA de um fator para :math:`H_0: \mu_1 = \dots = \mu_K`.

    .. math:: F = \frac{SQE/(K-1)}{SQD/(N-K)} \sim F_{K-1,\,N-K} \text{ sob } H_0.
    """
    k = int(groups.max()) + 1
    res = stats.f_oneway(*(y[groups == j] for j in range(k)))
    return ParametricResult(
        float("nan"), float(res.statistic), (k - 1.0, float(y.size - k)), float(res.pvalue), (float("nan"), float("nan"))
    )
