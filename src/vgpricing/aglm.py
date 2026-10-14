"""AGLM (Fujita, Tanaka, Kondo e Iwasawa, 2020): GLM regularizado com codificações que capturam não linearidade
sem perder a forma de tabela de fatores.

- Variáveis O (dummies ordinais, "thermometer"): d_j(x) = 1 se o nível de x < j. Com LASSO, equivalem ao fused lasso:
  faixas vizinhas parecidas são fundidas automaticamente.
- Variáveis L: l_j(x) = |x - b_j| nos nós internos b_j, mais o próprio x. A contribuição vira uma curva linear por
  partes (contínua), e cada coeficiente é a mudança de inclinação num nó.

O pacote `aglm` (R) faz o mesmo sobre o glmnet; aqui a codificação é montada em Python e o ajuste usa o glmnet do
projeto (`elasticnet.py`), com offset (inclusive a priori da SOA). Na máquina do usuário dá para trocar pelo pacote.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def nos_por_quantil(x, n_nos: int) -> np.ndarray:
    """Nós internos nos quantis de x (bins de frequência igual, como o padrão do pacote aglm)."""
    q = np.quantile(np.asarray(x, float), np.linspace(0, 1, n_nos + 2)[1:-1])
    return np.unique(np.round(q, 6))


def variaveis_L(x, nos, prefixo: str) -> pd.DataFrame:
    x = np.asarray(x, float)
    return pd.DataFrame({f"{prefixo}_L{j}": np.abs(x - b) for j, b in enumerate(nos)})


def variaveis_O(niveis: pd.Series, ordem: list[str], prefixo: str) -> pd.DataFrame:
    """d_j = 1 se o nível vem antes do j-ésimo da ordem (j = 2..m)."""
    pos = niveis.astype(str).map({n: i for i, n in enumerate(ordem)}).to_numpy()
    return pd.DataFrame({f"{prefixo}_O{j}": (pos < j).astype(float) for j in range(1, len(ordem))})


def codificar(df: pd.DataFrame, numericas: dict[str, np.ndarray], ordenadas: dict[str, list[str]]) -> pd.DataFrame:
    """Colunas AGLM: para cada numérica, as variáveis L nos nós dados (o termo linear fica a cargo do chamador);
    para cada ordenada, as variáveis O na ordem dada."""
    partes = [variaveis_L(df[v], nos, v) for v, nos in numericas.items()]
    partes += [variaveis_O(df[v], ordem, v) for v, ordem in ordenadas.items()]
    return pd.concat([p.set_index(df.index) for p in partes], axis=1)


def efeito_L(x, nos, coeficientes: dict[str, float], prefixo: str) -> np.ndarray:
    """Contribuição (escala log) das variáveis L de uma numérica."""
    x = np.asarray(x, float)
    return sum(coeficientes.get(f"{prefixo}_L{j}", 0.0) * np.abs(x - b) for j, b in enumerate(nos)) + 0 * x


def fatores_O(ordem: list[str], coeficientes: dict[str, float], prefixo: str) -> dict[str, float]:
    """Fator multiplicativo de cada nível de uma ordenada: exp(soma dos coeficientes O ativos)."""
    return {n: float(np.exp(sum(coeficientes.get(f"{prefixo}_O{j}", 0.0) for j in range(i + 1, len(ordem)))))
            for i, n in enumerate(ordem)}
