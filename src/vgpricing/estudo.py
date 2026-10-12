"""Estudo de experiência: A/E por variável, com intervalo exato e exposição.

Funciona com qualquer tabela de células que tenha observados e esperados (dados da SOA, carteira da
seguradora). É a primeira leitura, univariada; o modelo (taxa manual) separa os efeitos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2


def ic_poisson(observados, nivel: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Intervalo exato (Garwood) para a média de uma Poisson observada."""
    o = np.asarray(observados, float)
    a = 1 - nivel
    inf = np.where(o > 0, chi2.ppf(a / 2, 2 * o) / 2, 0.0)
    sup = chi2.ppf(1 - a / 2, 2 * (o + 1)) / 2
    return inf, sup


def tabela_ae(df: pd.DataFrame, por: str | list[str], obs: str = "obitos", esp: str = "esperados_n",
              exposicao: str = "vidas_ano", obs_valor: str | None = None, esp_valor: str | None = None,
              nivel: float = 0.95) -> pd.DataFrame:
    """A/E por contagem com IC exato; opcionalmente A/E por valor (só estimativa pontual: com dados
    agregados não há como medir a variância dos valores)."""
    por = [por] if isinstance(por, str) else list(por)
    cols = [obs, esp, exposicao] + [c for c in (obs_valor, esp_valor) if c]
    t = df.groupby(por, observed=True, as_index=False)[cols].sum()
    t["exposicao_%"] = t[exposicao] / t[exposicao].sum()
    t["ae"] = t[obs] / t[esp]
    inf, sup = ic_poisson(t[obs], nivel)
    t["ae_inf"], t["ae_sup"] = inf / t[esp], sup / t[esp]
    if obs_valor and esp_valor:
        t["ae_valor"] = t[obs_valor] / t[esp_valor]
    return t


def mapa_ae(df: pd.DataFrame, linhas: str, colunas: str, obs: str = "obitos", esp: str = "esperados_n",
            min_obitos: int = 30) -> pd.DataFrame:
    """A/E cruzado (duas variáveis); células com menos de `min_obitos` óbitos ficam vazias
    (com 30 óbitos o IC de 95% já tem cerca de ±36%)."""
    t = df.groupby([linhas, colunas], observed=True)[[obs, esp]].sum()
    ae = (t[obs] / t[esp]).where(t[obs] >= min_obitos)
    return ae.unstack(colunas)


def grafico_ae(ax, t: pd.DataFrame, rotulo: str, titulo: str = "", col_valor: bool = False):
    """Barras de exposição (eixo direito) + A/E com IC de 95% (eixo esquerdo), no formato dos
    relatórios de estudo de experiência."""
    x = np.arange(len(t))
    ax2 = ax.twinx()
    ax2.bar(x, t["exposicao_%"], color="#d9d9d9", width=0.7, zorder=0)
    ax2.set_ylabel("% da exposição", color="#7f7f7f")
    ax2.set_ylim(0, t["exposicao_%"].max() * 3)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.errorbar(x, t["ae"], yerr=[t["ae"] - t["ae_inf"], t["ae_sup"] - t["ae"]], fmt="o", color="#1f4e79",
                capsize=2, label="A/E por contagem (IC 95%)")
    if col_valor and "ae_valor" in t:
        ax.plot(x, t["ae_valor"], "s", color="#c00000", ms=4, label="A/E por valor")
    ax.axhline(1, color="grey", lw=0.7, ls="--")
    ax.set_xticks(x, [str(v) for v in t[rotulo]], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("A/E")
    ax.set_title(titulo)
    ax.legend(fontsize=7, loc="upper right")
    return ax


def agrupar_niveis_raros(df: pd.DataFrame, var: str, peso: str = "E", minimo: float = 30.0, ordenado: bool = True,
                         desconhecidos: tuple[str, ...] = ("unknown", "desconhecid", "nd")) -> tuple[pd.Series, dict]:
    """Agrupa níveis com pouca experiência (CAS Monograph 5, seção 5.5), para que todo nível do modelo tenha
    pelo menos `minimo` de `peso` (por padrão, óbitos esperados: com 30, o IC de 95% do A/E tem cerca de ±36%).

    - Níveis "desconhecidos" raros vão para o nível de maior peso (não são uma faixa real).
    - Variável ordenada (faixas): o nível mais raro se junta ao vizinho de menor peso, até todos passarem do mínimo.
    - Variável sem ordem: níveis raros vão para "OUTROS"; se "OUTROS" continuar raro, vai para o de maior peso.
    Devolve (nova coluna, mapa nível original -> nível agrupado)."""
    x = df[var].astype(str)
    pesos = df.groupby(x)[peso].sum()
    niveis = list(pesos.index) if not ordenado else sorted(pesos.index)
    mapa = {n: n for n in niveis}
    modal = pesos.idxmax()
    desc = [n for n in niveis if any(d in n.lower() for d in desconhecidos) and pesos[n] < minimo]
    for n in desc:
        mapa[n] = modal
    reais = [n for n in niveis if n not in desc]
    if ordenado:
        grupos = [[n] for n in reais]
        while len(grupos) > 1:
            w = [pesos[g].sum() for g in grupos]
            i = int(np.argmin(w))
            if w[i] >= minimo:
                break
            j = i - 1 if i == len(grupos) - 1 else (i + 1 if i == 0 else (i - 1 if w[i - 1] <= w[i + 1] else i + 1))
            a, b = sorted((i, j))
            grupos[a] = grupos[a] + grupos[b]
            del grupos[b]
        for g in grupos:
            rotulo = g[0] if len(g) == 1 else f"{g[0]} a {g[-1].split('.')[0].strip()}"
            for n in g:
                mapa[n] = rotulo
    else:
        raros = [n for n in reais if pesos[n] < minimo]
        if raros:
            if pesos[raros].sum() >= minimo:
                for n in raros:
                    mapa[n] = "OUTROS"
            else:
                for n in raros:
                    mapa[n] = modal
    # o nível de maior peso pode ter sido renomeado ao agrupar: aponta os desconhecidos para o rótulo final
    for n in desc:
        mapa[n] = mapa[modal]
    return x.map(mapa), mapa
