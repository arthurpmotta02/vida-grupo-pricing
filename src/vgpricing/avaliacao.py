"""Avaliação e comparação de modelos em dados reservados (holdout).

Segue a monografia nº 5 da CAS (Goldburd, Khare, Tevet e Guller, 2025, cap. 7) e o framework de análise
preditiva da SOA (2024, seções 2.4 e 5):
- métricas no teste: deviance de Poisson, MSE, MAE (SOA 5.1);
- gráfico de quantis (CAS 7.2.1) e *double lift* contra o plano atual (CAS 7.2.2);
- curva de Lorenz ordenada e Gini (CAS 7.2.4; SOA 2.4.1);
- tabelas de A/M (observado / modelo) por uma e duas variáveis, com marcação de ±5% e ±10% (SOA 2.4.5) e,
  além do que a SOA faz, um intervalo exato de Poisson para separar desvio real de ruído.

Convenção para mortalidade: a "exposição" de cada linha são os óbitos esperados pela tábua (E), e o "custo" que
o modelo prevê é a relatividade mu / E. Assim as faixas têm o mesmo volume de esperados, como na CAS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .estudo import ic_poisson


def deviance_poisson(N, mu) -> float:
    return float(deviance_por_unidade(N, mu).sum())


def folds_por_grupo(grupos, k: int, semente: int) -> np.ndarray:
    """Partição 0..k-1 para validação cruzada em que todas as linhas de um grupo (ex.: estipulante) ficam no mesmo
    fold: as células de um grupo não são independentes. Usada por todo o projeto (uma implementação só)."""
    g = pd.Series(np.asarray(grupos))
    u = g.unique()
    mapa = dict(zip(np.random.default_rng(semente).permutation(u), np.arange(len(u)) % k))
    return g.map(mapa).to_numpy()


def deviance_por_unidade(N, mu, unidade=None) -> np.ndarray:
    """Contribuição de cada linha para a deviance de Poisson, somada por `unidade` (ex.: estipulante), se dada.
    Para a deviance de uma unidade, os totais da unidade são comparados: N e mu são somados antes."""
    N, mu = np.asarray(N, float), np.asarray(mu, float)
    if unidade is not None:
        g = pd.DataFrame({"u": np.asarray(unidade), "N": N, "mu": mu}).groupby("u", sort=False)[["N", "mu"]].sum()
        N, mu = g["N"].to_numpy(), g["mu"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(N > 0, N * np.log(N / mu), 0.0)
    return 2 * (t - (N - mu))


def diferenca_deviance(dev_novo: np.ndarray, dev_ref: np.ndarray, n_boot: int = 2000, semente: int = 2026) -> dict:
    """Diferença de deviance (novo - referência) com IC de 95% por bootstrap pareado sobre as unidades.
    Negativa = o novo prevê melhor; `significativa` = o IC não contém zero. Feito em blocos (memória limitada)."""
    dif = np.asarray(dev_novo, float) - np.asarray(dev_ref, float)
    rng = np.random.default_rng(semente)
    boot = np.empty(n_boot)
    bloco = max(1, int(2e7 // max(len(dif), 1)))
    for i in range(0, n_boot, bloco):
        k = min(bloco, n_boot - i)
        boot[i:i + k] = dif[rng.integers(0, len(dif), (k, len(dif)))].sum(axis=1)
    inf, sup = np.percentile(boot, [2.5, 97.5])
    return {"diferenca": float(dif.sum()), "ic95_inf": float(inf), "ic95_sup": float(sup),
            "significativa": bool(sup < 0 or inf > 0)}


def comparar_modelos(df: pd.DataFrame, previsoes: dict[str, str], referencia: str, obs: str = "N",
                     unidade: str | None = None, n_boot: int = 2000, semente: int = 2026) -> pd.DataFrame:
    """Para cada modelo: deviance no teste e diferença contra a referência, com IC por bootstrap pareado
    (por `unidade`, ex.: estipulante, quando as linhas de uma unidade não são independentes)."""
    u = df[unidade].to_numpy() if unidade else None
    devs = {n: deviance_por_unidade(df[obs], df[c], u) for n, c in previsoes.items()}
    linhas = []
    for n, d in devs.items():
        r = diferenca_deviance(d, devs[referencia], n_boot, semente)
        linhas.append({"modelo": n, "deviance": float(d.sum()), f"diferença vs {referencia}": r["diferenca"],
                       "IC95 inf": r["ic95_inf"], "IC95 sup": r["ic95_sup"], "significativa": r["significativa"]})
    return pd.DataFrame(linhas)


def metricas(df: pd.DataFrame, previsoes: dict[str, str], obs: str = "N") -> pd.DataFrame:
    """Uma linha por modelo: deviance de Poisson, MSE, MAE e A/M total (nível)."""
    linhas = []
    for nome, col in previsoes.items():
        N, mu = df[obs].to_numpy(float), df[col].to_numpy(float)
        linhas.append({"modelo": nome, "deviance": deviance_poisson(N, mu), "MSE": float(np.mean((N - mu) ** 2)),
                       "MAE": float(np.mean(np.abs(N - mu))), "A/M total": float(N.sum() / mu.sum())})
    out = pd.DataFrame(linhas)
    out["deviance vs melhor"] = out["deviance"] - out["deviance"].min()
    return out


def _faixas_iguais(ordem: np.ndarray, peso: np.ndarray, n: int) -> np.ndarray:
    """Faixa (1..n) de cada linha, depois de ordenar por `ordem`, com o mesmo peso acumulado por faixa."""
    idx = np.argsort(ordem, kind="mergesort")
    acum = np.cumsum(peso[idx]) / peso.sum()
    faixa = np.empty(len(ordem), int)
    faixa[idx] = np.minimum((acum * n - 1e-12).astype(int) + 1, n)
    return faixa


def quantis(df: pd.DataFrame, pred: str, obs: str = "N", base: str = "E", n: int = 10) -> pd.DataFrame:
    """Gráfico de quantis (CAS 7.2.1): ordena pela relatividade prevista, faixas de igual esperado pela tábua;
    previsto e observado por faixa, divididos pela média geral do modelo."""
    E = df[base].to_numpy(float)
    rel = df[pred].to_numpy(float) / E
    f = _faixas_iguais(rel, E, n)
    g = pd.DataFrame({"faixa": f, "E": E, "pred": df[pred].to_numpy(float), "obs": df[obs].to_numpy(float)}) \
        .groupby("faixa").sum()
    media = g["pred"].sum() / g["E"].sum()
    g["previsto"] = g["pred"] / g["E"] / media
    g["observado"] = g["obs"] / g["E"] / media
    inf, sup = ic_poisson(g["obs"])
    g["obs_inf"], g["obs_sup"] = inf / g["E"] / media, sup / g["E"] / media
    return g


def double_lift(df: pd.DataFrame, pred_a: str, pred_b: str, obs: str = "N", base: str = "E", n: int = 10) -> pd.DataFrame:
    """Double lift (CAS 7.2.2): ordena pela razão A/B; em cada faixa, A, B e o observado divididos pelas
    respectivas médias gerais. Vence o modelo cuja linha acompanha o observado."""
    E = df[base].to_numpy(float)
    a, b, o = (df[c].to_numpy(float) for c in (pred_a, pred_b, obs))
    f = _faixas_iguais(a / b, E, n)
    g = pd.DataFrame({"faixa": f, "a": a, "b": b, "obs": o}).groupby("faixa").sum()
    # razão de cada faixa sobre a média geral, com o mesmo denominador (esperados da faixa sob B)
    tot = {c: g[c].sum() for c in ("a", "b", "obs")}
    base_b = g["b"] / tot["b"]
    g[pred_a] = (g["a"] / tot["a"]) / base_b
    g[pred_b] = 1.0
    g["observado"] = (g["obs"] / tot["obs"]) / base_b
    inf, sup = ic_poisson(g["obs"])
    g["obs_inf"], g["obs_sup"] = (inf / tot["obs"]) / base_b, (sup / tot["obs"]) / base_b
    return g[[pred_a, pred_b, "observado", "obs_inf", "obs_sup"]]


def lorenz_gini(df: pd.DataFrame, pred: str, obs: str = "N", base: str = "E") -> tuple[pd.DataFrame, float]:
    """Curva de Lorenz ordenada (CAS 7.2.4; SOA 2.4.1): ordena pela relatividade prevista (melhores riscos
    primeiro); x = parcela acumulada dos esperados pela tábua, y = parcela acumulada dos óbitos observados.
    Gini = 2 x área entre a diagonal e a curva."""
    E = df[base].to_numpy(float)
    ordem = np.argsort(df[pred].to_numpy(float) / E, kind="mergesort")
    x = np.r_[0, np.cumsum(E[ordem]) / E.sum()]
    y = np.r_[0, np.cumsum(df[obs].to_numpy(float)[ordem]) / df[obs].sum()]
    area = float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2))
    return pd.DataFrame({"esperados_acum": x, "obitos_acum": y}), 2 * (0.5 - area)


def tabela_am(df: pd.DataFrame, por: str | list[str], pred: str, obs: str = "N", min_obitos: int = 0) -> pd.DataFrame:
    """A/M (observado / modelo) por uma ou duas variáveis (SOA 2.4.5). Marca |A/M − 1| > 5% e > 10% e se o
    intervalo exato de Poisson de 95% exclui 100% (desvio que o ruído não explica)."""
    por = [por] if isinstance(por, str) else list(por)
    t = df.groupby(por, observed=True, as_index=False)[[obs, pred]].sum()
    t = t[t[obs] >= min_obitos]
    t["A/M"] = t[obs] / t[pred]
    inf, sup = ic_poisson(t[obs])
    t["A/M inf"], t["A/M sup"] = inf / t[pred], sup / t[pred]
    t["fora de ±5%"] = (t["A/M"] - 1).abs() > 0.05
    t["fora de ±10%"] = (t["A/M"] - 1).abs() > 0.10
    t["IC exclui 100%"] = (t["A/M inf"] > 1) | (t["A/M sup"] < 1)
    return t.reset_index(drop=True)


# ----------------------------------------------------------------------------- gráficos

def grafico_quantis(ax, q: pd.DataFrame, titulo: str):
    x = q.index.to_numpy()
    ax.plot(x, q["previsto"], "o-", color="#1f4e79", label="previsto")
    ax.errorbar(x, q["observado"], yerr=[q["observado"] - q["obs_inf"], q["obs_sup"] - q["observado"]], fmt="s--",
                color="#c00000", capsize=2, ms=4, label="observado (IC 95%)")
    ax.axhline(1, color="grey", lw=0.6, ls=":")
    ax.set_xlabel("faixa (ordenada pela relatividade prevista)"); ax.set_ylabel("relativo à média")
    ax.set_title(titulo); ax.set_xticks(x); ax.legend()


def grafico_double_lift(ax, d: pd.DataFrame, nome_a: str, nome_b: str, titulo: str = ""):
    x = d.index.to_numpy()
    ax.plot(x, d.iloc[:, 0], "o-", color="#1f4e79", label=nome_a)
    ax.plot(x, d.iloc[:, 1], "-", color="#7f7f7f", label=nome_b)
    ax.errorbar(x, d["observado"], yerr=[d["observado"] - d["obs_inf"], d["obs_sup"] - d["observado"]], fmt="s--",
                color="#c00000", capsize=2, ms=4, label="observado (IC 95%)")
    ax.set_xlabel(f"faixa da razão {nome_a} / {nome_b}"); ax.set_ylabel(f"relativo a {nome_b}")
    ax.set_title(titulo or f"Double lift: {nome_a} × {nome_b}"); ax.set_xticks(x); ax.legend()


def grafico_lorenz(ax, curvas: dict[str, tuple[pd.DataFrame, float]], titulo: str = "Curva de Lorenz ordenada"):
    for nome, (c, g) in curvas.items():
        ax.plot(c["esperados_acum"], c["obitos_acum"], label=f"{nome} (Gini {g:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=0.8)
    ax.set_xlabel("parcela acumulada dos esperados (tábua)"); ax.set_ylabel("parcela acumulada dos óbitos")
    ax.set_title(titulo); ax.legend()
