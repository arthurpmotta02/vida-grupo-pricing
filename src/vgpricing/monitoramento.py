"""Monitoramento da taxa manual em produção.

- Série mensal de A/E da carteira contra a TAXA MANUAL (não contra a tábua), com os meses recentes completados
  pelo padrão de aviso (IBNR): A ajustado = avisados / proporção já avisada na idade do mês.
- Limites de controle para o A/E dado o volume esperado (funnel plot, Spiegelhalter, 2005, Statistics in
  Medicine). Carteira: quantis exatos de Poisson(E). Grupos: em vida em grupo cada grupo tem mortalidade própria,
  então os limites incluem a variância entre grupos (sobredispersão; Poisson-Gama = Binomial Negativa com média E e
  variância E + tau2 E^2), coerente com a credibilidade da renovação.
- Os meses mais recentes (imaturos) ficam fora do A/E móvel.
- Gatilhos objetivos de revisão; uma revisão segue o protocolo champion/challenger (README).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

from .exposicao import exposicao_detalhada
from .ibnr import fracao_nao_avisada, padrao_aviso
from .manual import multiplicador_manual
from .tabua import Tabua


def limites_poisson(E, nivel: float, tau2: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Limites do A/E sob a hipótese de taxa correta: quantis exatos de Poisson(E) ou, com variância entre grupos
    tau2 > 0, da Binomial Negativa com média E e variância E + tau2 E^2; divididos por E."""
    E = np.asarray(E, float)
    a = (1 - nivel) / 2
    if tau2 <= 0:
        return poisson.ppf(a, E) / E, poisson.ppf(1 - a, E) / E
    n = 1 / tau2
    p = n / (n + E)
    return nbinom.ppf(a, n, p) / E, nbinom.ppf(1 - a, n, p) / E


def variancia_entre_grupos(N, E) -> float:
    """Estimador de momentos de tau2 (variância relativa entre grupos): sum((N - E)^2 - N) / sum(E^2), >= 0."""
    N, E = np.asarray(N, float), np.asarray(E, float)
    return float(max(0.0, (((N - E) ** 2 - N).sum()) / (E ** 2).sum()))


def serie_mensal(vidas: pd.DataFrame, sinistros: pd.DataFrame, estipulantes: pd.DataFrame, tabua: Tabua,
                 tabela: dict, data_referencia, meses: int = 24, dias_por_ano: float = 365.25) -> pd.DataFrame:
    """Esperado pela taxa manual e observado (avisado e ajustado por IBNR), por estipulante e mês de ocorrência."""
    ref = pd.Timestamp(data_referencia)
    est = estipulantes[["id_estipulante", "cnae_secao", "cnae_divisao"]]
    linhas = []
    for m in pd.period_range(ref.to_period("M") - (meses - 1), ref.to_period("M"), freq="M"):
        ini, fim = m.start_time, min(m.end_time.normalize(), ref)
        v = vidas[(vidas["data_inicio"] <= fim) & (vidas["data_fim"].isna() | (vidas["data_fim"] >= ini))]
        det = exposicao_detalhada(v, estipulantes, tabua, fim, dias_por_ano, data_inicio=ini).merge(est, on="id_estipulante")
        det["E_manual"] = det["obitos_esperados"] * multiplicador_manual(det, tabela)[0]
        g = det.groupby("id_estipulante", as_index=False)["E_manual"].sum()
        linhas.append(g.assign(mes=m))
    esp = pd.concat(linhas, ignore_index=True)
    s = sinistros[(sinistros["data_aviso"] <= ref) & (sinistros["data_ocorrencia"] <= ref)].copy()
    s["mes"] = s["data_ocorrencia"].dt.to_period("M")
    obs = s.groupby(["id_estipulante", "mes"], as_index=False).size().rename(columns={"size": "N"})
    d = esp.merge(obs, on=["id_estipulante", "mes"], how="left").fillna({"N": 0})
    idade = (ref.year - d["mes"].dt.year) * 12 + ref.month - d["mes"].dt.month + 1
    d["proporcao_avisada"] = 1 - fracao_nao_avisada(idade.to_numpy(), padrao_aviso(sinistros, ref))
    d["N_ajustado"] = d["N"] / d["proporcao_avisada"]
    return d


def carteira_mensal(serie: pd.DataFrame, niveis=(0.95, 0.998), janela: int = 12, meses_imaturos: int = 3) -> pd.DataFrame:
    """A/E mensal e A/E móvel de `janela` meses da carteira (sem os meses imaturos), com limites de controle."""
    c = serie.groupby("mes")[["E_manual", "N", "N_ajustado"]].sum()
    c["A/E"] = c["N_ajustado"] / c["E_manual"]
    c["imaturo"] = np.arange(len(c)) >= len(c) - meses_imaturos
    E_m, N_m = c["E_manual"].where(~c["imaturo"]), c["N_ajustado"].where(~c["imaturo"])
    E_mov, N_mov = E_m.rolling(janela).sum(), N_m.rolling(janela).sum()
    c[f"A/E {janela}m"] = N_mov / E_mov
    for n in niveis:
        inf, sup = limites_poisson(E_mov.fillna(1.0), n)
        c[f"lim inf {n:.1%}"], c[f"lim sup {n:.1%}"] = np.where(E_mov.notna(), inf, np.nan), np.where(E_mov.notna(), sup, np.nan)
    return c


def funil(serie: pd.DataFrame, niveis=(0.95, 0.998), janela: int = 12, meses_imaturos: int = 3) -> pd.DataFrame:
    """Funnel plot por estipulante nos `janela` meses maduros mais recentes: A/E contra o esperado, com limites que
    incluem a variância entre grupos (estimada nos próprios dados)."""
    fim = serie["mes"].max() - meses_imaturos
    ult = serie[(serie["mes"] > fim - janela) & (serie["mes"] <= fim)]
    g = ult.groupby("id_estipulante")[["E_manual", "N_ajustado"]].sum()
    g = g[g["E_manual"] > 0]
    g["A/E"] = g["N_ajustado"] / g["E_manual"]
    tau2 = variancia_entre_grupos(g["N_ajustado"], g["E_manual"])
    for n in niveis:
        inf, sup = limites_poisson(g["E_manual"], n, tau2)
        g[f"lim inf {n:.1%}"], g[f"lim sup {n:.1%}"] = inf, sup
        g[f"acima {n:.1%}"] = g["A/E"] > sup
    g.attrs["tau2"] = tau2
    return g


def gatilhos(carteira: pd.DataFrame, grupos: pd.DataFrame, tabela_meta: dict, data_referencia, cfg: dict) -> pd.DataFrame:
    """Gatilhos de revisão. Cada um diz o que fazer, não só que algo mudou."""
    j, sup_n = cfg["janela_meses"], max(cfg["niveis"])
    ult = carteira.dropna(subset=[f"A/E {j}m"]).iloc[-1]
    fora = not (ult[f"lim inf {sup_n:.1%}"] <= ult[f"A/E {j}m"] <= ult[f"lim sup {sup_n:.1%}"])
    frac = float(grupos[f"acima {sup_n:.1%}"].mean())
    idade_meses = (pd.Timestamp(data_referencia) - pd.Timestamp(tabela_meta["corte_experiencia"])).days / 30.44
    g = cfg["gatilhos"]
    return pd.DataFrame([
        {"gatilho": f"A/E {j}m da carteira fora do limite de {sup_n:.1%}", "disparou": bool(fora),
         "valor": f"{ult[f'A/E {j}m']:.3f} (limites {ult[f'lim inf {sup_n:.1%}']:.3f} a {ult[f'lim sup {sup_n:.1%}']:.3f})",
         "ação": "recalibrar a taxa manual (champion/challenger)"},
        {"gatilho": f"grupos acima do limite de {sup_n:.1%}", "disparou": frac > g["grupos_acima_998_maximo"],
         "valor": f"{frac:.2%} dos grupos (esperado: {(1 - sup_n) / 2:.2%}; limites com variância entre grupos "
                  f"tau2 = {grupos.attrs.get('tau2', 0):.3f})",
         "ação": "investigar subscrição e cadastro dos grupos"},
        {"gatilho": "idade da experiência que calibrou a tabela", "disparou": idade_meses > g["idade_maxima_experiencia_meses"],
         "valor": f"{idade_meses:.0f} meses desde o corte ({tabela_meta['corte_experiencia']})",
         "ação": "ciclo anual de recalibração"},
    ])
