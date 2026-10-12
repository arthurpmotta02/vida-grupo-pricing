"""Forma do estudo de mercado (SOA) como "priori" da taxa manual da carteira.

1. `forma_soa`: GLM de Poisson nos dados da SOA (glmmTMB), com offset dos esperados do estudo, log do capital
   médio da FAIXA de capital e seção CNAE (mapeada por julgamento, `config/prioris.yaml`). Sem salário, porque
   a carteira não tem salário: a elasticidade ao capital estimada assim inclui o que o capital carrega de renda.
   O capital médio da célula não serve: dentro de uma faixa ele varia muito (há células da faixa de US$ 50-74
   mil com média de US$ 546 mil), e esse erro de medida atenua a inclinação (de ~-0,2 para ~-0,04). O capital
   médio da faixa é o valor representativo padrão.
2. `offset_priori`: na carteira, log E + elasticidade x log(capital / ref) + log(fator relativo da seção).
   Seções sem correspondência na SOA ficam com fator 1 (sem priori: a carteira decide).
O nível não é transferido: os fatores de seção são normalizados para média 1 ponderada pelos esperados.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .glmm import ajustar_glmm


def forma_soa(soa: pd.DataFrame, cfg_prioris: dict) -> dict:
    d = soa[soa["esperados_n"] > 0].copy()
    d["cnae_secao"] = d["atividade"].astype(str).map(cfg_prioris["soa_para_cnae"])
    faixa = d.groupby("faixa_capital", observed=True)[["capital_ano", "vidas_ano"]].sum()
    cap_faixa = (faixa["capital_ano"] / faixa["vidas_ano"]).rename("cap_faixa")
    d = d.merge(cap_faixa, left_on="faixa_capital", right_index=True)
    d = d[d["cnae_secao"].notna() & (d["cap_faixa"] > 0)].copy()       # faixa "Unknown" não tem capital
    d["N"], d["E"] = d["obitos"], d["esperados_n"]
    d["log_capital_c"] = np.log(d["cap_faixa"] / cfg_prioris["capital_referencia_usd"])
    base = d.groupby("cnae_secao")["E"].sum().idxmax()
    r = ajustar_glmm(d, ["log_capital_c", "cnae_secao"], [], categoricas=["cnae_secao"], numericas=["log_capital_c"],
                     niveis_base={"cnae_secao": base})
    coef = r.fixos.set_index("nome")
    fat = {base: 1.0}
    ep = {base: 0.0}
    for nome, linha in coef.iterrows():
        if nome.startswith("cnae_secao"):
            fat[nome[len("cnae_secao"):]] = float(np.exp(linha["estimativa"]))
            ep[nome[len("cnae_secao"):]] = float(linha["ep"])
    pesos = d.groupby("cnae_secao")["E"].sum()
    media = sum(fat[s] * pesos[s] for s in fat) / pesos[list(fat)].sum()
    return {"elasticidade_capital": float(coef.loc["log_capital_c", "estimativa"]),
            "ep_elasticidade": float(coef.loc["log_capital_c", "ep"]),
            "fator_secao": {s: v / media for s, v in fat.items()}, "ep_log_secao": ep,
            "esperados_por_secao": {s: float(v) for s, v in pesos.items()},
            "capital_medio_por_faixa_usd": {str(k): float(v) for k, v in cap_faixa.items()},
            "obitos": int(d["N"].sum()), "convergiu": r.convergiu, "formula": r.formula}


def offset_priori(cel: pd.DataFrame, prioris: dict) -> pd.Series:
    """log E + forma do estudo (elasticidade ao capital e fator relativo da seção)."""
    f_sec = cel["cnae_secao"].astype(str).map(prioris["fator_secao"]).fillna(1.0)
    return (np.log(cel["E"]) + prioris["elasticidade_capital"] * cel["log_capital_c"] + np.log(f_sec)).rename("off_priori")
