"""Backtest fora do tempo da renovação.

Para cada grupo, o "ano-alvo" é o último ano-apólice completo que terminou há pelo menos
`meses_maturacao_alvo` (sinistros já avisados). A renovação é simulada no início do ano-alvo usando SÓ
o que se sabia na véspera: exposição e sinistros avisados até ali, IBNR estimado naquela data, taxa
manual estimada com dados anteriores e parâmetro de credibilidade estimado na carteira da data mais
antiga. Depois compara-se o previsto com o que aconteceu no ano-alvo.

É o mesmo teste que seguradoras usam para calibrar fórmulas de credibilidade (experiência dos primeiros
anos prevendo os seguintes), e roda igual na base real.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .exposicao import anos_apolice
from .renovacao import (buhlmann_straub, experiencia_renovacao, fatores_experiencia, niveis_pooling,
                        padroes_e_apriori)
from .tabua import Tabua

METODOS = {"manual": "Só taxa manual (Z = 0)", "mercado": "Credibilidade de mercado (raiz, vidas-ano)",
           "bs": "Bühlmann-Straub (n efetivo)", "experiencia_pura": "Só experiência (Z = 1)"}


def anos_alvo(estipulantes: pd.DataFrame, data_referencia, meses_maturacao: int) -> pd.DataFrame:
    ref = pd.Timestamp(data_referencia)
    py = anos_apolice(estipulantes, ref)
    ok = py[py["ano_completo"] & (py["fim_ano"] <= ref - pd.DateOffset(months=meses_maturacao))]
    alvo = ok.sort_values("ano_apolice").groupby("id_estipulante").tail(1)
    return alvo[alvo["ano_apolice"] >= 2].reset_index(drop=True)          # precisa de 1 ano de histórico


def executar(dados: dict, tabua: Tabua, tabela_manual: dict, cfg_renov: dict, data_referencia) -> dict:
    v, s, e = dados["vidas"], dados["sinistros"], dados["estipulantes"]
    ref = pd.Timestamp(data_referencia)
    alvo = anos_alvo(e, ref, cfg_renov["backtest"]["meses_maturacao_alvo"])
    d0 = alvo["inicio_ano"].min() - pd.Timedelta(days=1)

    # parâmetro de credibilidade: carteira inteira na data mais antiga (nenhuma informação futura)
    P0 = niveis_pooling(v, d0, cfg_renov["pooling"])
    pn0, pv0, _ = padroes_e_apriori(s, P0, None, d0)
    xp0 = experiencia_renovacao(v, s, e, tabua, tabela_manual, d0, P0, pn0, pv0, (1.0, 1.0))
    xp0["AE_v"] = np.where(xp0["V_manual"] > 0, xp0["A_v"] / xp0["V_manual"], np.nan)
    bs = buhlmann_straub(xp0, "id_estipulante", "AE_v", "n_ef")

    historicos, pools = [], []
    for d, grupo in alvo.groupby("inicio_ano"):
        aval = d - pd.Timedelta(days=1)
        ids = set(grupo["id_estipulante"])
        P_all = niveis_pooling(v, aval, cfg_renov["pooling"])
        pn, pv, _ = padroes_e_apriori(s, P_all, None, aval)
        vg, sg = v[v["id_estipulante"].isin(ids)], s[s["id_estipulante"].isin(ids)]
        P = P_all.reindex(sorted(ids))
        xp = experiencia_renovacao(vg, sg, e, tabua, tabela_manual, aval, P, pn, pv, (1.0, 1.0))
        historicos.append(xp.assign(data_renovacao=d))
        pools.append(P)
    hist = pd.concat(historicos, ignore_index=True)
    P_alvo = pd.concat(pools)
    fat = fatores_experiencia(hist, cfg_renov["credibilidade"], k_bs=bs["k"], anos_max=cfg_renov["anos_experiencia_max"])

    # o que aconteceu no ano-alvo (avaliado na data de referência, com o pooling da data de renovação)
    pn_r, pv_r, _ = padroes_e_apriori(s, P_alvo, None, ref)
    real = experiencia_renovacao(v, s, e, tabua, tabela_manual, ref, P_alvo, pn_r, pv_r, (1.0, 1.0))
    real = real.merge(alvo[["id_estipulante", "ano_apolice"]], on=["id_estipulante", "ano_apolice"])
    pr = dados["premios"].merge(alvo[["id_estipulante", "inicio_ano"]], left_on=["id_estipulante", "inicio_periodo"],
                                right_on=["id_estipulante", "inicio_ano"])
    real = real.merge(pr[["id_estipulante", "premio_emitido"]], on="id_estipulante", how="left")
    res = real.merge(fat, on="id_estipulante", suffixes=("", "_hist"))

    c = cfg_renov["reajuste"]["carregamento_total"]
    linhas, devs = [], {}
    for m, nome in METODOS.items():
        prev_lim = res["V_manual"] * res[f"fator_{m}"]
        prev_n = res["E_manual"] * res[f"fator_{m}"]
        premio = (prev_lim + res["V_excesso"]) / (1 - c)
        devs[nome] = _deviance_por_grupo(res["N"], prev_n)
        linhas.append(_metricas(nome, res, prev_lim, prev_n, premio))
    if res["premio_emitido"].notna().all():
        linhas.append(_metricas("Regra contratual 75/85% (prêmio efetivamente cobrado)", res, None, None,
                                res["premio_emitido"]))
    resumo = pd.DataFrame(linhas)
    comp = comparar_com_referencia(devs, METODOS["manual"], semente=cfg_renov["backtest"].get("semente", 2026))
    return {"resumo": resumo, "comparacao_deviance": comp, "por_grupo": res, "bs": bs, "alvo": alvo, "data_k": d0,
            "pooling_alvo": P_alvo}


def _deviance_por_grupo(N, mu) -> np.ndarray:
    N, mu = np.asarray(N, float), np.asarray(mu, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(N > 0, N * np.log(N / mu), 0.0)
    return 2 * (t - (N - mu))


def comparar_com_referencia(devs: dict, referencia: str, n_boot: int = 2000, semente: int = 2026) -> pd.DataFrame:
    """Diferença de deviance (método - referência) com IC de 95% por bootstrap pareado por grupo.
    Diferença negativa = método melhor que a referência."""
    rng = np.random.default_rng(semente)
    ref = devs[referencia]
    n = len(ref)
    idx = rng.integers(0, n, (n_boot, n))
    linhas = []
    for nome, d in devs.items():
        dif = d - ref
        boot = dif[idx].sum(axis=1)
        linhas.append({"método": nome, "deviance": float(d.sum()), "diferença vs só manual": float(dif.sum()),
                       "IC95 inf": float(np.percentile(boot, 2.5)), "IC95 sup": float(np.percentile(boot, 97.5))})
    return pd.DataFrame(linhas)


def _metricas(nome, res, prev_lim, prev_n, premio) -> dict:
    out = {"método": nome}
    if prev_lim is not None:
        out["deviance (contagem)"] = float(_deviance_por_grupo(res["N"], prev_n).sum())
        out["erro relativo (valor limitado)"] = float(np.sqrt(((res["V"] - prev_lim) ** 2).sum()) / res["V"].sum())
        out["previsto/observado (valor limitado)"] = float(prev_lim.sum() / res["V"].sum())
    sp = res["V_total"] / premio
    out["S/P carteira"] = float(res["V_total"].sum() / premio.sum())
    out["dispersão da S/P entre grupos"] = float(np.sqrt(np.average((sp - out["S/P carteira"]) ** 2, weights=premio)))
    return out
