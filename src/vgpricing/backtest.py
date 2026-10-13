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

from .avaliacao import deviance_por_unidade, diferenca_deviance
from .exposicao import anos_apolice
from .renovacao import (buhlmann_straub, experiencia_renovacao, fatores_experiencia, niveis_pooling,
                        padroes_e_apriori)
from .tabua import Tabua

METODOS = {"manual": "Só taxa manual (Z = 0)", "mercado": "Credibilidade de mercado (raiz, vidas-ano)",
           "bs": "Bühlmann-Straub (n efetivo)", "experiencia_pura": "Só experiência (Z = 1)"}


def anos_alvo(estipulantes: pd.DataFrame, data_referencia, meses_maturacao: int, todos: bool = False,
              ano_inicio: int | None = None) -> pd.DataFrame:
    """Anos-apólice completos e maduros com pelo menos 1 ano de histórico. `todos=False`: só o último de cada
    grupo; `todos=True`: todos (vários anos-alvo). `ano_inicio`: só os que começam nesse ano-calendário."""
    ref = pd.Timestamp(data_referencia)
    py = anos_apolice(estipulantes, ref)
    ok = py[py["ano_completo"] & (py["fim_ano"] <= ref - pd.DateOffset(months=meses_maturacao)) & (py["ano_apolice"] >= 2)]
    if ano_inicio is not None:
        ok = ok[ok["inicio_ano"].dt.year == ano_inicio]
    alvo = ok if todos or ano_inicio is not None else ok.sort_values("ano_apolice").groupby("id_estipulante").tail(1)
    return alvo.reset_index(drop=True)


def executar(dados: dict, tabua: Tabua, tabela_manual: dict, cfg_renov: dict, data_referencia,
             ano_inicio: int | None = None) -> dict:
    """Backtest de um conjunto de anos-alvo (o último de cada grupo ou, com `ano_inicio`, os que começam nesse
    ano). A taxa manual passada deve ter sido estimada só com dados anteriores a esses anos-alvo."""
    v, s, e = dados["vidas"], dados["sinistros"], dados["estipulantes"]
    ref = pd.Timestamp(data_referencia)
    alvo = anos_alvo(e, ref, cfg_renov["backtest"]["meses_maturacao_alvo"], ano_inicio=ano_inicio)
    d0 = alvo["inicio_ano"].min() - pd.Timedelta(days=1)

    # parâmetro de credibilidade: carteira inteira na data mais antiga (nenhuma informação futura)
    P0 = niveis_pooling(v, d0, cfg_renov["pooling"])
    pn0, pv0, _ = padroes_e_apriori(s, P0, None, d0)
    xp0 = experiencia_renovacao(v, s, e, tabua, tabela_manual, d0, P0, pn0, pv0, (1.0, 1.0))
    xp0["AE_v"] = np.where(xp0["V_manual"] > 0, xp0["A_v"] / xp0["V_manual"], np.nan)
    bs = buhlmann_straub(xp0, "id_estipulante", "ano_apolice", "AE_v", "n_ef")

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
        devs[nome] = deviance_por_unidade(res["N"], prev_n)
        linhas.append(_metricas(nome, res, prev_lim, prev_n, premio))
    if res["premio_emitido"].notna().all():
        linhas.append(_metricas("Regra contratual 75/85% (prêmio efetivamente cobrado)", res, None, None,
                                res["premio_emitido"]))
    resumo = pd.DataFrame(linhas)
    comp = comparar_com_referencia(devs, METODOS["manual"], semente=cfg_renov["backtest"].get("semente", 2026))
    return {"resumo": resumo, "comparacao_deviance": comp, "por_grupo": res, "bs": bs, "alvo": alvo, "data_k": d0,
            "pooling_alvo": P_alvo}


def comparar_com_referencia(devs: dict, referencia: str, n_boot: int = 2000, semente: int = 2026) -> pd.DataFrame:
    """Diferença de deviance (método - referência) com IC de 95% por bootstrap pareado por grupo
    (implementação única em `avaliacao.diferenca_deviance`). Diferença negativa = método melhor."""
    linhas = []
    for nome, d in devs.items():
        r = diferenca_deviance(d, devs[referencia], n_boot, semente)
        linhas.append({"método": nome, "deviance": float(d.sum()), "diferença vs só manual": r["diferenca"],
                       "IC95 inf": r["ic95_inf"], "IC95 sup": r["ic95_sup"]})
    return pd.DataFrame(linhas)


def _metricas(nome, res, prev_lim, prev_n, premio) -> dict:
    out = {"método": nome}
    if prev_lim is not None:
        out["deviance (contagem)"] = float(deviance_por_unidade(res["N"], prev_n).sum())
        out["erro relativo (valor limitado)"] = float(np.sqrt(((res["V"] - prev_lim) ** 2).sum()) / res["V"].sum())
        out["previsto/observado (valor limitado)"] = float(prev_lim.sum() / res["V"].sum())
    sp = res["V_total"] / premio
    out["S/P carteira"] = float(res["V_total"].sum() / premio.sum())
    out["dispersão da S/P entre grupos"] = float(np.sqrt(np.average((sp - out["S/P carteira"]) ** 2, weights=premio)))
    return out


def executar_varios_anos(dados: dict, tabua: Tabua, ajustar_tabela_ate, cfg_renov: dict, data_referencia,
                         anos: list[int]) -> dict:
    """Empilha backtests de vários anos-alvo. Para cada ano Y, `ajustar_tabela_ate(data)` estima a taxa manual com
    a experiência até `data` (início de Y menos a maturação), e o k da credibilidade vem da carteira no início
    de Y: nenhum ano usa informação posterior a ele. Mais anos-alvo = mais poder para separar os métodos."""
    partes, ks = [], {}
    meses = cfg_renov["backtest"]["meses_maturacao_alvo"]
    for ano in anos:
        tabela = ajustar_tabela_ate(pd.Timestamp(f"{ano}-01-01") - pd.DateOffset(months=meses))
        r = executar(dados, tabua, tabela, cfg_renov, data_referencia, ano_inicio=ano)
        partes.append(r["por_grupo"].assign(ano_alvo=ano))
        ks[ano] = r["bs"]["k"]
    res = pd.concat(partes, ignore_index=True)
    c = cfg_renov["reajuste"]["carregamento_total"]
    linhas, devs = [], {}
    for m, nome in METODOS.items():
        prev_lim, prev_n = res["V_manual"] * res[f"fator_{m}"], res["E_manual"] * res[f"fator_{m}"]
        devs[nome] = deviance_por_unidade(res["N"], prev_n)
        linhas.append(_metricas(nome, res, prev_lim, prev_n, (prev_lim + res["V_excesso"]) / (1 - c)))
    if res["premio_emitido"].notna().all():
        linhas.append(_metricas("Regra contratual 75/85% (prêmio efetivamente cobrado)", res, None, None,
                                res["premio_emitido"]))
    comp = comparar_com_referencia(devs, METODOS["manual"], semente=cfg_renov["backtest"].get("semente", 2026))
    return {"resumo": pd.DataFrame(linhas), "comparacao_deviance": comp, "por_grupo": res, "k_por_ano": ks}


def credibilidade_mercado(anual: pd.DataFrame, alvos: list[int], janela: int, anos_excluidos_historico: tuple = (),
                          anos_alvo_excluidos: tuple = (), n_boot: int = 2000, semente: int = 2026) -> dict:
    """Teste de credibilidade fora do tempo em dados reais agregados (ex.: sinistralidade por seguradora no SES).

    `anual`: colunas unidade, ano, x (razão observada), w (peso, ex.: prêmio). Para cada ano-alvo, usa só os
    `janela` anos anteriores (sem `anos_excluidos_historico`) e prevê x de cada unidade por: média da carteira,
    experiência própria e Bühlmann-Straub (actuar). Erro quadrático ponderado por w; IC da diferença contra a
    experiência própria por bootstrap sobre unidades (pareado). Anos em `anos_alvo_excluidos` (ex.: anos de
    pandemia, imprevisíveis por qualquer método) ficam fora da pontuação."""
    from .renovacao import buhlmann_straub
    linhas = []
    for alvo in alvos:
        h = anual[(anual["ano"] < alvo) & (anual["ano"] >= alvo - janela) & ~anual["ano"].isin(list(anos_excluidos_historico))]
        if h["ano"].nunique() < 2:
            continue
        bs = buhlmann_straub(h, "unidade", "ano", "x", "w")
        pg = bs["por_grupo"]
        t = anual[anual["ano"] == alvo].set_index("unidade")
        t = t[t.index.isin(pg.index)]
        Z, media = pg.loc[t.index, "Z"], pg.loc[t.index, "media"]
        prev = {"média da carteira": pd.Series(float(np.average(h["x"], weights=h["w"])), index=t.index),
                "experiência própria": media,
                "Bühlmann-Straub": Z * media + (1 - Z) * bs["media_carteira"]}
        for nome, p in prev.items():
            linhas.append(pd.DataFrame({"ano_alvo": alvo, "metodo": nome, "unidade": t.index, "w": t["w"].to_numpy(),
                                        "erro2": (t["x"] - p).to_numpy() ** 2, "Z": Z.to_numpy(), "k": bs["k"]}))
    d = pd.concat(linhas, ignore_index=True)
    pont = d[~d["ano_alvo"].isin(list(anos_alvo_excluidos))]
    resumo = pont.groupby("metodo").apply(lambda g: np.average(g["erro2"], weights=g["w"]), include_groups=False) \
        .rename("erro quadrático ponderado").to_frame()
    # bootstrap pareado por (ano-alvo, unidade): contribuição ponderada de cada uma
    ref = pont[pont["metodo"] == "experiência própria"].set_index(["ano_alvo", "unidade"])
    rng = np.random.default_rng(semente)
    idx = rng.integers(0, len(ref), (n_boot, len(ref)))
    for m in resumo.index:
        cur = pont[pont["metodo"] == m].set_index(["ano_alvo", "unidade"]).loc[ref.index]
        dif = (cur["erro2"] - ref["erro2"]).to_numpy() * ref["w"].to_numpy() / ref["w"].sum()
        boot = dif[idx].sum(axis=1)
        resumo.loc[m, "dif. vs experiência própria"] = dif.sum()
        resumo.loc[m, "IC95 inf"], resumo.loc[m, "IC95 sup"] = np.percentile(boot, [2.5, 97.5])
    return {"resumo": resumo.reset_index(), "detalhe": d}
