"""Etapa 3: renovação (experience rating).

Fluxo para cada grupo, numa data de avaliação:
1. Pooling: nível de pooling P do grupo = max(piso, múltiplo x capital médio das vidas ativas),
   limitado à retenção. Sinistros acima de P não entram na experiência do grupo; o excesso esperado
   é cobrado como encargo de pooling (prática formal de pooling em vida em grupo).
2. Experiência por ano-apólice contra a TAXA MANUAL (não contra a tábua), em valor limitado a P:
   A/E = (sinistros limitados avisados + IBNR) / esperado limitado pela manual.
3. Credibilidade Z: flutuação limitada (prática de mercado) ou Bühlmann-Straub (actuar::cm, no R).
4. Fator de experiência = Z x A/E + (1 - Z) x 1 (o complemento é a própria taxa manual).
5. Taxa de renovação = [esperado limitado na manual sobre o censo atual x fator + encargo de
   pooling] / capital; comercial = pura / (1 - carregamento); reajuste = comercial / vigente - 1,
   com piso e teto.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .exposicao import exposicao_detalhada, idade_completa
from .ibnr import ibnr_por_periodo, padrao_aviso
from .manual import multiplicador_manual
from .tabua import Tabua


# --------------------------------------------------------------------------- pooling

def niveis_pooling(vidas: pd.DataFrame, data_avaliacao, cfg_pool: dict) -> pd.Series:
    """Nível de pooling por estipulante, a partir das vidas ativas na data de avaliação."""
    d = pd.Timestamp(data_avaliacao)
    ativas = vidas[(vidas["data_inicio"] <= d) & (vidas["data_fim"].isna() | (vidas["data_fim"] >= d))]
    medio = ativas.groupby("id_estipulante")["capital"].mean()
    nivel = np.maximum(cfg_pool["piso"], cfg_pool["multiplo_capital_medio"] * medio)
    if cfg_pool.get("limite_retencao"):
        nivel = np.minimum(nivel, cfg_pool["limite_retencao"])
    return nivel.rename("nivel_pooling")


# --------------------------------------------------------------------------- experiência

def experiencia_renovacao(vidas: pd.DataFrame, sinistros: pd.DataFrame, estipulantes: pd.DataFrame, tabua: Tabua,
                          tabela_manual: dict, data_avaliacao, nivel_pooling: pd.Series,
                          padrao_n: pd.DataFrame, padrao_v: pd.DataFrame, ae_apriori: tuple[float, float],
                          dias_por_ano: float = 365.25) -> pd.DataFrame:
    """Experiência por estipulante x ano-apólice até a data de avaliação, contra a taxa manual.

    ae_apriori: (A/E em contagem, A/E em valor limitado) da carteira, usados como a priori do IBNR.
    Colunas principais: E_manual (óbitos), V_manual (valor limitado), V_excesso (valor acima do pooling),
    n_ef (número efetivo do valor limitado), N e V (avisados, limitados), IBNR_n, IBNR_v, A_n, A_v."""
    aval = pd.Timestamp(data_avaliacao)
    det = exposicao_detalhada(vidas, estipulantes, tabua, aval, dias_por_ano)
    det = det.merge(estipulantes[["id_estipulante", "cnae_secao", "cnae_divisao"]], on="id_estipulante", how="left")
    mult, _ = multiplicador_manual(det, tabela_manual)
    P = det["id_estipulante"].map(nivel_pooling).fillna(np.inf).to_numpy()
    C = det["capital"].to_numpy(float)
    em = det["qx"].to_numpy() * det["exposicao"].to_numpy() * mult
    det["E_manual"] = em
    det["V_manual"] = em * np.minimum(C, P)
    det["V_excesso"] = em * np.maximum(C - P, 0.0)
    det["c2"] = em * np.minimum(C, P) ** 2
    chave = ["id_estipulante", "ano_apolice"]
    xp = det.groupby(chave, as_index=False).agg(
        inicio_ano=("inicio_ano", "first"), fim_ano=("fim_ano", "first"), ano_completo=("ano_completo", "first"),
        vidas_ano=("exposicao", "sum"), E_tabua=("obitos_esperados", "sum"), E_manual=("E_manual", "sum"),
        V_manual=("V_manual", "sum"), V_excesso=("V_excesso", "sum"), c2=("c2", "sum"))
    xp["n_ef"] = np.where(xp["c2"] > 0, xp["V_manual"] ** 2 / xp["c2"], 0.0)

    s = sinistros[(sinistros["data_aviso"] <= aval) & (sinistros["data_ocorrencia"] <= aval)].copy()
    s["valor"] = s["valor_indenizado"].fillna(0) + s["valor_reserva"].fillna(0)
    s["valor_lim"] = np.minimum(s["valor"], s["id_estipulante"].map(nivel_pooling).fillna(np.inf))
    m = s.merge(xp[chave + ["inicio_ano", "fim_ano"]], on="id_estipulante")
    m = m[(m["data_ocorrencia"] >= m["inicio_ano"]) & (m["data_ocorrencia"] <= m["fim_ano"])]
    obs = m.groupby(chave, as_index=False).agg(N=("id_sinistro", "count"), V=("valor_lim", "sum"),
                                               V_total=("valor", "sum"))
    xp = xp.merge(obs, on=chave, how="left").fillna({"N": 0, "V": 0.0, "V_total": 0.0})
    xp["E_apriori_n"] = xp["E_manual"] * ae_apriori[0]
    xp["V_apriori"] = xp["V_manual"] * ae_apriori[1]
    xp["IBNR_n"] = ibnr_por_periodo(xp, aval, padrao_n, "E_apriori_n")
    xp["IBNR_v"] = ibnr_por_periodo(xp, aval, padrao_v, "V_apriori")
    xp["A_n"] = xp["N"] + xp["IBNR_n"]
    xp["A_v"] = xp["V"] + xp["IBNR_v"]
    return xp


def padroes_e_apriori(sinistros: pd.DataFrame, nivel_pooling: pd.Series, xp_preliminar: pd.DataFrame | None,
                      data_avaliacao, meses_maduro: int = 12) -> tuple[pd.DataFrame, pd.DataFrame, tuple[float, float]]:
    """Padrões de aviso (contagem e valor limitado) e A/E a priori da carteira, medido nos anos-apólice
    já maduros (encerrados há pelo menos `meses_maduro`)."""
    s = sinistros.copy()
    s["valor_lim"] = np.minimum(s["valor_indenizado"].fillna(0) + s["valor_reserva"].fillna(0),
                                s["id_estipulante"].map(nivel_pooling).fillna(np.inf))
    pn = padrao_aviso(s, data_avaliacao)
    pv = padrao_aviso(s, data_avaliacao, coluna_valor="valor_lim")
    ae = (1.0, 1.0)
    if xp_preliminar is not None:
        maduro = xp_preliminar[xp_preliminar["fim_ano"] <= pd.Timestamp(data_avaliacao) - pd.DateOffset(months=meses_maduro)]
        if len(maduro) and maduro["E_manual"].sum() > 0:
            ae = (float(maduro["N"].sum() / maduro["E_manual"].sum()), float(maduro["V"].sum() / maduro["V_manual"].sum()))
    return pn, pv, ae


# --------------------------------------------------------------------------- credibilidade

def credibilidade_flutuacao_limitada(n, n_completo: float, n_minimo: float = 0.0) -> np.ndarray:
    """Regra da raiz quadrada: Z = min(1, sqrt(n / n_completo)); Z = 0 abaixo do mínimo."""
    n = np.asarray(n, float)
    return np.where(n < n_minimo, 0.0, np.minimum(1.0, np.sqrt(np.clip(n, 0, None) / n_completo)))


def buhlmann_straub(df: pd.DataFrame, grupo: str, periodo: str, x: str, w: str) -> dict:
    """Bühlmann-Straub pelo actuar::cm (R), com os estimadores de Bühlmann-Gisler.

    x: razão observada por grupo-período (aqui, A/E em valor limitado); w: peso (aqui, n_ef).
    Devolve s2 (variância dentro), a (variância entre grupos), k = s2/a e, por grupo, peso total,
    média ponderada e Z = w / (w + k). Com a <= 0 (sem heterogeneidade detectável), Z = 0."""
    import json
    import subprocess
    import tempfile
    from pathlib import Path

    from .config import RAIZ
    d = df[[grupo, periodo, x, w]].dropna()
    d = d[d[w] > 0].rename(columns={grupo: "grupo", periodo: "periodo", x: "x", w: "w"})
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        d.to_csv(t / "e.csv", index=False)
        r = subprocess.run(["Rscript", str(RAIZ / "r" / "buhlmann_straub.R"), str(t / "e.csv"), str(t / "s.json")],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"actuar::cm falhou:\n{r.stderr[-2000:]}")
        o = json.loads((t / "s.json").read_text())
    grupos = [str(g) for g in (o["grupos"] if isinstance(o["grupos"], list) else [o["grupos"]])]
    lst = lambda v: v if isinstance(v, list) else [v]
    por = pd.DataFrame({"Z": lst(o["Z"]), "media": lst(o["media_grupo"])}, index=pd.Index(grupos, name=grupo))
    g = d.groupby("grupo")
    por["peso"] = g["w"].sum().reindex(por.index).to_numpy()
    por["periodos"] = g.size().reindex(por.index).to_numpy()
    a, s2 = float(o["a"]), float(o["s2"])
    return {"s2": s2, "a": a, "k": s2 / a if a > 0 else np.inf, "media_carteira": float(o["media_carteira"]),
            "grupos": len(grupos), "por_grupo": por}


def fatores_experiencia(xp: pd.DataFrame, cfg_cred: dict, k_bs: float | None = None,
                        anos_max: int = 5) -> pd.DataFrame:
    """Fator de experiência por grupo com cada método de credibilidade, usando até `anos_max`
    anos-apólice mais recentes de cada grupo."""
    x = xp.sort_values(["id_estipulante", "ano_apolice"]).groupby("id_estipulante").tail(anos_max)
    g = x.groupby("id_estipulante").agg(vidas_ano=("vidas_ano", "sum"), E_manual=("E_manual", "sum"),
                                         V_manual=("V_manual", "sum"), n_ef=("n_ef", "sum"),
                                         A_v=("A_v", "sum"), A_n=("A_n", "sum"))
    g["AE_valor"] = np.where(g["V_manual"] > 0, g["A_v"] / g["V_manual"], 1.0)
    lf = cfg_cred["flutuacao_limitada"]
    g["Z_mercado"] = credibilidade_flutuacao_limitada(g[lf["base"]], lf["n_completo"], lf["n_minimo"])
    g["Z_bs"] = g["n_ef"] / (g["n_ef"] + k_bs) if k_bs is not None and np.isfinite(k_bs) else 0.0
    for m in ("mercado", "bs"):
        g[f"fator_{m}"] = g[f"Z_{m}"] * g["AE_valor"] + (1 - g[f"Z_{m}"])
    g["fator_manual"] = 1.0
    g["fator_experiencia_pura"] = g["AE_valor"]
    return g.reset_index()


# --------------------------------------------------------------------------- reajuste

def custo_manual_censo(vidas_ativas: pd.DataFrame, estipulantes: pd.DataFrame, tabua: Tabua, tabela_manual: dict,
                       nivel_pooling: pd.Series, data_renovacao) -> pd.DataFrame:
    """Para o censo de cada grupo na data de renovação: capital, custo esperado na manual limitado ao
    pooling e encargo de pooling (excesso esperado), em R$ por ano."""
    v = vidas_ativas.merge(estipulantes[["id_estipulante", "cnae_secao", "cnae_divisao"]], on="id_estipulante")
    d = pd.Timestamp(data_renovacao)
    v["idade"] = idade_completa(v["data_nascimento"], pd.Series(d, index=v.index)).astype(int)
    q = tabua.q(v["idade"].to_numpy(), v["sexo"].astype(object).to_numpy())
    mult, _ = multiplicador_manual(v, tabela_manual)
    P = v["id_estipulante"].map(nivel_pooling).fillna(np.inf).to_numpy()
    C = v["capital"].to_numpy(float)
    v["custo_lim"] = q * mult * np.minimum(C, P)
    v["encargo_pool"] = q * mult * np.maximum(C - P, 0.0)
    return v.groupby("id_estipulante").agg(vidas=("id_vida", "count"), capital=("capital", "sum"),
                                           custo_manual_lim=("custo_lim", "sum"), encargo_pooling=("encargo_pool", "sum"))


def reajuste_tecnico(custo: pd.DataFrame, fatores: pd.DataFrame, coluna_fator: str, taxa_vigente_mensal: pd.Series,
                     cfg_reaj: dict) -> pd.DataFrame:
    """Taxa pura e comercial de renovação e reajuste sobre a taxa vigente, com piso e teto."""
    r = custo.join(fatores.set_index("id_estipulante")[[coluna_fator]], how="left").fillna({coluna_fator: 1.0})
    r["premio_puro"] = r["custo_manual_lim"] * r[coluna_fator] + r["encargo_pooling"]
    r["taxa_pura_mensal_mil"] = 1000 * r["premio_puro"] / r["capital"] / 12
    r["taxa_comercial_mensal_mil"] = r["taxa_pura_mensal_mil"] / (1 - cfg_reaj["carregamento_total"])
    r["taxa_vigente_mensal_mil"] = taxa_vigente_mensal.reindex(r.index)
    r["reajuste_tecnico"] = r["taxa_comercial_mensal_mil"] / r["taxa_vigente_mensal_mil"] - 1
    r["reajuste_aplicado"] = r["reajuste_tecnico"].clip(cfg_reaj["piso"], cfg_reaj["teto"])
    r["limitado"] = r["reajuste_tecnico"].ne(r["reajuste_aplicado"])
    return r


def reajuste_regra_contratual(premio: pd.Series, sinistros_avisados: pd.Series, faixas: list[dict],
                              ibnr_pct_premio: float) -> pd.DataFrame:
    """Regra por faixa de sinistralidade: IS = (sinistros avisados + IBNR fixo % do prêmio) / prêmio."""
    sp = (sinistros_avisados.reindex(premio.index).fillna(0) + ibnr_pct_premio * premio) / premio
    reaj = sp.apply(lambda v: next(float(f["reajuste"]) for f in faixas if v <= float(f["ate"])))
    return pd.DataFrame({"sinistralidade_regra": sp, "reajuste_regra": reaj})


# --------------------------------------------------------------------------- carteira inteira

def ae_apriori_carteira(xp: pd.DataFrame, data_avaliacao, meses_maduro: int = 12) -> tuple[float, float]:
    """A/E da carteira (contagem, valor limitado) contra a manual, nos anos-apólice já maduros."""
    maduro = xp[xp["fim_ano"] <= pd.Timestamp(data_avaliacao) - pd.DateOffset(months=meses_maduro)]
    if not len(maduro) or maduro["E_manual"].sum() <= 0:
        return 1.0, 1.0
    return float(maduro["N"].sum() / maduro["E_manual"].sum()), float(maduro["V"].sum() / maduro["V_manual"].sum())


def renovar_carteira(dados: dict, tabua: Tabua, tabela_manual: dict, cfg: dict, data_avaliacao) -> dict:
    """Renovação de todos os grupos na data de avaliação: experiência com IBNR, credibilidade,
    reajuste técnico e comparação com a regra contratual."""
    v, s, e = dados["vidas"], dados["sinistros"], dados["estipulantes"]
    aval = pd.Timestamp(data_avaliacao)
    P = niveis_pooling(v, aval, cfg["pooling"])
    pn, pv, _ = padroes_e_apriori(s, P, None, aval)
    xp = experiencia_renovacao(v, s, e, tabua, tabela_manual, aval, P, pn, pv, (1.0, 1.0))
    ae = ae_apriori_carteira(xp, aval)
    xp = experiencia_renovacao(v, s, e, tabua, tabela_manual, aval, P, pn, pv, ae)
    xp["AE_v"] = np.where(xp["V_manual"] > 0, xp["A_v"] / xp["V_manual"], np.nan)
    bs = buhlmann_straub(xp, "id_estipulante", "ano_apolice", "AE_v", "n_ef")
    fat = fatores_experiencia(xp, cfg["credibilidade"], bs["k"], cfg["anos_experiencia_max"])
    ativas = v[(v["data_inicio"] <= aval) & (v["data_fim"].isna() | (v["data_fim"] >= aval))]
    custo = custo_manual_censo(ativas, e, tabua, tabela_manual, P, aval + pd.Timedelta(days=1))
    col = {"bs": "fator_bs", "mercado": "fator_mercado"}[cfg["credibilidade"]["metodo"]]
    vigente = e.set_index("id_estipulante")["taxa_vigente_mensal_mil"].astype(float)
    reaj = reajuste_tecnico(custo, fat, col, vigente, cfg["reajuste"])

    # regra contratual sobre o último ano-apólice completo de cada grupo
    ult = xp[xp["ano_completo"]].sort_values("ano_apolice").groupby("id_estipulante").tail(1)
    pr = dados["premios"].merge(ult[["id_estipulante", "inicio_ano"]], left_on=["id_estipulante", "inicio_periodo"],
                                right_on=["id_estipulante", "inicio_ano"])
    premio = pr.set_index("id_estipulante")["premio_emitido"]
    sin = ult.set_index("id_estipulante")["V_total"]
    rc = cfg["regra_contratual"]
    regra = reajuste_regra_contratual(premio, sin, rc["faixas"], rc["ibnr_pct_premio"])
    reaj = reaj.join(regra, how="left")
    return {"experiencia": xp, "padrao_contagem": pn, "padrao_valor": pv, "ae_apriori": ae, "pooling": P,
            "bs": bs, "fatores": fat, "reajustes": reaj}
