"""Exposição e óbitos esperados por estipulante e ano-apólice.

Convenções (todas também no config e no relatório):
- Segmento cobre de data_inicio a data_fim, ambos inclusive; data_fim nula = em vigor
  até a data de referência.
- Ano-apólice: do aniversário da apólice até a véspera do próximo; o último pode
  ser parcial (cortado na data de referência ou no fim de vigência).
- Exposição em anos = dias / dias_por_ano.
- Idade: idade completa no início de cada pedaço (segmento ∩ ano-apólice).
- Óbitos esperados de um pedaço = q_x × exposição (aproximação linear em t·q_x,
  adequada para q_x pequeno e pedaços de até um ano).
- Número efetivo de óbitos esperados (peso de credibilidade quando o A/E é
  ponderado por capital): n_ef = (Σ C·q·t)² / Σ C²·q·t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .tabua import Tabua


def idade_completa(nascimento: pd.Series, data: pd.Series) -> pd.Series:
    """Idade em anos completos (último aniversário) na data."""
    n = pd.to_datetime(nascimento)
    d = pd.to_datetime(data)
    idade = d.dt.year - n.dt.year
    antes = (d.dt.month < n.dt.month) | ((d.dt.month == n.dt.month) & (d.dt.day < n.dt.day))
    return (idade - antes.astype(int)).astype("Int64")


def anos_apolice(estipulantes: pd.DataFrame, data_referencia) -> pd.DataFrame:
    data_ref = pd.Timestamp(data_referencia)
    linhas = []
    for r in estipulantes.itertuples(index=False):
        ini = pd.Timestamp(r.data_inicio_vigencia)
        fim_vig = getattr(r, "data_fim_vigencia", pd.NaT)
        limite = data_ref if pd.isna(fim_vig) else min(pd.Timestamp(fim_vig), data_ref)
        k = 0
        while True:
            ps = ini + pd.DateOffset(years=k)
            if ps > limite:
                break
            pe_cheio = ini + pd.DateOffset(years=k + 1) - pd.Timedelta(days=1)
            pe = min(pe_cheio, limite)
            linhas.append((r.id_estipulante, k + 1, ps, pe, bool(pe == pe_cheio)))
            k += 1
    return pd.DataFrame(linhas, columns=["id_estipulante", "ano_apolice", "inicio_ano", "fim_ano", "ano_completo"])


HIERARQUIA_SEXO = [["id_estipulante", "faixa10"], ["id_estipulante"], ["cnae_secao", "faixa10"], ["faixa10"]]


def proporcao_masculina(df: pd.DataFrame, niveis: list[list[str]] = HIERARQUIA_SEXO, peso: str | None = None,
                        padrao: float = 0.5) -> np.ndarray:
    """Proporção de homens para cada linha, entre as linhas com sexo conhecido, no nível mais fino da hierarquia que
    tiver dados (padrão: mesmo grupo e faixa de 10 anos, o grupo, a seção CNAE e a faixa, a faixa na carteira).
    Sem nenhuma informação, 0,5. Usada para o esperado das vidas com sexo ausente."""
    d = df.copy()
    if "faixa10" not in d:
        d["faixa10"] = (d["idade"] // 10).astype(int)
    sx = d["sexo"].astype("string")
    d["_w"] = d[peso].astype(float) if peso else 1.0
    d["_mw"] = (sx == "M").astype(float) * d["_w"]
    conhecido = d[sx.isin(["M", "F"]).fillna(False).to_numpy()]
    p = pd.Series(np.nan, index=d.index)
    for chaves in niveis:
        chaves = [c for c in chaves if c in d]
        if not chaves or not p.isna().any():
            continue
        g = conhecido.groupby(chaves)[["_mw", "_w"]].sum()
        taxa = (g["_mw"] / g["_w"]).rename("_p")
        p = p.fillna(d[chaves].merge(taxa, left_on=chaves, right_index=True, how="left")["_p"])
    return p.fillna(padrao).to_numpy()


def exposicao_detalhada(vidas: pd.DataFrame, estipulantes: pd.DataFrame, tabua: Tabua,
                        data_referencia, dias_por_ano: float = 365.25, data_inicio=None) -> pd.DataFrame:
    """Um registro por pedaço (segmento ∩ ano-apólice) com exposição e esperados. `data_inicio` (opcional)
    corta a exposição anterior a essa data: com `data_referencia`, define uma janela de observação (usada para
    separar treino e teste fora do tempo)."""
    data_ref = pd.Timestamp(data_referencia)
    py = anos_apolice(estipulantes, data_ref)
    v = vidas.copy()
    v["data_fim"] = v["data_fim"].fillna(data_ref).clip(upper=data_ref)
    m = v.merge(py, on="id_estipulante", how="inner")
    m["ini"] = m[["data_inicio", "inicio_ano"]].max(axis=1)
    if data_inicio is not None:
        m["ini"] = m["ini"].clip(lower=pd.Timestamp(data_inicio))
    m["fim"] = m[["data_fim", "fim_ano"]].min(axis=1)
    m = m[m["ini"] <= m["fim"]].copy()
    m["dias"] = (m["fim"] - m["ini"]).dt.days + 1
    m["exposicao"] = m["dias"] / dias_por_ano
    m["idade"] = idade_completa(m["data_nascimento"], m["ini"]).astype(int)
    m["sexo_informado"] = m["sexo"].astype("string").isin(["M", "F"]).fillna(False).to_numpy()
    p_masc = None
    if not m["sexo_informado"].all():
        aux = m[["id_estipulante", "idade", "sexo", "exposicao"]].merge(
            estipulantes[["id_estipulante", "cnae_secao"]], on="id_estipulante", how="left")   # mantém a ordem
        p_masc = proporcao_masculina(aux, peso="exposicao")
    m["qx"] = tabua.q(m["idade"].to_numpy(), m["sexo"].astype(object).to_numpy(), p_masc)
    m["obitos_esperados"] = m["qx"] * m["exposicao"]
    m["capital_ano"] = m["capital"] * m["exposicao"]
    m["valor_esperado"] = m["qx"] * m["capital_ano"]
    m["soma_c2q"] = m["capital"] ** 2 * m["obitos_esperados"]
    cols = ["id_linha", "id_estipulante", "id_vida", "ano_apolice", "inicio_ano", "fim_ano", "ano_completo",
            "ini", "fim", "dias", "exposicao", "idade", "sexo", "sexo_informado", "capital", "qx",
            "obitos_esperados", "capital_ano", "valor_esperado", "soma_c2q"]
    return m[[c for c in cols if c in m.columns]].reset_index(drop=True)


def sinistros_por_ano(sinistros: pd.DataFrame, estipulantes: pd.DataFrame, data_referencia) -> pd.DataFrame:
    """Atribui cada sinistro ao ano-apólice da data de ocorrência."""
    py = anos_apolice(estipulantes, data_referencia)
    s = sinistros.merge(py, on="id_estipulante", how="inner")
    s = s[(s["data_ocorrencia"] >= s["inicio_ano"]) & (s["data_ocorrencia"] <= s["fim_ano"])].copy()
    s["valor_incorrido"] = s["valor_indenizado"].fillna(0) + s["valor_reserva"].fillna(0)
    return s


def experiencia(detalhe: pd.DataFrame, sinistros: pd.DataFrame | None, estipulantes: pd.DataFrame,
                data_referencia, por: tuple[str, ...] = ("id_estipulante", "ano_apolice")) -> pd.DataFrame:
    """Tabela de experiência A/E agregada (padrão: estipulante × ano-apólice).

    Os observados são os sinistros AVISADOS até a data de referência (sem IBNR);
    o ajuste de IBNR entra na etapa de renovação.
    """
    por = list(por)
    agg = detalhe.groupby(por, as_index=False).agg(
        vidas_ano=("exposicao", "sum"), capital_ano=("capital_ano", "sum"),
        obitos_esperados=("obitos_esperados", "sum"), valor_esperado=("valor_esperado", "sum"),
        soma_c2q=("soma_c2q", "sum"))
    if sinistros is not None and len(sinistros):
        s = sinistros_por_ano(sinistros, estipulantes, data_referencia)
        extras = [c for c in por if c not in s.columns and c in estipulantes.columns]
        if extras:
            s = s.merge(estipulantes[["id_estipulante", *extras]], on="id_estipulante")
        obs = s.groupby(por, as_index=False).agg(
            obitos_observados=("id_sinistro", "count"), valor_observado=("valor_incorrido", "sum"))
        agg = agg.merge(obs, on=por, how="left")
    else:
        agg["obitos_observados"] = 0
        agg["valor_observado"] = 0.0
    agg[["obitos_observados", "valor_observado"]] = agg[["obitos_observados", "valor_observado"]].fillna(0)
    agg["n_efetivo"] = np.where(agg["soma_c2q"] > 0, agg["valor_esperado"] ** 2 / agg["soma_c2q"], 0.0)
    agg["ae_obitos"] = agg["obitos_observados"] / agg["obitos_esperados"]
    agg["ae_valor"] = agg["valor_observado"] / agg["valor_esperado"]
    return agg


def capital_exposto_periodos(vidas: pd.DataFrame, periodos: pd.DataFrame, data_referencia,
                             dias_por_ano: float = 365.25) -> pd.Series:
    """Capital-ano exposto de cada período (linhas de `periodos`: id_estipulante,
    inicio_periodo, fim_periodo). Usado na conciliação de prêmio."""
    data_ref = pd.Timestamp(data_referencia)
    v = vidas[["id_estipulante", "data_inicio", "data_fim", "capital"]].copy()
    v["data_fim"] = v["data_fim"].fillna(data_ref).clip(upper=data_ref)
    p = periodos[["id_estipulante", "inicio_periodo", "fim_periodo"]].reset_index().rename(columns={"index": "_i"})
    m = p.merge(v, on="id_estipulante")
    ini = m[["data_inicio", "inicio_periodo"]].max(axis=1)
    fim = m[["data_fim", "fim_periodo"]].min(axis=1)
    dias = ((fim - ini).dt.days + 1).clip(lower=0)
    m["cap_ano"] = m["capital"] * dias / dias_por_ano
    return m.groupby("_i")["cap_ano"].sum().reindex(p["_i"]).fillna(0.0).set_axis(periodos.index)
