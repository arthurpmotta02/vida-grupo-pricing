"""IBNR (sinistros ocorridos e não avisados) para a renovação.

1. Padrão de aviso da CARTEIRA por chain ladder mensal (pacote chainladder, da CAS): origem = mês de
   ocorrência; desenvolvimento = mês de aviso. Por grupo não há sinistros suficientes para um triângulo.
2. Alocação por grupo e ano-apólice no estilo Bornhuetter-Ferguson: IBNR = esperados a priori x
   fração ainda não avisada. A fração depende da idade (meses entre a ocorrência e a avaliação), e os
   esperados a priori vêm da taxa manual ajustada pelo A/E da carteira. É mais estável que aplicar o
   chain ladder grupo a grupo, que daria IBNR zero a quem não teve sinistro avisado.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def padrao_aviso(sinistros: pd.DataFrame, data_avaliacao, coluna_valor: str | None = None,
                 media: str = "volume") -> pd.DataFrame:
    """Proporção acumulada avisada por idade em meses (1 = avisado no mês da ocorrência).

    Usa só sinistros ocorridos e avisados até a data de avaliação. Sem cauda: além da maior idade
    observada, considera-se tudo avisado. Devolve colunas idade_meses, ldf, cdf, proporcao_avisada."""
    import chainladder as cl
    aval = pd.Timestamp(data_avaliacao)
    s = sinistros[(sinistros["data_ocorrencia"] <= aval) & (sinistros["data_aviso"] <= aval)].copy()
    s["v"] = 1.0 if coluna_valor is None else s[coluna_valor].astype(float)
    df = s[["data_ocorrencia", "data_aviso", "v"]].copy()
    # linha de valor zero na data de avaliação: fixa a data de valoração do triângulo
    df = pd.concat([df, pd.DataFrame({"data_ocorrencia": [aval], "data_aviso": [aval], "v": [0.0]})], ignore_index=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tri = cl.Triangle(df, origin="data_ocorrencia", development="data_aviso", columns=["v"],
                          cumulative=False, origin_grain="M", development_grain="M").incr_to_cum()
        dev = cl.Development(average=media).fit(tri)
    ldf = dev.ldf_.to_frame(origin_as_datetime=False).iloc[0].to_numpy(float)
    ldf = np.where(np.isfinite(ldf) & (ldf > 0), ldf, 1.0)
    cdf = np.cumprod(ldf[::-1])[::-1]
    idades = np.arange(1, len(cdf) + 1)
    out = pd.DataFrame({"idade_meses": idades, "ldf": ldf, "cdf": cdf})
    out["proporcao_avisada"] = 1.0 / out["cdf"]
    return out


def fracao_nao_avisada(idade_meses: np.ndarray, padrao: pd.DataFrame) -> np.ndarray:
    idade = np.asarray(idade_meses, int)
    p = padrao.set_index("idade_meses")["proporcao_avisada"]
    prop = pd.Series(idade).map(p).to_numpy(float)
    prop = np.where(idade > p.index.max(), 1.0, prop)                 # além da maior idade: tudo avisado
    prop = np.where(idade < 1, p.iloc[0], prop)
    return 1.0 - np.nan_to_num(prop, nan=1.0)


def ibnr_por_periodo(periodos: pd.DataFrame, data_avaliacao, padrao: pd.DataFrame, coluna_apriori: str
                     ) -> pd.Series:
    """IBNR de cada período (linha com inicio_ano, fim_ano e o esperado a priori em `coluna_apriori`),
    supondo exposição uniforme dentro do período: soma, mês a mês, a priori do mês x fração não
    avisada na idade correspondente."""
    aval = pd.Timestamp(data_avaliacao)
    p = periodos[["inicio_ano", "fim_ano", coluna_apriori]].copy()
    p["fim_ef"] = p["fim_ano"].clip(upper=aval)
    p = p[p["fim_ef"] >= p["inicio_ano"]]
    meses = []
    for idx, r in p.iterrows():
        dias_tot = (r["fim_ef"] - r["inicio_ano"]).days + 1
        inicio_mes = r["inicio_ano"].to_period("M")
        for m in pd.period_range(inicio_mes, r["fim_ef"].to_period("M"), freq="M"):
            ini = max(m.start_time, r["inicio_ano"])
            fim = min(m.end_time.normalize(), r["fim_ef"])
            meses.append((idx, (fim - ini).days + 1, (aval.year - m.year) * 12 + aval.month - m.month + 1,
                          r[coluna_apriori] / dias_tot))
    if not meses:
        return pd.Series(0.0, index=periodos.index)
    m = pd.DataFrame(meses, columns=["idx", "dias", "idade", "apriori_dia"])
    m["ibnr"] = m["apriori_dia"] * m["dias"] * fracao_nao_avisada(m["idade"].to_numpy(), padrao)
    return m.groupby("idx")["ibnr"].sum().reindex(periodos.index).fillna(0.0)
