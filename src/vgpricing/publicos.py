"""Leitura dos dados públicos reais usados no projeto.

- SOA 2016 Group Term Life Mortality Study (EUA, 2010-2013): células de exposição, óbitos e esperados
  por atividade, faixa de capital, salário, porte, sexo e idade. A extração é validada contra as
  tabelas publicadas no relatório (teste automático).
- SUSEP SES (Brasil): prêmios e sinistros por seguradora, ramo e mês, nos ramos de pessoas.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import caminho_repo

COLUNAS_SOA = {
    "industry": "atividade", "industry_cat": "colarinho", "face_amount_band": "faixa_capital",
    "salary": "faixa_salario", "group_size": "porte", "sex": "sexo", "central_age": "idade_central",
    "Type_of_Coverage": "tipo_cobertura",
    "death_policies_exposed": "vidas_ano", "death_amount_exposed": "capital_ano", "death_count": "obitos",
    "death_claim_amount": "valor_obitos", "expected_death_by_policy": "esperados_n",
    "expected_death_by_amount": "esperados_v",
}


def _categoria_ordenada(s: pd.Series) -> pd.Categorical:
    """Rótulos da SOA começam com letra de ordenação ('A. < 25', 'B. 25-49'...): mantém essa ordem."""
    niveis = sorted(s.dropna().unique(), key=lambda x: (x.split(".")[0].strip(), x))
    return pd.Categorical(s, categories=niveis, ordered=True)


def ler_soa(cfg_publicos: dict, raiz: Path | None = None) -> pd.DataFrame:
    c = cfg_publicos["soa"]
    caminho = caminho_repo(Path(cfg_publicos["pasta"]) / c["arquivo"], raiz)
    df = pd.read_csv(caminho, dtype=str).rename(columns=COLUNAS_SOA)
    for col in ["vidas_ano", "capital_ano", "obitos", "valor_obitos", "esperados_n", "esperados_v"]:
        df[col] = pd.to_numeric(df[col])
    df["idade_central"] = df["idade_central"].astype(int)
    df["sexo"] = df["sexo"].map({"Male": "M", "Female": "F"}).fillna(df["sexo"])
    for col in ["atividade", "colarinho", "faixa_capital", "faixa_salario", "porte"]:
        df[col] = _categoria_ordenada(df[col])
    return df


def _numero_br(s: pd.Series) -> pd.Series:
    """'191319500,08' -> 191319500.08 (o SES usa vírgula decimal)."""
    t = s.astype("string").str.strip().str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(t, errors="coerce")


def ler_ses(cfg_publicos: dict, raiz: Path | None = None) -> pd.DataFrame:
    """Prêmios e sinistros mensais por seguradora e ramo, em R$, com nomes. A coluna `sinistro` usa
    sinistro_ocorrido a partir de `ano_inicio_serie` (antes disso a SUSEP publicava sinistro_retido;
    as duas não são comparáveis, então anos anteriores ficam com sinistro vazio)."""
    c = cfg_publicos["ses"]
    pasta = caminho_repo(cfg_publicos["pasta"], raiz)
    df = pd.read_csv(pasta / c["arquivo"], sep=";", dtype=str)
    valores = [x for x in df.columns if x not in ("damesano", "coenti", "cogrupo", "coramo")]
    for col in valores:
        convertido = _numero_br(df[col])
        falhas = convertido.isna() & df[col].notna()
        if falhas.any():
            raise ValueError(f"SES: {int(falhas.sum())} valores não numéricos em {col}")
        df[col] = convertido
    df["competencia"] = pd.to_datetime(df["damesano"], format="%Y%m")
    df["ano"] = df["competencia"].dt.year
    df["sinistro"] = np.where(df["ano"] >= c["ano_inicio_serie"], df["sinistro_ocorrido"], np.nan)
    df["coenti"], df["coramo"] = df["coenti"].str.strip(), df["coramo"].str.strip()
    # os cadastros da SUSEP trazem os códigos com espaços à direita ('04561     ')
    cias = _cadastro(pasta / c["cias"], {"coenti": "coenti", "noenti": "noenti"})
    ramos = _cadastro(pasta / c["ramos"], {"coramo": "coramo", "noramo": "noramo"})
    df = df.merge(cias.drop_duplicates("coenti"), on="coenti", how="left")
    df = df.merge(ramos.drop_duplicates("coramo"), on="coramo", how="left")
    return df


def _cadastro(caminho: Path, colunas: dict) -> pd.DataFrame:
    d = pd.read_csv(caminho, sep=";", dtype=str, encoding="latin-1")
    d.columns = [x.strip().lower() for x in d.columns]
    d = d[list(colunas)].rename(columns=colunas)
    return d.apply(lambda col: col.str.strip())


def sinistralidade_anual(ses: pd.DataFrame, ramo: str, por_cia: bool = False) -> pd.DataFrame:
    """Sinistralidade = sinistro ocorrido / prêmio ganho, por ano (e por seguradora)."""
    chave = ["ano", "coenti", "noenti"] if por_cia else ["ano"]
    d = ses[(ses["coramo"] == ramo) & ses["sinistro"].notna()]
    g = d.groupby(chave, as_index=False)[["premio_ganho", "sinistro", "desp_com"]].sum()
    g["sinistralidade"] = g["sinistro"] / g["premio_ganho"]
    g["comissao_sobre_pg"] = g["desp_com"] / g["premio_ganho"]
    return g
