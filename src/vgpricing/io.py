"""Entrada de dados: Excel (template), CSV ou parquet, com mapeamento de colunas.

Princípios:
- O nome das colunas na fonte vem de config/colunas.yaml (nada hardcoded).
- Valores que não convertem para o tipo esperado viram nulos E entram no log
  `problemas_leitura`; nunca são descartados em silêncio.
- Cada linha recebe `id_linha` (posição na fonte) para rastreabilidade.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import ESQUEMA, colunas

TABELAS = ("estipulantes", "vidas", "sinistros", "premios")


def hash_id(valor: str, sal: str) -> str:
    """Pseudonimiza CPF/identificador (SHA-256 com sal). O sal fica fora do repositório."""
    return hashlib.sha256(f"{sal}|{valor}".encode()).hexdigest()[:16]


def _converter(serie: pd.Series, tipo: str) -> pd.Series:
    if tipo == "date":
        return pd.to_datetime(serie, errors="coerce", dayfirst=True)
    if tipo == "float":
        if serie.dtype == object or pd.api.types.is_string_dtype(serie):
            # aceita "1.234,56" (pt-BR) e "1234.56"
            s = serie.astype("string").str.strip()
            br = s.str.contains(",", na=False)
            s = s.where(~br, s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
            return pd.to_numeric(s, errors="coerce").astype(float)
        return pd.to_numeric(serie, errors="coerce").astype(float)
    if tipo == "bool":
        mapa = {"s": True, "sim": True, "true": True, "1": True, "n": False,
                "nao": False, "não": False, "false": False, "0": False}
        return serie.astype("string").str.strip().str.lower().map(mapa).astype("boolean")
    s = serie.astype("string").str.strip()
    return s.where(s != "", pd.NA)


def padronizar(df_fonte: pd.DataFrame, tabela: str, mapeamento: dict[str, str],
               linha_inicial: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Renomeia para nomes canônicos, converte tipos e devolve (df, problemas_leitura)."""
    mapa = {v: k for k, v in mapeamento.items() if v in df_fonte.columns}
    df = df_fonte.rename(columns=mapa)
    obrig = ESQUEMA[tabela]["obrigatorias"]
    faltando = [c for c in obrig if c not in df.columns]
    if faltando:
        raise KeyError(f"[{tabela}] colunas obrigatórias ausentes na fonte: {faltando}")
    tipos = colunas(tabela)
    out = pd.DataFrame(index=df.index)
    problemas = []
    for col, tipo in tipos.items():
        if col not in df.columns:
            out[col] = pd.Series(pd.NA, index=df.index, dtype="object") if tipo != "date" \
                else pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
            continue
        bruto = df[col]
        conv = _converter(bruto, tipo)
        falhou = conv.isna() & bruto.notna() & (bruto.astype("string").str.strip() != "")
        for i in df.index[falhou.fillna(False).to_numpy(bool)]:
            problemas.append({"tabela": tabela, "coluna": col, "id_linha": int(i) + linha_inicial,
                              "valor_original": str(bruto.loc[i]), "tipo_esperado": tipo})
        out[col] = conv
    out.insert(0, "id_linha", (df.index.to_numpy() + linha_inicial).astype(int))
    return out.reset_index(drop=True), pd.DataFrame(
        problemas, columns=["tabela", "coluna", "id_linha", "valor_original", "tipo_esperado"])


def ler_excel(caminho: str | Path, mapeamentos: dict) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Lê o template Excel (uma aba por tabela). Abas ausentes são permitidas
    para 'sinistros' e 'premios' (cotação de grupo novo)."""
    abas = pd.read_excel(caminho, sheet_name=None, dtype=object)
    dados, probs = {}, []
    for t in TABELAS:
        if t not in abas:
            if t in ("estipulantes", "vidas"):
                raise KeyError(f"Aba obrigatória ausente: {t}")
            continue
        df, p = padronizar(abas[t], t, mapeamentos[t], linha_inicial=2)  # linha 1 = cabeçalho
        dados[t] = df
        probs.append(p)
    return dados, pd.concat(probs, ignore_index=True)


def ler_pasta(pasta: str | Path, mapeamentos: dict, formato: str = "parquet"
              ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Lê <pasta>/<tabela>.<formato> (parquet ou csv)."""
    pasta = Path(pasta)
    dados, probs = {}, []
    for t in TABELAS:
        arq = pasta / f"{t}.{formato}"
        if not arq.exists():
            continue
        bruto = pd.read_parquet(arq) if formato == "parquet" else pd.read_csv(arq, dtype=str, sep=None, engine="python")
        if "id_linha" in bruto.columns:      # preserva rastreio quando já existe
            ids = bruto.pop("id_linha").to_numpy()
            df, p = padronizar(bruto, t, mapeamentos[t], linha_inicial=0)
            df["id_linha"] = ids.astype(int)
        else:
            df, p = padronizar(bruto, t, mapeamentos[t], linha_inicial=2)
        dados[t] = df
        probs.append(p)
    return dados, pd.concat(probs, ignore_index=True) if probs else pd.DataFrame()


def salvar_pasta(dados: dict[str, pd.DataFrame], pasta: str | Path, formato: str = "parquet") -> None:
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    for nome, df in dados.items():
        if formato == "parquet":
            df.to_parquet(pasta / f"{nome}.parquet", index=False)
        else:
            df.to_csv(pasta / f"{nome}.csv", index=False)


def faturamento_para_segmentos(fat: pd.DataFrame) -> pd.DataFrame:
    """Converte base de faturamento mensal (uma linha por vida x competência)
    em segmentos de cobertura.

    Entrada mínima: id_estipulante, id_vida, competencia (qualquer dia do mês),
    capital, data_nascimento, sexo; `situacao` opcional.
    Meses consecutivos com o mesmo capital viram um segmento; mudança de capital
    ou falha de competência abre um novo. data_fim = último dia do último mês.
    """
    f = fat.copy()
    f["competencia"] = pd.to_datetime(f["competencia"]).dt.to_period("M")
    if "situacao" not in f.columns:
        f["situacao"] = "ativo"
    f = f.sort_values(["id_estipulante", "id_vida", "competencia"]).reset_index(drop=True)
    chave = ["id_estipulante", "id_vida"]
    mes = f["competencia"].dt.year * 12 + f["competencia"].dt.month
    mesmo = (f[chave].eq(f[chave].shift()).all(axis=1)
             & mes.diff().eq(1)
             & f["capital"].eq(f["capital"].shift())
             & f["situacao"].eq(f["situacao"].shift()))
    f["seg"] = (~mesmo).cumsum()
    seg = f.groupby("seg").agg(
        id_estipulante=("id_estipulante", "first"), id_vida=("id_vida", "first"),
        data_nascimento=("data_nascimento", "first"), sexo=("sexo", "first"),
        inicio=("competencia", "min"), fim=("competencia", "max"),
        capital=("capital", "first"), situacao=("situacao", "first"))
    seg["data_inicio"] = seg.pop("inicio").dt.start_time.dt.normalize()
    seg["data_fim"] = seg.pop("fim").dt.end_time.dt.normalize()
    return seg.reset_index(drop=True)[
        ["id_estipulante", "id_vida", "data_nascimento", "sexo", "data_inicio", "data_fim", "capital", "situacao"]]
