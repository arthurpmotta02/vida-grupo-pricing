"""Verdade do gerador sintético, para medir o erro dos modelos. Só existe para dados sintéticos:
na base real não há 'verdade', e esses números não podem ser calculados."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .exposicao import exposicao_detalhada
from .manual import _faixa
from .tabua import Tabua


def detalhe_com_verdade(vidas, estipulantes, tabua: Tabua, data_referencia, cfg_manual, verdade_vidas,
                        cfg_sintetico, dias_por_ano=365.25) -> pd.DataFrame:
    corte = pd.Timestamp(data_referencia) - pd.DateOffset(months=cfg_manual["meses_maturacao"])
    det = exposicao_detalhada(vidas, estipulantes, tabua, corte, dias_por_ano)
    det = det.merge(estipulantes[["id_estipulante", "setor"]], on="id_estipulante")
    det = det.merge(verdade_vidas, on=["id_estipulante", "id_vida"], how="left")
    det["mult_renda"] = det["mult_renda"].fillna(1.0)
    fs = {s: v["fator_mortalidade"] for s, v in cfg_sintetico["setores"].items()}
    det["mult_estrutural"] = det["mult_renda"] * det["setor"].map(fs)   # sem efeito de grupo
    det["faixa_capital"] = _faixa(det["capital"], cfg_manual["faixas"]["capital"])
    return det


def relatividades_capital(det: pd.DataFrame, base: str) -> pd.Series:
    """Fator verdadeiro de cada faixa de capital (ponderado por óbitos esperados, como o modelo),
    relativo à faixa base."""
    g = det.assign(Em=det["obitos_esperados"] * det["mult_renda"]).groupby("faixa_capital")[["Em", "obitos_esperados"]].sum()
    r = g["Em"] / g["obitos_esperados"]
    return r / r[base]


def custo_estrutural_por_grupo(det: pd.DataFrame) -> pd.Series:
    """Custo esperado verdadeiro em R$ por grupo, sem o efeito aleatório do grupo."""
    return (det["valor_esperado"] * det["mult_estrutural"]).groupby(det["id_estipulante"]).sum()
