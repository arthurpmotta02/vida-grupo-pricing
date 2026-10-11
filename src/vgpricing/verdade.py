"""Verdade do gerador sintético. Serve só para TESTE DE SOFTWARE (o estimador recupera o que
foi plantado?). Não é evidência sobre o mercado: na base real não existe 'verdade', e a validação
de negócio é o backtest fora do tempo (Etapa 6)."""
from __future__ import annotations

import pandas as pd


def relatividades_secao(verdade_estipulantes: pd.DataFrame, base: str) -> pd.Series:
    f = verdade_estipulantes.groupby("cnae_secao")["fator_secao"].first()
    return f / f[base]


def custo_estrutural_por_grupo(detalhe: pd.DataFrame, verdade_estipulantes: pd.DataFrame,
                               verdade_vidas: pd.DataFrame) -> pd.Series:
    """Custo esperado verdadeiro em R$ por grupo, sem o efeito aleatório do grupo
    (tábua x seção x divisão x gradiente de renda)."""
    d = detalhe.merge(verdade_estipulantes[["id_estipulante", "fator_secao", "fator_divisao"]], on="id_estipulante")
    d = d.merge(verdade_vidas, on=["id_estipulante", "id_vida"], how="left")
    mult = d["fator_secao"] * d["fator_divisao"] * d["mult_renda"].fillna(1.0)
    return (d["valor_esperado"] * mult).groupby(d["id_estipulante"]).sum()
