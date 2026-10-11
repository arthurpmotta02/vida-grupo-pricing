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


def avaliar_contra_verdade(por_grupo: pd.DataFrame, detalhe_alvo: pd.DataFrame, verdade_estipulantes: pd.DataFrame,
                           verdade_vidas: pd.DataFrame, nivel_pooling: pd.Series, prob_negado: float,
                           metodos: dict) -> pd.DataFrame:
    """Compara o previsto por cada método com o custo ESPERADO verdadeiro do ano-alvo (sem o sorteio de
    Poisson), em valor limitado ao pooling. O nível de cada método é igualado à verdade antes do erro,
    para medir só a estrutura (quem paga quanto em relação aos outros).

    detalhe_alvo: exposicao_detalhada restrita aos anos-alvo (id_estipulante, id_vida, qx, exposicao, capital)."""
    import numpy as np
    d = detalhe_alvo.merge(verdade_estipulantes[["id_estipulante", "fator_secao", "fator_divisao", "fator_grupo"]],
                           on="id_estipulante").merge(verdade_vidas, on=["id_estipulante", "id_vida"], how="left")
    P = d["id_estipulante"].map(nivel_pooling).fillna(np.inf)
    m = d["qx"] * d["exposicao"] * d["fator_secao"] * d["fator_divisao"] * d["mult_renda"].fillna(1.0) * (1 - prob_negado)
    verdadeiro = (m * d["fator_grupo"] * np.minimum(d["capital"], P)).groupby(d["id_estipulante"]).sum()
    g = por_grupo.set_index("id_estipulante")
    v = verdadeiro.reindex(g.index)
    linhas = []
    for col, nome in metodos.items():
        p = g["V_manual"] * g[f"fator_{col}"] if col != "premio_emitido" else g["premio_emitido"]
        nivel = p.sum() / v.sum()
        pn = p / nivel
        linhas.append({"método": nome, "RMSE do log (estrutura)": float(np.sqrt(np.average(np.log(pn / v) ** 2, weights=v)))})
    return pd.DataFrame(linhas).sort_values("RMSE do log (estrutura)")
