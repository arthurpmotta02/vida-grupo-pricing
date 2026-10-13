"""Cotação de grupo novo: do censo de vidas à taxa comercial, com alertas de subscrição.

taxa pura = [custo esperado na taxa manual, limitado ao pooling + encargo de pooling] / capital
taxa comercial = taxa pura x (1 + margem) / (1 - carregamentos)
Usa a tabela de fatores da taxa manual (`artefatos/fatores_manual.yaml`) e a mesma regra de pooling da renovação.
Não exige R.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .exposicao import idade_completa
from .renovacao import custo_manual_censo, niveis_pooling
from .tabua import Tabua


def cotar(censo: pd.DataFrame, cnae_secao: str, cnae_divisao: str | None, tabela: dict, tabua: Tabua,
          data_inicio_vigencia, cfg_cotador: dict, cfg_renovacao: dict) -> dict:
    """`censo`: uma linha por vida com data_nascimento, sexo e capital (o formato da aba `vidas` do template).
    Devolve taxas (anual e mensal, por mil), prêmios, decomposição do custo e alertas."""
    d = pd.Timestamp(data_inicio_vigencia)
    v = censo.copy()
    v["id_estipulante"], v["data_inicio"], v["data_fim"] = "COTACAO", d, pd.NaT
    if "id_vida" not in v:
        v["id_vida"] = np.arange(len(v)).astype(str)
    est = pd.DataFrame({"id_estipulante": ["COTACAO"], "cnae_secao": [cnae_secao], "cnae_divisao": [cnae_divisao]})
    P = niveis_pooling(v, d, cfg_renovacao["pooling"])
    custo = custo_manual_censo(v, est, tabua, tabela, P, d).iloc[0]
    puro = float(custo["custo_manual_lim"] + custo["encargo_pooling"])
    cap = float(custo["capital"])
    c = cfg_cotador
    carga = sum(c["carregamentos"].values())
    comercial = puro * (1 + c["margem_seguranca"]) / (1 - carga)
    return {
        "vidas": int(custo["vidas"]), "capital_total": cap, "nivel_pooling": float(P.iloc[0]),
        "custo_esperado_limitado": float(custo["custo_manual_lim"]), "encargo_pooling": float(custo["encargo_pooling"]),
        "premio_puro_anual": puro, "premio_comercial_anual": comercial,
        "taxa_pura_mensal_mil": 1000 * puro / cap / 12, "taxa_comercial_mensal_mil": 1000 * comercial / cap / 12,
        "carregamento_total": carga, "alertas": alertas(v, d, c["alertas"]),
    }


def alertas(censo: pd.DataFrame, data, cfg_alertas: dict) -> pd.DataFrame:
    """Alertas de subscrição: retenção, concentração de capital, idade e porte."""
    a = cfg_alertas
    cap = censo["capital"].astype(float)
    idade = idade_completa(censo["data_nascimento"], pd.Series(pd.Timestamp(data), index=censo.index)).astype(int)
    ordenado = cap.sort_values(ascending=False)
    regras = [
        ("capital acima da retenção (resseguro facultativo)", int((cap > a["limite_retencao"]).sum()),
         f"{int((cap > a['limite_retencao']).sum())} vida(s) acima de R$ {a['limite_retencao']:,.0f}"),
        ("concentração: maior capital", int(ordenado.iloc[0] / cap.sum() > a["maior_capital_sobre_total"]),
         f"maior capital = {ordenado.iloc[0] / cap.sum():.1%} do total"),
        ("concentração: 10 maiores capitais", int(ordenado.head(10).sum() / cap.sum() > a["dez_maiores_sobre_total"]),
         f"10 maiores = {ordenado.head(10).sum() / cap.sum():.1%} do total"),
        ("idade acima do limite de aceitação", int((idade > a["idade_maxima_aceitacao"]).sum()),
         f"{int((idade > a['idade_maxima_aceitacao']).sum())} vida(s) acima de {a['idade_maxima_aceitacao']} anos"),
        ("grupo abaixo do mínimo de vidas", int(len(censo) < a["minimo_vidas"]), f"{len(censo)} vidas"),
    ]
    t = pd.DataFrame(regras, columns=["alerta", "disparou", "detalhe"])
    t["disparou"] = t["disparou"] > 0
    return t
