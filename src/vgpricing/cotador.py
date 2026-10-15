"""Cotação de grupo novo: do censo de vidas à taxa comercial, com alertas de subscrição.

taxa pura = [custo esperado na taxa manual, limitado ao pooling x fator de experiência + encargo de pooling] / capital

Na subscrição, além do censo, a seguradora costuma receber a sinistralidade dos últimos anos do grupo na seguradora
atual (sinistros e capital segurado por ano, agregados). Se ela vier, entra com a mesma credibilidade da renovação:
fator = Z x A/E + (1 - Z), Z = n_ef / (n_ef + k), com o k de Bühlmann-Straub estimado na carteira.
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


def fator_historico(historico: pd.DataFrame, custo_lim_atual: float, capital_atual: float, n_ef_ano: float,
                    k: float) -> dict:
    """Credibilidade da experiência anterior do grupo. `historico`: uma linha por ano com `sinistros` (R$, de
    preferência já limitados ao nível de pooling) e, opcionalmente, `capital` segurado no ano. O esperado de cada ano é
    o custo atual na taxa manual escalado pelo capital do ano; n_ef cresce com o número de anos."""
    h = historico.copy()
    escala = (h["capital"].to_numpy(float) / capital_atual) if "capital" in h else np.ones(len(h))
    esperado = float(custo_lim_atual * escala.sum())
    # sinistros ainda não avisados na data da cotação (IBNR): divide pela proporção já avisada, se informada
    sinistros = h["sinistros"] / h["proporcao_avisada"].clip(lower=0.05) if "proporcao_avisada" in h else h["sinistros"]
    ae = float(sinistros.sum() / esperado) if esperado > 0 else 1.0
    n_ef = n_ef_ano * float(escala.sum())
    z = n_ef / (n_ef + k) if np.isfinite(k) else 0.0
    return {"anos": len(h), "A/E historico": ae, "n_ef historico": n_ef, "Z": z, "fator": z * ae + (1 - z)}


def completar_ibnr(historico: pd.DataFrame, padrao: pd.DataFrame, data_cotacao) -> pd.DataFrame:
    """Acrescenta ao histórico (com `inicio` e `fim` de cada ano) a proporção dos sinistros que já deveria estar avisada
    na data da cotação, pelo padrão de aviso da carteira. O último ano costuma estar incompleto."""
    from .ibnr import proporcao_avisada_periodo
    h = historico.copy()
    h["proporcao_avisada"] = [proporcao_avisada_periodo(padrao, a, b, data_cotacao) for a, b in zip(h["inicio"], h["fim"])]
    return h


def cotar(censo: pd.DataFrame, cnae_secao: str, cnae_divisao: str | None, tabela: dict, tabua: Tabua,
          data_inicio_vigencia, cfg_cotador: dict, cfg_renovacao: dict, historico: pd.DataFrame | None = None,
          k_credibilidade: float | None = None) -> dict:
    """`censo`: uma linha por vida com data_nascimento, sexo e capital (o formato da aba `vidas` do template).
    `historico` (opcional): sinistralidade anterior do grupo, com `k_credibilidade` (k de Bühlmann-Straub da carteira).
    Devolve taxas (anual e mensal, por mil), prêmios, decomposição do custo, experiência e alertas."""
    d = pd.Timestamp(data_inicio_vigencia)
    v = censo.copy()
    v["id_estipulante"], v["data_inicio"], v["data_fim"] = "COTACAO", d, pd.NaT
    if "id_vida" not in v:
        v["id_vida"] = np.arange(len(v)).astype(str)
    est = pd.DataFrame({"id_estipulante": ["COTACAO"], "cnae_secao": [cnae_secao], "cnae_divisao": [cnae_divisao]})
    P = niveis_pooling(v, d, cfg_renovacao["pooling"])
    custo = custo_manual_censo(v, est, tabua, tabela, P, d).iloc[0]
    exp = {"fator": 1.0, "Z": 0.0}
    if historico is not None and len(historico) and k_credibilidade is not None:
        exp = fator_historico(historico, float(custo["custo_manual_lim"]), float(custo["capital"]),
                              float(custo["n_ef"]), k_credibilidade)
    puro = float(custo["custo_manual_lim"] * exp["fator"] + custo["encargo_pooling"])
    cap = float(custo["capital"])
    c = cfg_cotador
    carga = sum(c["carregamentos"].values())
    comercial = puro * (1 + c["margem_seguranca"]) / (1 - carga)
    return {
        "vidas": int(custo["vidas"]), "capital_total": cap, "nivel_pooling": float(P.iloc[0]),
        "obitos_esperados_manual": float(custo["obitos_esperados"]),
        "obitos_previstos": float(custo["obitos_esperados"] * exp["fator"]),
        "custo_esperado_limitado": float(custo["custo_manual_lim"]), "encargo_pooling": float(custo["encargo_pooling"]),
        "premio_puro_anual": puro, "premio_comercial_anual": comercial,
        "taxa_pura_mensal_mil": 1000 * puro / cap / 12, "taxa_comercial_mensal_mil": 1000 * comercial / cap / 12,
        "carregamento_total": carga, "experiencia": exp, "alertas": alertas(v, d, c["alertas"]),
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
