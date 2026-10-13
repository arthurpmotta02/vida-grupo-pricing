import pandas as pd
import pytest

from vgpricing.config import carregar_config
from vgpricing.cotador import alertas, cotar

T = pd.Timestamp


def _tabela():
    return {"ae_base": 1.0, "secao": {"N": 1.0}, "secao_nao_vista": 1.0, "divisao": {}, "divisao_nao_vista": 1.0,
            "capital": {"referencia": 50000.0, "elasticidade": 0.0}, "fatores_categoricos": {}, "faixas": {}}


def test_cotacao_golden(tabua_constante):
    """q = 0,001 (M), capitais 50 mil e 200 mil, pooling = max(100 mil, 3 x 125 mil) = 375 mil (sem excesso).
    puro = 0,001 x 250 mil = 250; comercial = 250 x 1,05 / (1 - carga)."""
    cc, cr = carregar_config("cotador"), carregar_config("renovacao")
    censo = pd.DataFrame({"data_nascimento": [T("1980-01-01")] * 2, "sexo": ["M", "M"], "capital": [50000.0, 200000.0]})
    r = cotar(censo, "N", "81", _tabela(), tabua_constante, "2026-01-01", cc, cr)
    carga = sum(cc["carregamentos"].values())
    assert r["nivel_pooling"] == pytest.approx(375000)
    assert r["premio_puro_anual"] == pytest.approx(250.0)
    assert r["premio_comercial_anual"] == pytest.approx(250 * 1.05 / (1 - carga))
    assert r["taxa_pura_mensal_mil"] == pytest.approx(1000 * 250 / 250000 / 12)


def test_encargo_de_pooling_cobra_o_excesso(tabua_constante):
    cc, cr = carregar_config("cotador"), carregar_config("renovacao")
    censo = pd.DataFrame({"data_nascimento": [T("1980-01-01")] * 3, "sexo": "M", "capital": [10000.0, 10000.0, 1000000.0]})
    r = cotar(censo, "N", "81", _tabela(), tabua_constante, "2026-01-01", cc, cr)
    # pooling = max(100 mil, 3 x 340 mil) = 1,02 mi: o capital de 1 mi fica abaixo; sem excesso
    assert r["encargo_pooling"] == pytest.approx(0.0)
    censo.loc[2, "capital"] = 5000000.0          # agora acima do teto de retenção (2 mi)
    r = cotar(censo, "N", "81", _tabela(), tabua_constante, "2026-01-01", cc, cr)
    assert r["nivel_pooling"] == pytest.approx(2e6) and r["encargo_pooling"] == pytest.approx(0.001 * 3e6)
    assert r["alertas"].set_index("alerta").loc["capital acima da retenção (resseguro facultativo)", "disparou"]


def test_alertas_de_concentracao_e_porte():
    cfg = carregar_config("cotador")["alertas"]
    censo = pd.DataFrame({"data_nascimento": [T("1940-01-01")] + [T("1990-01-01")] * 4, "capital": [1e6] + [1e4] * 4})
    a = alertas(censo, "2026-01-01", cfg).set_index("alerta")["disparou"]
    assert a["concentração: maior capital"] and a["idade acima do limite de aceitação"] and a["grupo abaixo do mínimo de vidas"]
