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


def test_historico_do_grupo_entra_com_credibilidade(tabua_constante):
    cc, cr = carregar_config("cotador"), carregar_config("renovacao")
    censo = pd.DataFrame({"data_nascimento": [T("1980-01-01")] * 200, "sexo": "M", "capital": [50000.0] * 200})
    base = cotar(censo, "N", "81", _tabela(), tabua_constante, "2026-01-01", cc, cr)
    hist = pd.DataFrame({"ano": [2023, 2024, 2025], "sinistros": [2 * base["custo_esperado_limitado"]] * 3})
    pouca = cotar(censo, "N", "81", _tabela(), tabua_constante, "2026-01-01", cc, cr, hist, k_credibilidade=1e6)
    muita = cotar(censo, "N", "81", _tabela(), tabua_constante, "2026-01-01", cc, cr, hist, k_credibilidade=1e-6)
    assert pouca["premio_puro_anual"] == pytest.approx(base["premio_puro_anual"], rel=1e-3)   # k enorme: Z ~ 0
    assert muita["experiencia"]["A/E historico"] == pytest.approx(2.0)
    assert muita["premio_puro_anual"] == pytest.approx(2 * base["premio_puro_anual"], rel=1e-3)  # Z ~ 1
    assert 0 < cotar(censo, "N", "81", _tabela(), tabua_constante, "2026-01-01", cc, cr, hist, k_credibilidade=0.5)["experiencia"]["Z"] < 1


def test_completar_ibnr_eleva_o_ultimo_ano():
    from vgpricing.cotador import completar_ibnr
    pad = pd.DataFrame({"idade_meses": [1, 2, 3, 4], "proporcao_avisada": [0.2, 0.6, 0.9, 1.0]})
    h = pd.DataFrame({"inicio": [T("2023-01-01"), T("2025-10-01")], "fim": [T("2023-12-31"), T("2025-12-31")]})
    r = completar_ibnr(h, pad, "2026-01-01")
    assert r.loc[0, "proporcao_avisada"] == pytest.approx(1.0)
    assert 0.2 < r.loc[1, "proporcao_avisada"] < 0.9                      # out-dez: idades 2 a 4 em jan
