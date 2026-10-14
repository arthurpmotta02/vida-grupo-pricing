import numpy as np
import pandas as pd
import pytest

from vgpricing import renovacao as R
from vgpricing.ibnr import fracao_nao_avisada, ibnr_por_periodo, padrao_aviso

T = pd.Timestamp


def test_flutuacao_limitada_raiz_e_minimo():
    z = R.credibilidade_flutuacao_limitada([100, 5000, 20000, 80000], n_completo=20000, n_minimo=500)
    assert np.allclose(z, [0.0, 0.5, 1.0, 1.0])


requer_r = pytest.mark.skipif(not __import__("vgpricing.glmm", fromlist=["r_disponivel"]).r_disponivel(),
                              reason="R com actuar não disponível")


@requer_r
def test_buhlmann_straub_caso_calculado_a_mao():
    # 2 grupos, 2 períodos, pesos 1: médias 1 e 3; s2 = [(0-1)^2+(2-1)^2+(2-3)^2+(4-3)^2]/2 = 2
    # a = [sum w_i (x_i - x)^2 - (I-1) s2] / (w - sum w_i^2 / w) = (2*1 + 2*1 - 2) / (4 - 2) = 1 -> k = 2
    d = pd.DataFrame({"g": ["A", "A", "B", "B"], "p": [1, 2, 1, 2], "x": [0.0, 2.0, 2.0, 4.0], "w": [1.0] * 4})
    r = R.buhlmann_straub(d, "g", "p", "x", "w")
    assert r["s2"] == pytest.approx(2.0) and r["a"] == pytest.approx(1.0) and r["k"] == pytest.approx(2.0)
    assert np.allclose(r["por_grupo"]["Z"], 0.5)
    assert r["por_grupo"]["peso"].tolist() == [2.0, 2.0]


@requer_r
def test_buhlmann_straub_sem_heterogeneidade_da_z_zero():
    d = pd.DataFrame({"g": ["A", "A", "B", "B"], "p": [1, 2, 1, 2], "x": [0.0, 2.0, 0.0, 2.0], "w": [1.0] * 4})
    r = R.buhlmann_straub(d, "g", "p", "x", "w")
    assert r["a"] <= 0 and np.isinf(r["k"]) and np.allclose(r["por_grupo"]["Z"], 0.0)


def test_niveis_pooling():
    v = pd.DataFrame({"id_estipulante": ["E1", "E1", "E2"], "capital": [10000.0, 50000.0, 3e6],
                      "data_inicio": T("2024-01-01"), "data_fim": [pd.NaT, pd.NaT, pd.NaT]})
    P = R.niveis_pooling(v, "2025-01-01", {"multiplo_capital_medio": 3, "piso": 100000, "limite_retencao": 2e6})
    assert P["E1"] == pytest.approx(100000) and P["E2"] == pytest.approx(2e6)


def test_reajuste_piso_teto_e_regra_contratual():
    custo = pd.DataFrame({"capital": [1e6, 1e6], "custo_manual_lim": [1200.0, 6000.0], "encargo_pooling": [0.0, 0.0]},
                         index=["E1", "E2"])
    fat = pd.DataFrame({"id_estipulante": ["E1", "E2"], "fator_bs": [1.0, 1.0]})
    r = R.reajuste_tecnico(custo, fat, "fator_bs", pd.Series({"E1": 0.2, "E2": 0.2}),
                           {"carregamento_total": 0.5, "piso": -0.15, "teto": 0.4})
    # E1: 1200/1e6*1000/12 = 0.1 pura -> 0.2 comercial -> reajuste 0; E2: 0.5 pura -> 1.0 -> +400% -> teto
    assert r.loc["E1", "reajuste_tecnico"] == pytest.approx(0.0)
    assert r.loc["E2", "reajuste_aplicado"] == pytest.approx(0.4) and r.loc["E2", "limitado"]
    regra = R.reajuste_regra_contratual(pd.Series({"A": 100.0, "B": 100.0, "C": 100.0}),
                                        pd.Series({"A": 50.0, "B": 65.0, "C": 90.0}),
                                        [{"ate": 0.75, "reajuste": 0.0}, {"ate": 0.85, "reajuste": 0.1},
                                         {"ate": float("inf"), "reajuste": 0.15}], 0.15)
    assert regra["reajuste_regra"].tolist() == [0.0, 0.10, 0.15]


def test_padrao_aviso_recupera_atraso_simulado():
    rng = np.random.default_rng(0)
    oc = T("2021-01-01") + pd.to_timedelta(rng.integers(0, 1460, 4000), unit="D")
    atraso = np.round(np.exp(rng.normal(np.log(45), 1.0, 4000))).astype(int)
    s = pd.DataFrame({"data_ocorrencia": oc, "data_aviso": oc + pd.to_timedelta(atraso, unit="D")})
    aval = T("2024-12-31")
    pad = padrao_aviso(s, aval)
    ocorr = s[s["data_ocorrencia"] <= aval]
    per = pd.DataFrame({"inicio_ano": [T("2024-01-01")], "fim_ano": [T("2024-12-31")], "E": [1.0]})
    verdade = (ocorr.loc[ocorr["data_ocorrencia"].dt.year == 2024, "data_aviso"] > aval).mean()
    assert ibnr_por_periodo(per, aval, pad, "E").iloc[0] == pytest.approx(verdade, abs=0.03)
    assert fracao_nao_avisada(np.array([200]), pad)[0] == 0.0


def test_experiencia_limita_ao_pooling_e_separa_excesso(tabua_constante):
    est = pd.DataFrame({"id_estipulante": ["E1"], "cnae_secao": ["N"], "cnae_divisao": ["81"],
                        "data_inicio_vigencia": [T("2023-01-01")], "data_fim_vigencia": [pd.NaT]})
    vidas = pd.DataFrame({"id_linha": [1, 2], "id_estipulante": "E1", "id_vida": ["A", "B"],
                          "data_nascimento": T("1980-01-01"), "sexo": "M", "data_inicio": T("2023-01-01"),
                          "data_fim": [T("2023-12-31"), T("2023-12-31")], "capital": [50000.0, 500000.0]})
    sin = pd.DataFrame({"id_sinistro": ["S1"], "id_estipulante": ["E1"], "id_vida": ["B"],
                        "data_ocorrencia": [T("2023-06-01")], "data_aviso": [T("2023-07-01")],
                        "valor_indenizado": [500000.0], "valor_reserva": [0.0]})
    tabela = {"ae_base": 1.0, "secao": {"N": 1.0}, "secao_nao_vista": 1.0, "divisao": {}, "divisao_nao_vista": 1.0,
              "capital": {"referencia": 50000.0, "elasticidade": 0.0}, "fatores_categoricos": {}, "faixas": {}}
    P = pd.Series({"E1": 100000.0})
    pad = pd.DataFrame({"idade_meses": [1], "ldf": [1.0], "cdf": [1.0], "proporcao_avisada": [1.0]})
    xp = R.experiencia_renovacao(vidas, sin, est, tabua_constante, tabela, "2024-06-30", P, pad, pad, (1.0, 1.0))
    t = 365 / 365.25
    assert xp.loc[0, "V_manual"] == pytest.approx(0.001 * t * (50000 + 100000))
    assert xp.loc[0, "V_excesso"] == pytest.approx(0.001 * t * 400000)
    assert xp.loc[0, "V"] == pytest.approx(100000.0) and xp.loc[0, "V_total"] == pytest.approx(500000.0)
    assert xp.loc[0, "IBNR_v"] == pytest.approx(0.0)


@requer_r
def test_credibilidade_mercado_favorece_credibilidade_com_unidades_pequenas():
    """Unidades com média própria verdadeira e pouco peso: Bühlmann-Straub erra menos que a experiência própria."""
    from vgpricing.backtest import credibilidade_mercado
    rng = np.random.default_rng(1)
    linhas = []
    for u in range(60):
        th = rng.normal(0.5, 0.05)
        w = 1.0 if u % 2 else 50.0
        for ano in range(2015, 2024):
            linhas.append({"unidade": f"U{u}", "ano": ano, "w": w, "x": th + rng.normal(0, 0.3 / np.sqrt(w))})
    r = credibilidade_mercado(pd.DataFrame(linhas), list(range(2019, 2024)), janela=4, n_boot=300)["resumo"].set_index("metodo")
    assert r.loc["Bühlmann-Straub", "erro quadrático ponderado"] < r.loc["experiência própria", "erro quadrático ponderado"]


def test_fator_desenvolvimento_completa_o_ibnr():
    from vgpricing.ibnr import fator_desenvolvimento
    rng = np.random.default_rng(0)
    oc = T("2021-01-01") + pd.to_timedelta(rng.integers(0, 1460, 6000), unit="D")
    s = pd.DataFrame({"data_ocorrencia": oc,
                      "data_aviso": oc + pd.to_timedelta(np.round(np.exp(rng.normal(np.log(45), 1.0, 6000))), unit="D")})
    aval, fim = T("2024-12-31"), T("2023-12-31")                       # 12 meses de maturação
    verdade = (s["data_ocorrencia"] <= fim).sum() / ((s["data_ocorrencia"] <= fim) & (s["data_aviso"] <= aval)).sum()
    assert fator_desenvolvimento(s, aval, fim) == pytest.approx(verdade, abs=0.004)
    assert 1.0 < verdade < 1.02


@requer_r
@pytest.mark.parametrize("d", [pd.DataFrame({"g": [], "p": [], "x": [], "w": []}),
                               pd.DataFrame({"g": ["A", "B"], "p": [1, 1], "x": [0.9, 1.1], "w": [2.0, 3.0]})])
def test_buhlmann_straub_sem_experiencia_suficiente_da_z_zero(d):
    with pytest.warns(UserWarning, match="insuficiente"):
        r = R.buhlmann_straub(d, "g", "p", "x", "w")
    assert np.isinf(r["k"]) and (r["por_grupo"]["Z"] == 0).all()
