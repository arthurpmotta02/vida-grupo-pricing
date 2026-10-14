import numpy as np
import pandas as pd
import pytest

from vgpricing import avaliacao as A


def _dados(n=4000, semente=0):
    rng = np.random.default_rng(semente)
    E = rng.uniform(0.5, 3, n)
    rel = np.exp(rng.normal(0, 0.4, n))                  # relatividade verdadeira
    return pd.DataFrame({"E": E, "N": rng.poisson(E * rel), "verdade": E * rel, "tabua": E,
                         "ruido": E * np.exp(rng.normal(0, 0.4, n))})


def test_deviance_zero_quando_previsto_igual_observado():
    assert A.deviance_poisson([0, 2, 5], [1e-12 + 0, 2, 5]) == pytest.approx(0, abs=1e-9)


def test_metricas_ordenam_verdade_melhor_que_tabua_e_ruido():
    d = _dados()
    m = A.metricas(d, {"verdade": "verdade", "tábua": "tabua", "ruído": "ruido"}).set_index("modelo")
    assert m["deviance"].idxmin() == "verdade"
    assert m.loc["ruído", "deviance"] > m.loc["tábua", "deviance"]       # estrutura errada é pior que nenhuma


def test_faixas_tem_mesmo_esperado():
    d = _dados()
    f = A._faixas_iguais(d["verdade"].to_numpy() / d["E"].to_numpy(), d["E"].to_numpy(), 10)
    soma = pd.Series(d["E"].to_numpy()).groupby(f).sum()
    assert len(soma) == 10 and soma.max() / soma.min() < 1.05


def test_quantis_do_modelo_verdadeiro_acompanham_o_observado():
    q = A.quantis(_dados(), "verdade")
    assert q["previsto"].is_monotonic_increasing
    assert np.allclose(q["observado"], q["previsto"], rtol=0.15)


def test_double_lift_favorece_o_modelo_correto():
    d = A.double_lift(_dados(), "verdade", "tabua")
    # observado acompanha o modelo verdadeiro, não a tábua (que é 1 em todas as faixas)
    assert np.corrcoef(d["observado"], d["verdade"])[0, 1] > 0.95
    assert d["observado"].iloc[-1] > 1.3 and d["observado"].iloc[0] < 0.75


def test_gini():
    d = _dados()
    _, g_verdade = A.lorenz_gini(d, "verdade")
    _, g_tabua = A.lorenz_gini(d.assign(tabua=d["E"] * 1.0), "tabua")
    assert g_verdade > 0.15 and abs(g_tabua) < 0.05


def test_tabela_am_marca_desvios():
    d = pd.DataFrame({"g": ["a"] * 2 + ["b"] * 2, "N": [500, 500, 100, 100], "m": [500.0, 500.0, 150.0, 150.0]})
    t = A.tabela_am(d, "g", "m").set_index("g")
    assert t.loc["a", "A/M"] == pytest.approx(1.0) and not t.loc["a", "IC exclui 100%"]
    assert t.loc["b", "fora de ±10%"] and t.loc["b", "IC exclui 100%"]


def test_sinistralidade_por_faixa_sobe_quando_o_premio_ignora_o_risco():
    d = _dados()
    d["premio"] = d["E"]                                                # prêmio só pela tábua
    g = A.sinistralidade_por_faixa(d, "verdade", "premio", "N")
    assert g["sinistralidade observada"].iloc[-1] > 1.5 * g["sinistralidade observada"].iloc[0]
