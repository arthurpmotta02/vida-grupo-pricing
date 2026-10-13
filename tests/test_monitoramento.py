import numpy as np
import pandas as pd
import pytest

from vgpricing import monitoramento as Mo


def test_limites_poisson_cobrem_o_nivel():
    rng = np.random.default_rng(0)
    for E in (20.0, 200.0):
        inf, sup = Mo.limites_poisson(E, 0.95)
        ae = rng.poisson(E, 20000) / E
        assert 0.94 <= np.mean((ae >= inf) & (ae <= sup)) <= 0.99
    inf, sup = Mo.limites_poisson([25.0, 2500.0], 0.998)
    assert (sup[1] - inf[1]) < (sup[0] - inf[0])                  # o funil fecha com o volume


def _serie(fator=1.0, n_grupos=200, meses=24, semente=1):
    rng = np.random.default_rng(semente)
    mes = pd.period_range("2024-01", periods=meses, freq="M")
    linhas = [{"id_estipulante": f"G{g}", "mes": m, "E_manual": E} for g, E in
              zip(range(n_grupos), rng.uniform(0.05, 1.0, n_grupos)) for m in mes]
    d = pd.DataFrame(linhas)
    d["N"] = rng.poisson(d["E_manual"] * fator)
    d["N_ajustado"] = d["N"].astype(float)
    return d


@pytest.mark.parametrize("fator,dispara", [(1.0, False), (1.3, True)])
def test_gatilho_da_carteira(fator, dispara):
    cfg = {"janela_meses": 12, "niveis": [0.95, 0.998],
           "gatilhos": {"grupos_acima_998_maximo": 0.01, "idade_maxima_experiencia_meses": 24}}
    s = _serie(fator)
    g = Mo.gatilhos(Mo.carteira_mensal(s), Mo.funil(s), {"corte_experiencia": "2024-12-31"}, "2025-12-31", cfg).set_index("gatilho")
    assert g.iloc[0]["disparou"] == dispara
    assert not g.loc["idade da experiência que calibrou a tabela", "disparou"]


def test_funil_com_heterogeneidade_entre_grupos_nao_dispara_sozinho():
    """Grupos com mortalidade própria (Gama, variância 0,1) e taxa média certa: com a variância entre grupos nos
    limites, a fração acima de 99,8% fica perto do nominal; com Poisson puro, explodiria."""
    rng = np.random.default_rng(3)
    s = _serie(1.0, n_grupos=600)
    u = pd.Series(rng.gamma(10, 0.1, 600), index=[f"G{g}" for g in range(600)])
    s["N_ajustado"] = rng.poisson(s["E_manual"] * s["id_estipulante"].map(u)).astype(float)
    f = Mo.funil(s, meses_imaturos=0)
    assert 0.05 < f.attrs["tau2"] < 0.2
    assert f["acima 99.8%"].mean() < 0.01
    inf, sup = Mo.limites_poisson(f["E_manual"], 0.998)
    assert (f["A/E"] > sup).mean() > f["acima 99.8%"].mean()
