import numpy as np
import pandas as pd
import pytest

from vgpricing import aglm
from vgpricing.glmm import r_disponivel


def test_variaveis_O_e_fatores():
    d = aglm.variaveis_O(pd.Series(["a", "b", "c"]), ["a", "b", "c"], "f")
    assert d.to_numpy().tolist() == [[1, 1], [0, 1], [0, 0]]
    fat = aglm.fatores_O(["a", "b", "c"], {"f_O1": 0.2, "f_O2": -0.1}, "f")
    assert fat["c"] == pytest.approx(1.0) and fat["b"] == pytest.approx(np.exp(-0.1)) and fat["a"] == pytest.approx(np.exp(0.1))


def test_variaveis_L_formam_curva_linear_por_partes():
    x = np.linspace(0, 10, 101)
    ef = aglm.efeito_L(x, [5.0], {"x_L0": 0.5}, "x")          # mudança de inclinação de 1 em x = 5
    assert ef[0] == pytest.approx(2.5) and ef[50] == pytest.approx(0.0) and ef[-1] == pytest.approx(2.5)


@pytest.mark.skipif(not r_disponivel(), reason="R não disponível")
def test_aglm_e_gam_recuperam_curva_com_dobra():
    """Mortalidade cai com o log do capital até 0 e depois fica plana: AGLM (L + LASSO) e GAM captam a dobra; um
    GLM log-linear não."""
    from vgpricing.elasticnet import ajustar_elastic_net
    from vgpricing.gam import ajustar_gam
    rng = np.random.default_rng(0)
    n = 6000
    lc = rng.uniform(-3, 3, n)
    E = rng.uniform(0.5, 2, n)
    verdade = np.where(lc < 0, -0.3 * lc, 0.0)
    d = pd.DataFrame({"lc": lc, "E": E, "N": rng.poisson(E * np.exp(verdade)), "log_E": np.log(E), "g": "x"})
    nos = aglm.nos_por_quantil(d["lc"], 9)
    X = pd.concat([d, aglm.codificar(d, {"lc": nos}, {})], axis=1)
    cols = [c for c in X if c.startswith("lc_L")]
    r = ajustar_elastic_net(X, {}, ["lc"] + cols, [], ["lc"] + cols, "log_E", alpha=1.0, nfolds=5)
    b = r.coeficientes
    ef = b.get("lc", 0) * lc + aglm.efeito_L(lc, nos, b.to_dict(), "lc")
    ef = ef - ef[np.argmin(np.abs(lc - 2))]
    assert np.corrcoef(ef, verdade)[0, 1] > 0.95
    g = ajustar_gam(d, {"treino": d}, ["lc"], [], k=6)
    assert np.corrcoef(np.log(g["previsoes"]["treino"] / E), verdade)[0, 1] > 0.95
