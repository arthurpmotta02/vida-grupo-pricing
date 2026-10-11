"""Testes do GLMM no R. Rodam só onde o R com glmmTMB/lme4 estiver instalado."""
import numpy as np
import pandas as pd
import pytest

from vgpricing import glmm as G

pytestmark = pytest.mark.skipif(not G.r_disponivel(), reason="R com glmmTMB/lme4 não disponível")


def _simular(n_grupos=500, sd_grupo=0.3, efeito=(1.0, 1.4, 0.8), beta_cap=-0.1, efeito_ruido=1.0, semente=7):
    rng = np.random.default_rng(semente)
    linhas = []
    for g in range(n_grupos):
        s = ["A", "B", "C"][g % 3]
        u = np.exp(rng.normal(0, sd_grupo))
        for r in ("x", "y"):
            E, lc = rng.uniform(1, 6), rng.normal(0, 1)
            mu = E * efeito[g % 3] * np.exp(beta_cap * lc) * u * (efeito_ruido if r == "y" else 1.0)
            linhas.append({"id_estipulante": f"G{g}", "cnae_secao": s, "cnae_divisao": f"{s}{g % 2}", "ruido": r,
                           "log_capital_c": lc, "E": E, "N": rng.poisson(mu)})
    return pd.DataFrame(linhas)


def _ajustar(df, fixos, testar=(), lme4=False):
    return G.ajustar_glmm(df, fixos, ["(1|id_estipulante)"], categoricas=["id_estipulante", "cnae_secao", "ruido"],
                          numericas=["log_capital_c"], niveis_base={"cnae_secao": "A", "ruido": "x"},
                          testar_fixos=list(testar), conferir_lme4=lme4)


def test_recupera_parametros_e_confere_com_lme4():
    df = _simular()
    r = _ajustar(df, ["log_capital_c", "cnae_secao"], lme4=True)
    assert r.convergiu
    b = r.fixos.set_index("nome")
    for nome, verdade in {"log_capital_c": -0.1, "cnae_secaoB": np.log(1.4), "cnae_secaoC": np.log(0.8)}.items():
        assert abs(b.loc[nome, "estimativa"] - verdade) < 3 * b.loc[nome, "ep"], nome
    assert 0.03 < r.variancias["id_estipulante"] < 0.2                      # verdade 0,09
    lme = dict(zip(r.lme4["fixos"]["nome"], r.lme4["fixos"]["estimativa"]))
    assert all(abs(lme[n] - b.loc[n, "estimativa"]) < 1e-3 for n in b.index)


def test_razao_de_verossimilhanca_detecta_efeito_real_e_nao_o_nulo():
    com = _ajustar(_simular(efeito_ruido=1.3), ["log_capital_c", "ruido"], testar=["ruido"])
    sem = _ajustar(_simular(efeito_ruido=1.0), ["log_capital_c", "ruido"], testar=["ruido"])
    assert com.testes.loc[0, "p_valor"] < 0.001
    assert sem.testes.loc[0, "p_valor"] > 0.01


def test_prever_bate_com_total_observado():
    df = _simular()
    r = _ajustar(df, ["log_capital_c", "cnae_secao"])
    assert G.prever(r, df).sum() / df["N"].sum() == pytest.approx(1.0, abs=0.05)
