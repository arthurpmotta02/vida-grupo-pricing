"""Testes do glmmTMB no R. Rodam só onde o R com os pacotes do projeto estiver instalado."""
import numpy as np
import pandas as pd
import pytest

from vgpricing import glmm as G

pytestmark = pytest.mark.skipif(not G.r_disponivel(), reason="R com glmmTMB não disponível")


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


def _ajustar(df, fixos, testar=(), aleatorios=("(1|id_estipulante)",), familia="poisson"):
    return G.ajustar_glmm(df, fixos, list(aleatorios), categoricas=["id_estipulante", "cnae_secao", "ruido"],
                          numericas=["log_capital_c"], niveis_base={"cnae_secao": "A", "ruido": "x"},
                          testar_fixos=list(testar), familia=familia)


def test_recupera_parametros():
    df = _simular()
    r = _ajustar(df, ["log_capital_c", "cnae_secao"])
    assert r.convergiu
    b = r.fixos.set_index("nome")
    for nome, verdade in {"log_capital_c": -0.1, "cnae_secaoB": np.log(1.4), "cnae_secaoC": np.log(0.8)}.items():
        assert abs(b.loc[nome, "estimativa"] - verdade) < 3 * b.loc[nome, "ep"], nome
    assert 0.03 < r.variancias["id_estipulante"] < 0.2                      # verdade 0,09
    ic = r.ic_variancias.set_index("termo").loc["id_estipulante"]
    assert ic["var_inf"] < r.variancias["id_estipulante"] < ic["var_sup"]


def test_modelos_de_comparacao_no_mesmo_pacote():
    """Sem efeitos aleatórios: GLM Poisson e Binomial Negativa. Com heterogeneidade entre grupos,
    o GLMM tem log-verossimilhança maior que os dois."""
    df = _simular()
    glmm = _ajustar(df, ["log_capital_c", "cnae_secao"])
    p = _ajustar(df, ["log_capital_c", "cnae_secao"], aleatorios=())
    nb = _ajustar(df, ["log_capital_c", "cnae_secao"], aleatorios=(), familia="nbinom2")
    assert p.variancias == {} and nb.variancias == {}
    assert nb.loglik >= p.loglik - 1e-6
    assert glmm.loglik > nb.loglik


def test_razao_de_verossimilhanca_detecta_efeito_real_e_nao_o_nulo():
    com = _ajustar(_simular(efeito_ruido=1.3), ["log_capital_c", "ruido"], testar=["ruido"])
    sem = _ajustar(_simular(efeito_ruido=1.0), ["log_capital_c", "ruido"], testar=["ruido"])
    assert com.testes.loc[0, "p_valor"] < 0.001
    assert sem.testes.loc[0, "p_valor"] > 0.01


def test_prever_bate_com_total_observado():
    df = _simular()
    r = _ajustar(df, ["log_capital_c", "cnae_secao"])
    assert G.prever(r, df).sum() / df["N"].sum() == pytest.approx(1.0, abs=0.05)


def test_ajuste_singular_retira_efeito_com_variancia_nula():
    """Uma divisão por seção: o efeito de divisão fica confundido com o fixo da seção e tem variância nula.
    O termo é retirado e o modelo reajustado, com aviso, em vez de devolver NA (o que versões recentes do
    glmmTMB fazem no Windows). O resultado é o mesmo em qualquer versão."""
    df = _simular()
    df["cnae_divisao"] = df["cnae_secao"] + "1"
    with pytest.warns(UserWarning, match="variância nula"):
        r = G.ajustar_glmm(df, ["log_capital_c", "cnae_secao"], ["(1|cnae_secao:cnae_divisao)", "(1|id_estipulante)"],
                           categoricas=["id_estipulante", "cnae_secao", "cnae_divisao"], numericas=["log_capital_c"],
                           niveis_base={"cnae_secao": "A"}, testar_aleatorios=["(1|cnae_secao:cnae_divisao)"])
    assert r.extra["aleatorios_removidos"] == ["(1|cnae_secao:cnae_divisao)"]
    assert set(r.variancias) == {"id_estipulante"} and np.isfinite(r.loglik)
    assert r.testes.loc[0, "p_valor"] == 1.0                                 # sem evidência de heterogeneidade
    b = r.fixos.set_index("nome")["estimativa"]
    assert abs(b["cnae_secaoB"] - np.log(1.4)) < 0.1


def test_glm_sem_efeitos_aleatorios_usa_stats_glm_com_anova():
    df = _simular()
    r = G.ajustar_glmm(df, ["log_capital_c", "cnae_secao"], [], categoricas=["cnae_secao"], numericas=["log_capital_c"],
                       niveis_base={"cnae_secao": "A"}, anova=True)
    assert r.extra["motor"] == "stats::glm" and r.convergiu
    a = r.extra["anova"]
    assert list(a["termo"]) == ["NULL", "log_capital_c", "cnae_secao"]
    assert a["deviance"].iloc[1:].sum() == pytest.approx(r.extra["deviance_nula"] - r.extra["deviance"])


def test_fator_de_um_nivel_sai_do_glm():
    df = _simular().assign(cnae_secao="OUTRAS")
    with pytest.warns(UserWarning, match="único nível"):
        r = G.ajustar_glmm(df, ["log_capital_c", "cnae_secao"], [], categoricas=["cnae_secao"], numericas=["log_capital_c"])
    assert r.convergiu and list(r.fixos["nome"]) == ["(Intercept)", "log_capital_c"]
