import numpy as np
import pandas as pd
import pytest

from vgpricing import diagnostico as D
from vgpricing.config import carregar_config


def _base(efeito_regiao=1.0, n_grupos=300, semente=0):
    """Linhas com esperado pela manual conhecido; opcionalmente um efeito plantado nas idades >= 55."""
    rng = np.random.default_rng(semente)
    linhas = []
    for g in range(n_grupos):
        sec = ["C", "F", "N"][g % 3]
        for idade in range(20, 70, 5):
            for sexo in ("M", "F"):
                mu = rng.uniform(0.05, 0.6) * (1 + (idade - 20) / 20)
                linhas.append({"id_estipulante": f"G{g}", "cnae_secao": sec, "cnae_divisao": sec + "1", "idade": idade + 2,
                               "sexo": sexo, "log_capital": rng.normal(11, 0.5), "tipo_capital": "uniforme",
                               "contributario": "False", "log_porte": 5.0, "ano_calendario": 2023, "mu_manual": mu,
                               "E": mu})
    df = pd.DataFrame(linhas)
    df["N"] = rng.poisson(df["mu_manual"] * np.where(df["idade"] >= 55, efeito_regiao, 1.0))
    return df


@pytest.fixture(scope="module")
def cd():
    return carregar_config("diagnostico")


def test_sem_estrutura_faltante_nao_ha_melhora(cd):
    r = D.ajustar_residual(_base(1.0), cd)
    assert not r["comparacao"]["melhora_significativa"]


def test_estrutura_grande_e_detectada_e_shap_aponta_a_idade(cd):
    df = _base(1.6)
    r = D.ajustar_residual(df, cd)
    assert r["comparacao"]["melhora_significativa"]
    imp = D.importancia(D.contribuicoes(r["modelo"], r["X"]), df["mu_manual"].to_numpy())
    assert imp.index[0] == "idade"


def test_contribuicoes_somam_o_ajuste(cd):
    df = _base(1.6, n_grupos=60)
    r = D.ajustar_residual(df, cd)
    c = D.contribuicoes(r["modelo"], r["X"])
    assert np.allclose(c.sum(axis=1), r["modelo"].predict(r["X"], raw_score=True))


def test_prever_usa_a_codificacao_do_treino(cd):
    df = _base(1.6, n_grupos=60)
    r = D.ajustar_residual(df, cd)
    # teste com as categorias em outra ordem de aparição: a previsão tem de ser a mesma
    invertido = df.iloc[::-1].reset_index(drop=True)
    assert np.allclose(D.prever(r, invertido)[::-1], D.prever(r, df))


@pytest.mark.filterwarnings("ignore:Efeitos aleatórios com variância nula")   # dados sem heterogeneidade, de propósito
def test_hipotese_dirigida_no_glmm():
    from vgpricing.glmm import r_disponivel
    if not r_disponivel():
        pytest.skip("R não disponível")
    df = _base(1.0, n_grupos=150)
    regiao = (df["cnae_secao"] == "F") & (df["idade"] >= 50)
    df["N"] = np.random.default_rng(5).poisson(df["mu_manual"] * np.where(regiao, 1.6, 1.0))
    df["F_50mais"] = np.where(regiao, "sim", "nao")
    cm = carregar_config("manual")
    t = D.testar_candidato_glmm(df, cm, ["log_capital_c", "cnae_secao"], ["F_50mais"], categoricas_extras=["F_50mais"])
    assert t.loc[0, "p_valor"] < 0.001


def test_tabela_interacoes_aponta_a_interacao_plantada(cd):
    """Efeito só em seção F e idade >= 50: a interação cnae_secao:idade deve liderar a tabela."""
    df = _base(1.0, n_grupos=300)
    regiao = (df["cnae_secao"] == "F") & (df["idade"] >= 50)
    df["N"] = np.random.default_rng(5).poisson(df["mu_manual"] * np.where(regiao, 1.8, 1.0))
    r = D.ajustar_residual(df, {**cd, "variaveis": ["cnae_secao", "idade", "sexo", "log_capital"],
                                "categoricas": ["cnae_secao", "sexo"]})
    t = D.tabela_interacoes(r["modelo"])
    assert t.loc[0, "interacao"] == "cnae_secao:idade"
    assert {"sumGain", "sumCover", "frequency", "meanCover", "meanGain"} <= set(t.columns)
