"""Testes da taxa manual. Os modelos são testados em dados simulados direto nas células
(rápido e com parâmetros conhecidos), sem passar pelo gerador de carteira."""
import numpy as np
import pandas as pd
import pytest

from vgpricing import manual as M
from vgpricing.config import carregar_config

T = pd.Timestamp


def _celulas(n_grupos=300, k=8.0, efeito_setor=(1.0, 1.4, 0.8), semente=1):
    """Grupos com efeito Gama(média 1, variância 1/k); 3 setores; fator 'ruido' sem efeito."""
    rng = np.random.default_rng(semente)
    linhas = []
    for g in range(n_grupos):
        setor = ["s0", "s1", "s2"][g % 3]
        u = rng.gamma(k, 1 / k)
        for ruido in ("a", "b"):
            E = rng.uniform(0.5, 6.0)
            mu = E * efeito_setor[int(setor[1])] * u
            linhas.append({"id_estipulante": f"G{g}", "setor": setor, "ruido": ruido, "E": E,
                           "N": rng.poisson(mu)})
    return pd.DataFrame(linhas)


def test_gradiente_poisson_gama_confere_com_diferencas_finitas():
    df = _celulas(n_grupos=30)
    d = M.Desenho.criar(df, ["setor"])
    X, N, logE = d.matriz(df), df["N"].to_numpy(float), np.log(df["E"].to_numpy())
    cod, grupos = pd.factorize(df["id_estipulante"])
    th = np.array([0.1, 0.2, -0.1, np.log(5.0)])
    f, g = M._nll_poisson_gama(th, X, N, logE, cod, len(grupos))
    h = 1e-6
    num = [(M._nll_poisson_gama(th + h * np.eye(4)[j], X, N, logE, cod, len(grupos))[0] -
            M._nll_poisson_gama(th - h * np.eye(4)[j], X, N, logE, cod, len(grupos))[0]) / (2 * h) for j in range(4)]
    assert np.allclose(g, num, rtol=1e-5, atol=1e-6)


def test_poisson_gama_recupera_k_e_fatores():
    df = _celulas(n_grupos=600, k=8.0)
    r = M.ajustar_poisson_gama(df, ["setor"])
    lo, hi = r.extra["ic_k"]
    assert lo <= 8.0 <= hi
    f = r.fatores().set_index("nivel")["relatividade"]
    rel = f / f["s0"]
    assert rel["s1"] == pytest.approx(1.4, rel=0.15) and rel["s2"] == pytest.approx(0.8, rel=0.15)


def test_credibilidade_do_modelo_e_formula_de_buhlmann():
    df = _celulas(n_grupos=200)
    r = M.ajustar_poisson_gama(df, ["setor"])
    g = r.extra["grupos"]
    k = r.extra["k"]
    esperado = g["Z"] * g["N"] / g["M"] + (1 - g["Z"])
    assert np.allclose(g["fator_grupo_posteriori"], esperado)
    assert np.allclose(g["Z"], g["M"] / (g["M"] + k))


def test_sem_heterogeneidade_poisson_gama_igual_glm():
    df = _celulas(n_grupos=400, k=1e6)
    glm = M.ajustar_glm(df, ["setor"])
    pg = M.ajustar_poisson_gama(df, ["setor"], beta_inicial=glm.beta)
    assert np.allclose(glm.beta, pg.beta, atol=0.02)
    assert pg.extra["k"] > 100


def test_ridge_lambda_zero_e_glm_e_lambda_alto_encolhe():
    df = _celulas()
    glm = M.ajustar_glm(df, ["setor"])
    r0 = M.ajustar_ridge(df, ["setor"], 0.0)
    assert np.allclose(glm.beta, r0.beta)
    alto = M.ajustar_ridge(df, ["setor"], 1e6).fatores()
    assert np.allclose(alto["relatividade"], 1.0, atol=1e-3)


def test_validacao_cruzada_nao_separa_grupo():
    df = _celulas(n_grupos=50)
    fold = M._folds_por_grupo(df, 5, 0)
    assert df.assign(f=fold).groupby("id_estipulante")["f"].nunique().max() == 1


def test_selecao_retira_fator_sem_efeito_e_mantem_setor():
    df = _celulas(n_grupos=600, k=8.0)
    sel, _ = M.selecionar_fatores(df, ["setor", "ruido"])
    assert sel == ["setor"]


def test_glm_detecta_sobredispersao_que_o_modelo_de_grupo_explica():
    df = _celulas(n_grupos=600, k=3.0)
    glm = M.ajustar_glm(df, ["setor"])
    nb = M.teste_binomial_negativa(df, ["setor"], glm)
    assert nb["p_valor"] < 0.01


def test_taxa_pura_manual_golden(tabua_constante):
    cm = carregar_config("manual")
    tabela = {"meta": {"ae_base": 0.9}, "fatores": {"setor": {"industria": 1.2, "servicos": 1.0}}}
    vidas = pd.DataFrame({"data_nascimento": [T("1980-01-01"), T("1990-01-01")], "sexo": ["M", "F"],
                          "capital": [100000.0, 50000.0]})
    r = M.taxa_pura_manual(vidas, "industria", tabela, tabua_constante, "2026-01-01", cm)
    # custo = 0,9 x 1,2 x (100.000 x 0,001 + 50.000 x 0,002) = 216 -> 216/150.000 x 1000 = 1,44 por mil/ano
    assert r["premio_puro_anual"] == pytest.approx(216.0)
    assert r["taxa_pura_anual_mil"] == pytest.approx(1.44)
    assert r["taxa_pura_mensal_mil"] == pytest.approx(0.12)
    assert r["niveis_sem_fator"] == {}


def test_taxa_pura_sinaliza_nivel_sem_fator(tabua_constante):
    cm = carregar_config("manual")
    tabela = {"meta": {"ae_base": 1.0}, "fatores": {"setor": {"servicos": 1.0}}}
    vidas = pd.DataFrame({"data_nascimento": [T("1980-01-01")], "sexo": ["M"], "capital": [1000.0]})
    r = M.taxa_pura_manual(vidas, "mineracao", tabela, tabua_constante, "2026-01-01", cm)
    assert r["niveis_sem_fator"] == {"setor": ["mineracao"]}


def test_base_modelagem_respeita_maturacao_e_celula_do_obito(tabua_constante):
    cm = carregar_config("manual")
    est = pd.DataFrame({"id_estipulante": ["E1"], "setor": ["servicos"], "data_inicio_vigencia": [T("2022-01-01")],
                        "data_fim_vigencia": [pd.NaT]})
    vidas = pd.DataFrame({"id_linha": [1, 2], "id_estipulante": "E1", "id_vida": ["A", "B"],
                          "data_nascimento": [T("1990-06-01"), T("1960-06-01")], "sexo": ["M", "F"],
                          "data_inicio": T("2022-01-01"), "data_fim": [T("2023-03-10"), T("2025-08-01")],
                          "capital": [20000.0, 500000.0]})
    sin = pd.DataFrame({"id_sinistro": ["S1", "S2"], "id_estipulante": "E1", "id_vida": ["A", "B"],
                        "data_ocorrencia": [T("2023-03-10"), T("2025-08-01")]})
    cel, info = M.base_modelagem(vidas, sin, est, tabua_constante, "2025-12-31", cm)
    assert info["corte_experiencia"] == "2024-12-31"
    assert info["sinistros_no_periodo"] == 1                      # S2 é posterior ao corte
    linha = cel[cel["N"] == 1].iloc[0]
    assert (linha["faixa_idade"], linha["sexo"], linha["faixa_capital"]) == ("30-39", "M", "ate30k")


def test_tabela_salva_e_recarrega_igual(tmp_path):
    df = _celulas(n_grupos=120)
    r = M.ajustar_poisson_gama(df, ["setor"])
    tab = M.tabela_fatores(r, {"k_ic95": list(r.extra["ic_k"]), "faixas": {"x": {"limites": [0, np.inf]}}})
    M.salvar_tabela(tab, tmp_path / "f.yaml")
    lida = M.carregar_tabela(tmp_path / "f.yaml")
    assert lida["fatores"]["setor"] == pytest.approx(tab["fatores"]["setor"])
    assert lida["meta"]["ae_base"] == pytest.approx(tab["meta"]["ae_base"])


def _celulas_com_capital(n_grupos=300, k=8.0, semente=3):
    df = _celulas(n_grupos=n_grupos, k=k, semente=semente)
    df["faixa_capital"] = np.where(df["ruido"] == "a", "baixo", "alto")
    return df


def test_encolhimento_lambda_zero_igual_sem_penalizacao_e_alto_leva_a_um():
    df = _celulas_com_capital()
    livre = M.ajustar_poisson_gama(df, ["setor", "faixa_capital"])
    zero = M.ajustar_poisson_gama(df, ["setor", "faixa_capital"], penalizacao={"faixa_capital": 0.0})
    assert np.allclose(livre.beta, zero.beta, atol=1e-6)
    forte = M.ajustar_poisson_gama(df, ["setor", "faixa_capital"], penalizacao={"faixa_capital": 1e7}).fatores()
    cap = forte[forte["fator"] == "faixa_capital"]["relatividade"]
    assert np.allclose(cap, 1.0, atol=1e-3)
    setor = forte[forte["fator"] == "setor"].set_index("nivel")["relatividade"]
    assert setor["s1"] / setor["s0"] > 1.2                     # o outro fator não é afetado


def test_loglik_completa_coincide_com_statsmodels_sem_heterogeneidade():
    df = _celulas(n_grupos=300, k=1e7)
    glm = M.ajustar_glm(df, ["setor"])
    pg = M.ajustar_poisson_gama(df, ["setor"], beta_inicial=glm.beta)
    assert pg.extra["loglik_completa"] == pytest.approx(glm.extra["loglik"], abs=0.05)


def test_comparar_dispersao_prefere_efeito_de_grupo_quando_ele_existe():
    df = _celulas(n_grupos=500, k=3.0)
    tab = M.comparar_dispersao(df, ["setor"]).set_index("modelo")
    assert tab["AIC"].idxmin() == "Poisson-Gama"
    assert tab.loc["Poisson", "AIC"] > tab.loc["Binomial Negativa (NB2)", "AIC"]


@pytest.mark.filterwarnings("ignore:lambda de")
def test_calibrar_encolhimento_devolve_lambda_por_fator():
    df = _celulas_com_capital(n_grupos=150)
    lam, hist = M.calibrar_encolhimento(df, ["setor", "faixa_capital"], ["setor", "faixa_capital"],
                                        [0.0, 10, 1000], k=3)
    assert set(lam) == {"setor", "faixa_capital"}
    assert set(hist["fator"]) == {"setor", "faixa_capital"}


def test_subsidio_cruzado_zero_quando_estrutura_e_exata():
    df = _celulas(n_grupos=60, k=1e7)
    df["valor_esperado"] = df["E"] * 1000.0
    r = M.ajustar_poisson_gama(df, ["setor"])
    rel = r.fatores().set_index("nivel")["relatividade"]
    verdade = (df["valor_esperado"] * df["setor"].map(rel)).groupby(df["id_estipulante"]).sum()
    assert M.subsidio_cruzado(df, r, verdade) == pytest.approx(0.0, abs=1e-10)
