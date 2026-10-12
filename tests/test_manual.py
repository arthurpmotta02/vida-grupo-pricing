"""Testes da taxa manual que não dependem do R (os do GLMM ficam em test_glmm.py)."""
import numpy as np
import pandas as pd
import pytest

from vgpricing import manual as M
from vgpricing.config import carregar_config
from vgpricing.glmm import ResultadoGLMM, prever

T = pd.Timestamp


@pytest.fixture(scope="module")
def cm():
    return carregar_config("manual")


def _resultado_falso():
    """ResultadoGLMM montado à mão: permite testar previsão e tabela sem o R."""
    fixos = pd.DataFrame({"nome": ["(Intercept)", "log_capital_c", "cnae_secaoF", "sexoM"],
                          "estimativa": [np.log(0.9), -0.1, np.log(1.3), np.log(1.1)], "ep": 0.0})
    aleat = {"cnae_secao:cnae_divisao": pd.DataFrame({"nivel": ["F:41"], "modo": [np.log(1.2)], "sd_condicional": 0.1}),
             "id_estipulante": pd.DataFrame({"nivel": ["G1"], "modo": [np.log(1.5)], "sd_condicional": 0.2})}
    spec = {"fixos": ["log_capital_c", "cnae_secao", "sexo"], "categoricas": ["id_estipulante", "cnae_secao",
            "cnae_divisao", "sexo"], "niveis_base": {"cnae_secao": "N", "sexo": "F"}}
    return ResultadoGLMM(formula="falso", fixos=fixos, vcov=np.eye(4), variancias={"cnae_secao:cnae_divisao": 0.04,
                         "id_estipulante": 0.10}, aleatorios=aleat, loglik=0.0, aic=0.0, gl=6, convergiu=True,
                         testes=pd.DataFrame(), especificacao=spec)


def test_prever_usa_modo_conhecido_e_media_marginal_para_desconhecido():
    r = _resultado_falso()
    df = pd.DataFrame({"E": [1.0, 1.0], "log_capital_c": [0.0, 0.0], "cnae_secao": ["F", "F"],
                       "cnae_divisao": ["41", "42"], "sexo": ["M", "F"], "id_estipulante": ["G1", "G9"]})
    mu = prever(r, df)                                  # grupo novo: efeito de estipulante marginal
    assert mu[0] == pytest.approx(0.9 * 1.3 * 1.1 * 1.2 * np.exp(0.05))
    assert mu[1] == pytest.approx(0.9 * 1.3 * np.exp(0.02) * np.exp(0.05))   # divisão 42 não vista
    mu_g = prever(r, df, usar_efeito_grupo=True)
    assert mu_g[0] == pytest.approx(0.9 * 1.3 * 1.1 * 1.2 * 1.5)


def test_tabela_fatores_e_taxa_pura_coerentes_com_prever(tabua_constante, cm):
    r = _resultado_falso()
    cfg = {**cm, "modelo": {**cm["modelo"], "fixos_candidatos": ["sexo"]}}
    tab = M.tabela_fatores(r, cfg, {"esperados_por_secao": {"N": 10.0, "F": 10.0}})
    assert tab["ae_base"] == pytest.approx(0.9 * np.exp(0.05))
    assert tab["secao"] == pytest.approx({"N": 1.0, "F": 1.3})
    assert tab["secao_nao_vista"] == pytest.approx(1.15)          # média ponderada das seções
    assert tab["divisao"]["F:41"] == pytest.approx(1.2)
    assert tab["fatores_categoricos"]["sexo"] == pytest.approx({"F": 1.0, "M": 1.1})
    vidas = pd.DataFrame({"data_nascimento": [T("1980-01-01")], "sexo": ["M"], "capital": [50000.0]})
    cot = M.taxa_pura_manual(vidas, "F", "41", tab, tabua_constante, "2026-01-01")
    # capital = referência -> fator 1; q = 0,001 (M)
    esperado = 50000 * 0.001 * 0.9 * np.exp(0.05) * 1.3 * 1.2 * 1.1
    assert cot["premio_puro_anual"] == pytest.approx(esperado)
    assert cot["taxa_pura_mensal_mil"] == pytest.approx(1000 * esperado / 50000 / 12)
    assert cot["avisos"] == []


def test_taxa_pura_avisa_atividade_sem_experiencia(tabua_constante, cm):
    tab = M.tabela_fatores(_resultado_falso(), {**cm, "modelo": {**cm["modelo"], "fixos_candidatos": []}}, {})
    vidas = pd.DataFrame({"data_nascimento": [T("1980-01-01")], "sexo": ["F"], "capital": [100000.0]})
    cot = M.taxa_pura_manual(vidas, "A", "01", tab, tabua_constante, "2026-01-01")
    assert len(cot["avisos"]) == 2
    assert cot["fator_atividade"] == pytest.approx(tab["secao_nao_vista"] * tab["divisao_nao_vista"])


def test_fator_de_capital_segue_a_elasticidade(tabua_constante, cm):
    tab = M.tabela_fatores(_resultado_falso(), {**cm, "modelo": {**cm["modelo"], "fixos_candidatos": []}}, {})
    vidas = pd.DataFrame({"data_nascimento": [T("1980-01-01")] * 2, "sexo": ["F"] * 2,
                          "capital": [50000.0, 500000.0]})
    d = M.taxa_pura_manual(vidas, "N", "81", tab, tabua_constante, "2026-01-01")["detalhe"]
    assert d["q_ajustado"].iloc[1] / d["q_ajustado"].iloc[0] == pytest.approx(10 ** -0.1)


def test_holm():
    p = pd.Series({"a": 0.01, "b": 0.04, "c": 0.03})
    aj = M.holm(p)
    assert aj["a"] == pytest.approx(0.03) and aj["c"] == pytest.approx(0.06) and aj["b"] == pytest.approx(0.06)


def test_tabela_salva_e_recarrega(tmp_path, cm):
    tab = M.tabela_fatores(_resultado_falso(), {**cm, "modelo": {**cm["modelo"], "fixos_candidatos": ["sexo"]}}, {})
    M.salvar_tabela(tab, tmp_path / "f.yaml")
    lida = M.carregar_tabela(tmp_path / "f.yaml")
    assert lida["secao"] == pytest.approx(tab["secao"])
    assert lida["faixas"]["capital"]["limites"][-1] == float("inf")


def test_base_modelagem(tabua_constante, cm):
    est = pd.DataFrame({"id_estipulante": ["E1", "E2"], "cnae_secao": ["N", "T"], "cnae_divisao": ["81", None],
                        "data_inicio_vigencia": [T("2022-01-01")] * 2, "data_fim_vigencia": [pd.NaT] * 2})
    vidas = pd.DataFrame({"id_linha": [1, 2, 3], "id_estipulante": ["E1", "E1", "E2"], "id_vida": ["A", "B", "C"],
                          "data_nascimento": [T("1990-06-01"), T("1960-06-01"), T("1970-01-01")],
                          "sexo": ["M", "F", "M"], "data_inicio": T("2022-01-01"),
                          "data_fim": [T("2023-03-10"), T("2025-08-01"), pd.NaT],
                          "capital": [20000.0, 500000.0, 50000.0]})
    sin = pd.DataFrame({"id_sinistro": ["S1", "S2"], "id_estipulante": "E1", "id_vida": ["A", "B"],
                        "data_ocorrencia": [T("2023-03-10"), T("2025-08-01")]})
    cel, info = M.base_modelagem(vidas, sin, est, tabua_constante, "2025-12-31", cm)
    assert info["corte_experiencia"] == "2024-12-31"
    assert info["sinistros_no_periodo"] == 1                          # S2 é posterior ao corte
    linha = cel[cel["N"] == 1].iloc[0]
    assert (linha["faixa_idade"], linha["sexo"], linha["faixa_capital"]) == ("30-39", "M", "ate30k")
    assert linha["log_capital_c"] == pytest.approx(np.log(20000 / cm["capital_referencia"]))
    assert set(cel["cnae_secao"]) == {"OUTRAS"}                        # exposição pequena: seções agrupadas
    assert set(cel.loc[cel["id_estipulante"] == "E2", "cnae_divisao"]) == {"ND"}


def test_base_modelagem_janela_fora_do_tempo(tabua_constante, cm):
    est = pd.DataFrame({"id_estipulante": ["E1"], "cnae_secao": ["N"], "cnae_divisao": ["81"],
                        "data_inicio_vigencia": [T("2022-01-01")], "data_fim_vigencia": [pd.NaT]})
    vidas = pd.DataFrame({"id_linha": [1], "id_estipulante": ["E1"], "id_vida": ["A"], "data_nascimento": [T("1990-06-01")],
                          "sexo": ["M"], "data_inicio": [T("2022-01-01")], "data_fim": [pd.NaT], "capital": [20000.0]})
    sin = pd.DataFrame({"id_sinistro": ["S1", "S2"], "id_estipulante": "E1", "id_vida": ["A", "A"],
                        "data_ocorrencia": [T("2022-05-01"), T("2024-03-01")]})
    treino, i1 = M.base_modelagem(vidas, sin, est, tabua_constante, "2025-12-31", cm, janela=(None, "2023-12-31"))
    teste, i2 = M.base_modelagem(vidas, sin, est, tabua_constante, "2025-12-31", cm, janela=("2024-01-01", "2024-12-31"))
    assert i1["sinistros_no_periodo"] == 1 and i2["sinistros_no_periodo"] == 1
    assert treino["exposicao"].sum() == pytest.approx(730 / 365.25) and teste["exposicao"].sum() == pytest.approx(366 / 365.25)
    with pytest.raises(ValueError, match="maturada"):
        M.base_modelagem(vidas, sin, est, tabua_constante, "2025-12-31", cm, janela=("2025-01-01", "2025-12-31"))


def test_tabela_ridge_reproduz_a_previsao(tabua_constante, cm):
    """A tabela de fatores da ridge, aplicada pelo multiplicador do cotador, dá a mesma mortalidade que a
    fórmula do modelo: E x exp(b0 + (beta + delta) x log_cap + log f_priori + gamma)."""
    b = pd.Series({"(Intercept)": np.log(0.8), "log_capital_c": 0.02, "cnae_secaoC": 0.10, "cnae_secaoK": -0.05})
    pri = {"elasticidade_capital": -0.10, "fator_secao": {"C": 1.2, "K": 0.9, "F": 1.3}}
    cel = pd.DataFrame({"cnae_secao": ["C", "K"], "E": [10.0, 5.0]})
    tab = M.tabela_fatores_ridge(b, pri, cel, cm, {})
    for s_, gam in (("C", 0.10), ("K", -0.05)):
        df = pd.DataFrame({"capital": [100000.0], "cnae_secao": [s_], "cnae_divisao": ["x"]})
        mult, _ = M.multiplicador_manual(df, tab)
        lc = np.log(100000 / cm["capital_referencia"])
        esperado = np.exp(np.log(0.8) + (-0.10 + 0.02) * lc + np.log(pri["fator_secao"][s_]) + gam)
        assert mult[0] == pytest.approx(esperado)
    assert tab["secao"]["F"] == pytest.approx(1.3 / (1.2 * np.exp(0.10)))      # seção só da priori


def test_escolher_modelo():
    c = pd.DataFrame({"modelo": ["a", "b", "c"], "deviance": [10.0, 11.0, 30.0], "significativa": [False, False, True]})
    assert M.escolher_modelo(c, ["c", "b", "a"]) == ("b", ["a", "b"])


def test_ajustar_tabela_plano_atual_e_metodo_invalido(cm):
    cel = pd.DataFrame({"id_estipulante": ["A", "B"], "cnae_secao": ["C", "K"], "E": [10.0, 10.0], "N": [12, 10]})
    tab = M.ajustar_tabela(cel, "plano atual (tábua × A/E)", cm, None, {})
    assert tab["ae_base"] == pytest.approx(1.1) and tab["meta"]["modelo_escolhido"].startswith("plano")
    df = pd.DataFrame({"capital": [1e6], "cnae_secao": ["Q"], "cnae_divisao": ["x"]})
    assert M.multiplicador_manual(df, tab)[0][0] == pytest.approx(1.1)
    with pytest.raises(ValueError, match="não pode virar tarifa"):
        M.ajustar_tabela(cel, "LightGBM", cm, None, {})
