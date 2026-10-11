"""Casos 'golden': resultados conferidos à mão."""
import numpy as np
import pandas as pd
import pytest

from vgpricing.exposicao import anos_apolice, experiencia, exposicao_detalhada, idade_completa

T = pd.Timestamp


def test_idade_completa():
    nasc = pd.Series([T("1980-06-15")] * 2 + [T("2000-02-29")] * 2)
    data = pd.Series([T("2023-06-14"), T("2023-06-15"), T("2023-02-28"), T("2023-03-01")])
    assert idade_completa(nasc, data).tolist() == [42, 43, 22, 23]


def _estip(inicio="2022-03-01"):
    return pd.DataFrame({"id_estipulante": ["E1"], "data_inicio_vigencia": [T(inicio)],
                         "data_fim_vigencia": [pd.NaT], "cnae_secao": ["N"], "cnae_divisao": ["81"]})


def test_anos_apolice_com_ano_parcial():
    py = anos_apolice(_estip(), "2023-12-31")
    assert len(py) == 2
    assert py.loc[0, "fim_ano"] == T("2023-02-28") and py.loc[0, "ano_completo"]
    assert py.loc[1, "inicio_ano"] == T("2023-03-01") and py.loc[1, "fim_ano"] == T("2023-12-31")
    assert not py.loc[1, "ano_completo"]


def _vidas(linhas):
    df = pd.DataFrame(linhas, columns=["id_vida", "data_nascimento", "sexo", "data_inicio", "data_fim", "capital"])
    df.insert(0, "id_estipulante", "E1")
    df.insert(0, "id_linha", range(1, len(df) + 1))
    for c in ("data_nascimento", "data_inicio", "data_fim"):
        df[c] = pd.to_datetime(df[c])
    return df


def test_exposicao_golden(tabua_constante):
    v = _vidas([["V1", "1980-06-15", "M", "2022-03-01", None, 100000.0]])
    det = exposicao_detalhada(v, _estip(), tabua_constante, "2023-12-31")
    assert det["dias"].tolist() == [365, 306]
    assert det["idade"].tolist() == [41, 42]
    assert det["exposicao"].sum() == pytest.approx((365 + 306) / 365.25)
    assert det["obitos_esperados"].sum() == pytest.approx(0.001 * 671 / 365.25)
    assert det["valor_esperado"].sum() == pytest.approx(100000 * 0.001 * 671 / 365.25)


def test_segmento_antes_da_vigencia_e_cortado(tabua_constante):
    v = _vidas([["V1", "1980-06-15", "F", "2021-01-01", "2022-03-31", 1.0]])
    det = exposicao_detalhada(v, _estip(), tabua_constante, "2023-12-31")
    assert det["dias"].sum() == 31          # só março/2022 conta


def test_numero_efetivo(tabua_constante):
    # duas vidas, mesmo q e exposição de um ano completo; capitais 1 e 100
    v = _vidas([["V1", "1980-01-01", "M", "2022-03-01", "2023-02-28", 1.0],
                ["V2", "1980-01-01", "M", "2022-03-01", "2023-02-28", 100.0]])
    det = exposicao_detalhada(v, _estip(), tabua_constante, "2023-02-28")
    xp = experiencia(det, None, _estip(), "2023-02-28")
    qt = 0.001 * 365 / 365.25
    assert xp.loc[0, "n_efetivo"] == pytest.approx(qt * 101 ** 2 / (1 + 100 ** 2))
    assert xp.loc[0, "obitos_esperados"] == pytest.approx(2 * qt)
    # a vida de capital 100 domina: n_efetivo ~ 1,02 x q, bem menor que 2 x q
    assert xp.loc[0, "n_efetivo"] < 0.52 * xp.loc[0, "obitos_esperados"]


def test_sinistro_vai_para_ano_da_ocorrencia(tabua_constante):
    v = _vidas([["V1", "1980-06-15", "M", "2022-03-01", "2023-05-10", 50000.0]])
    s = pd.DataFrame({"id_sinistro": ["S1"], "id_estipulante": ["E1"], "id_vida": ["V1"],
                      "cobertura": ["MQC"], "data_ocorrencia": [T("2023-05-10")],
                      "data_aviso": [T("2023-06-01")], "valor_indenizado": [50000.0], "valor_reserva": [0.0]})
    det = exposicao_detalhada(v, _estip(), tabua_constante, "2023-12-31")
    xp = experiencia(det, s, _estip(), "2023-12-31")
    assert xp["obitos_observados"].tolist() == [0, 1]
    assert xp.loc[1, "valor_observado"] == 50000.0
