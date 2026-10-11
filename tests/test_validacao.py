"""Cada regra testada com um caso mínimo construído à mão."""
import pandas as pd
import pytest

from vgpricing.validacao import validar

T = pd.Timestamp


def _base():
    est = pd.DataFrame({"id_linha": [2], "id_estipulante": ["E1"], "setor": ["servicos"],
                        "data_inicio_vigencia": [T("2022-01-01")], "data_fim_vigencia": [pd.NaT],
                        "tipo_capital": ["uniforme"], "contributario": [False], "taxa_vigente_mensal_mil": [0.3]})
    vidas = pd.DataFrame({
        "id_linha": [2, 3],
        "id_estipulante": ["E1", "E1"], "id_vida": ["A", "B"],
        "data_nascimento": [T("1980-01-01"), T("1990-01-01")], "sexo": ["M", "F"],
        "data_inicio": [T("2022-01-01")] * 2, "data_fim": [pd.NaT, pd.NaT],
        "capital": [50000.0, 50000.0], "situacao": ["ativo", "ativo"]})
    sin = pd.DataFrame({
        "id_linha": [2], "id_sinistro": ["S1"], "id_estipulante": ["E1"], "id_vida": ["A"],
        "cobertura": ["MQC"], "data_ocorrencia": [T("2023-05-01")], "data_aviso": [T("2023-06-01")],
        "data_pagamento": [T("2023-06-20")], "valor_indenizado": [50000.0], "valor_reserva": [0.0],
        "status": ["pago"]})
    return {"estipulantes": est, "vidas": vidas, "sinistros": sin}


def _add_vida(d, **kw):
    nova = d["vidas"].iloc[[1]].copy()
    nova["id_linha"] = d["vidas"]["id_linha"].max() + 1
    for k, v in kw.items():
        nova[k] = v
    d["vidas"] = pd.concat([d["vidas"], nova], ignore_index=True)
    return int(nova["id_linha"].iloc[0])


def _codigos(r, id_linha=None, id_sinistro=None):
    ex = r.excecoes
    if id_linha is not None:
        ex = ex[(ex["tabela"] == "vidas") & (ex["id_linha"] == id_linha)]
    if id_sinistro is not None:
        ex = ex[ex["id_sinistro"] == id_sinistro]
    return set(ex["codigo"])


def test_base_limpa_sem_excecoes(cfg_base):
    r = validar(_base(), cfg_base)
    assert r.excecoes.empty
    assert len(r.dados["vidas"]) == 2


@pytest.mark.parametrize("campos,codigo", [
    ({"id_vida": "B"}, "V01"),                                        # cópia exata de B
    ({"id_vida": "C", "id_estipulante": "E9"}, "V02"),
    ({"id_vida": "C", "data_nascimento": pd.NaT}, "V03"),
    ({"id_vida": "C", "data_nascimento": T("1900-01-01")}, "V04"),
    ({"id_vida": "C", "sexo": None}, "V05"),
    ({"id_vida": "C", "capital": 0.0}, "V06"),
    ({"id_vida": "C", "data_inicio": T("2023-01-01"), "data_fim": T("2022-12-01")}, "V07"),
    ({"id_vida": "C", "data_inicio": T("2026-03-01")}, "V08"),
    ({"id_vida": "C", "situacao": "afastado"}, "V09"),
    ({"data_inicio": T("2023-01-01")}, "V10"),                        # B de novo, sobreposto
    ({"id_vida": "C", "data_nascimento": T("1940-01-01")}, "V12"),   # 82 anos na entrada
    ({"id_vida": "C", "capital": 9_000_000.0}, "V13"),
    ({"id_vida": "C", "data_inicio": T("2021-06-01")}, "V14"),
])
def test_regras_de_vidas(cfg_base, campos, codigo):
    d = _base()
    lid = _add_vida(d, **campos)
    r = validar(d, cfg_base)
    assert codigo in _codigos(r, id_linha=lid)


def test_v11_inconsistencia_entre_segmentos(cfg_base):
    d = _base()
    d["vidas"].loc[1, "data_fim"] = T("2022-12-31")
    _add_vida(d, data_inicio=T("2023-01-01"), data_fim=pd.NaT, data_nascimento=T("1991-01-01"))
    r = validar(d, cfg_base)
    assert "V11" in set(r.excecoes["codigo"])


def test_excluido_nao_e_reavaliado(cfg_base):
    d = _base()
    lid = _add_vida(d, id_vida="C", capital=0.0, data_nascimento=T("1900-01-01"))
    r = validar(d, cfg_base)
    assert _codigos(r, id_linha=lid) == {"V04"}       # sai no V04, não chega no V06


@pytest.mark.parametrize("campos,codigo", [
    ({"id_vida": "Z"}, "S02"),
    ({"data_ocorrencia": T("2021-05-01"), "data_aviso": T("2021-06-01")}, "S03"),
    ({"data_aviso": T("2023-04-01")}, "S04"),
    ({"valor_indenizado": 80000.0}, "S05"),
    ({"status": "negado"}, "S06"),
    ({"data_ocorrencia": T("2026-02-01"), "data_aviso": T("2026-02-10")}, "S07"),
])
def test_regras_de_sinistros(cfg_base, campos, codigo):
    d = _base()
    for k, v in campos.items():
        d["sinistros"][k] = v
    r = validar(d, cfg_base)
    assert codigo in _codigos(r, id_sinistro="S1")


def test_s01_duplicidade(cfg_base):
    d = _base()
    dup = d["sinistros"].copy()
    dup["id_sinistro"] = "S2"
    d["sinistros"] = pd.concat([d["sinistros"], dup], ignore_index=True)
    r = validar(d, cfg_base)
    assert _codigos(r, id_sinistro="S2") == {"S01"}
    assert len(r.dados["sinistros"]) == 1


def test_s08_sinistro_de_afastado(cfg_base):
    d = _base()
    d["vidas"].loc[0, "situacao"] = "afastado"
    r = validar(d, cfg_base)
    assert "S08" in _codigos(r, id_sinistro="S1")


def test_p01_conciliacao(cfg_base):
    d = _base()
    # 2 vidas x 50.000 x ano 2023 completo x 0,3‰/mês x 12
    teorico = 100000 * 365 / 365.25 * 0.3 * 12 / 1000
    d["premios"] = pd.DataFrame({"id_linha": [2, 3], "id_estipulante": ["E1", "E1"],
                                 "inicio_periodo": [T("2023-01-01")] * 2, "fim_periodo": [T("2023-12-31")] * 2,
                                 "premio_emitido": [teorico, teorico * 1.10], "taxa_mensal_mil": [0.3, 0.3]})
    r = validar(d, cfg_base)
    p01 = r.excecoes[r.excecoes["codigo"] == "P01"]
    assert p01["id_linha"].tolist() == [3]
