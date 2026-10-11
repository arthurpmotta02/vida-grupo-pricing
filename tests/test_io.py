import pandas as pd
import pytest

from vgpricing.io import faturamento_para_segmentos, hash_id, ler_excel, padronizar
from vgpricing.template import criar_template_excel

T = pd.Timestamp


def test_template_ida_e_volta(tmp_path, mapeamentos):
    arq = criar_template_excel(tmp_path / "t.xlsx", mapeamentos)
    dados, probs = ler_excel(arq, mapeamentos)
    assert probs.empty
    v = dados["vidas"].iloc[0]
    assert v["data_nascimento"] == T("1985-06-15")      # dd/mm/aaaa
    assert v["capital"] == 84000.0
    assert pd.isna(v["data_fim"])
    assert dados["premios"].iloc[0]["premio_emitido"] == pytest.approx(152340.55)


def test_mapeamento_de_colunas_e_falhas_registradas(mapeamentos):
    mapa = {**mapeamentos["vidas"], "capital": "CAP_SEGURADO", "data_nascimento": "DT_NASC"}
    fonte = pd.DataFrame({"id_estipulante": ["E1", "E1"], "id_vida": ["a", "b"],
                          "DT_NASC": ["15/06/1985", "31/02/1985"], "sexo": ["M", "F"],
                          "data_inicio": ["01/01/2024", "01/01/2024"], "CAP_SEGURADO": ["1.234,56", "abc"]})
    df, probs = padronizar(fonte, "vidas", mapa)
    assert df.loc[0, "capital"] == pytest.approx(1234.56)
    assert set(probs["coluna"]) == {"data_nascimento", "capital"}
    assert probs["id_linha"].tolist() == [3, 3]          # linha 3 do Excel


def test_coluna_obrigatoria_ausente(mapeamentos):
    with pytest.raises(KeyError, match="capital"):
        padronizar(pd.DataFrame({"id_vida": ["a"]}), "vidas",
                   {k: v for k, v in mapeamentos["vidas"].items() if k != "capital"})


def test_faturamento_para_segmentos():
    fat = pd.DataFrame({
        "id_estipulante": "E1", "id_vida": ["a"] * 5 + ["b"] * 2,
        "competencia": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-05-01", "2024-06-01",
                        "2024-01-01", "2024-02-01"],
        "capital": [10, 10, 12, 12, 12, 5, 5], "data_nascimento": T("1980-01-01"), "sexo": "M"})
    seg = faturamento_para_segmentos(fat)
    a = seg[seg["id_vida"] == "a"].reset_index(drop=True)
    # jan-fev (10), mar (12), mai-jun (12, após falha em abril)
    assert len(a) == 3
    assert a.loc[0, "data_fim"] == T("2024-02-29")
    assert a.loc[2, "data_inicio"] == T("2024-05-01") and a.loc[2, "data_fim"] == T("2024-06-30")
    assert len(seg[seg["id_vida"] == "b"]) == 1


def test_hash_id_estavel_e_depende_do_sal():
    assert hash_id("123", "s1") == hash_id("123", "s1")
    assert hash_id("123", "s1") != hash_id("123", "s2")
