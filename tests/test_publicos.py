"""Leitura dos dados públicos. Os testes com os arquivos reais rodam só onde eles existem
(gerados por scripts/preparar_dados_publicos.py; não versionados)."""
import numpy as np
import pandas as pd
import pytest

from vgpricing import estudo, publicos
from vgpricing.config import RAIZ, carregar_config

CFG = carregar_config("publicos")
SOA = RAIZ / CFG["pasta"] / CFG["soa"]["arquivo"]
SES = RAIZ / CFG["pasta"] / CFG["ses"]["arquivo"]


def test_numero_br():
    s = pd.Series(["191319500,08", "0", "-63957,63", None])
    assert publicos._numero_br(s).tolist()[:3] == [191319500.08, 0.0, -63957.63]


def test_categoria_mantem_ordem_da_soa():
    c = publicos._categoria_ordenada(pd.Series(["B. 25-49", "A. < 25", "K. 2000+", "C. 50-74"]))
    assert list(c.categories) == ["A. < 25", "B. 25-49", "C. 50-74", "K. 2000+"]


def test_ic_poisson_garwood():
    inf, sup = estudo.ic_poisson([0, 10])
    assert inf[0] == 0 and sup[0] == pytest.approx(3.689, abs=1e-3)
    assert inf[1] == pytest.approx(4.795, abs=1e-3) and sup[1] == pytest.approx(18.390, abs=1e-3)


@pytest.mark.skipif(not SOA.exists(), reason="dados da SOA não preparados")
def test_soa_reproduz_tabelas_publicadas():
    """A/E por valor da extração = exhibits 1.8, 1.9 e 1.10 do relatório da SOA (2016)."""
    soa = publicos.ler_soa(CFG)
    assert soa["vidas_ano"].sum() == pytest.approx(44.46e6, rel=1e-3)      # relatório: 44,5 milhões
    assert soa["obitos"].sum() == 96889                                     # relatório: 96,9 mil
    publicado = {"faixa_capital": [129.7, 124.6, 108.0, 101.3, 89.5, 77.9, 62.1, 54.1, 48.1, 51.6, 172.6],
                 "faixa_salario": [138.6, 123.5, 101.4, 89.8, 69.0, 58.9, 60.4, 38.5, 39.6, 42.3, 75.4, 107.5],
                 "porte": [57.9, 133.0, 106.5, 98.8, 94.2, 91.9, 91.1, 95.2, 99.8, 118.2]}
    for var, valores in publicado.items():
        t = estudo.tabela_ae(soa, var, obs_valor="valor_obitos", esp_valor="esperados_v")
        t = t[t["esperados_v"] > 0]
        assert np.allclose((100 * t["ae_valor"]).round(1).to_numpy()[:len(valores)], valores), var


@pytest.mark.skipif(not SES.exists(), reason="dados do SES não preparados")
def test_ses_valores_coerentes():
    ses = publicos.ler_ses(CFG)
    assert ses["noenti"].notna().mean() > 0.95 and ses["noramo"].notna().all()   # cadastros cruzaram
    s = publicos.sinistralidade_anual(ses, "0993").set_index("ano")
    assert 10e9 < s.loc[2024, "premio_ganho"] < 30e9                     # mercado de vida em bilhões
    assert 0.3 < s.loc[2024, "sinistralidade"] < 0.7
    assert s.loc[2021, "sinistralidade"] > s.loc[2019, "sinistralidade"] + 0.15   # pandemia


def test_agrupar_niveis_raros():
    d = pd.DataFrame({"f": ["A. 1", "B. 2", "C. 3", "D. 4", "L. Unknown"], "E": [100.0, 50.0, 10.0, 5.0, 1.0]})
    novo, mapa = estudo.agrupar_niveis_raros(d, "f", minimo=30)
    assert mapa["C. 3"] == mapa["D. 4"] == mapa["B. 2"]                # faixas raras juntam-se ao vizinho
    assert mapa["L. Unknown"] == mapa["A. 1"] == "A. 1"                 # desconhecido raro vai para o modal
    d2 = pd.DataFrame({"g": ["x", "y", "z", "w"], "E": [100.0, 20.0, 15.0, 1.0]})
    _, m2 = estudo.agrupar_niveis_raros(d2, "g", minimo=30, ordenado=False)
    assert m2["y"] == m2["z"] == m2["w"] == "OUTROS" and m2["x"] == "x"
