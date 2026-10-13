import numpy as np
import pandas as pd

from vgpricing.exposicao import experiencia, exposicao_detalhada
from vgpricing.sintetico import gerar_carteira, injetar_sujeira, tabelas_de_entrada
from vgpricing.validacao import validar


def test_reprodutivel(cfg_sint_pequeno, tabua, carteira):
    outra = gerar_carteira(cfg_sint_pequeno, tabua)
    for k in ("vidas", "sinistros", "premios"):
        pd.testing.assert_frame_equal(carteira[k], outra[k])


def test_faixas_de_tamanho(carteira, cfg_sint_pequeno):
    t = carteira["verdade_estipulantes"]["tamanho_inicial"]
    assert t.min() >= cfg_sint_pequeno["tamanho"]["min"] and t.max() <= cfg_sint_pequeno["tamanho"]["max"]


def test_carteira_limpa_so_dispara_regras_esperadas(carteira, cfg_base):
    r = validar(tabelas_de_entrada(carteira), cfg_base)
    assert set(r.excecoes["codigo"]) <= {"S06", "V09"}


def test_sujeira_injetada_e_detectada_exatamente(carteira, cfg_sint_pequeno, cfg_base):
    suja, verdade = injetar_sujeira(carteira, cfg_sint_pequeno)
    r = validar(tabelas_de_entrada(suja), cfg_base)
    ex = r.excecoes
    for (tab, cod), g in verdade.groupby(["tabela", "codigo"]):
        col = "id_linha" if tab == "vidas" else "id_sinistro"
        marcados = set(ex.loc[ex["codigo"] == cod, col].astype(str))
        assert marcados == set(g["chave"].astype(str)), cod


def test_gerador_sem_vies(carteira, cfg_base, tabua):
    """Óbitos simulados (avisados + IBNR + negados) batem com Σ E × fator verdadeiro."""
    r = validar(tabelas_de_entrada(carteira), cfg_base)
    det = exposicao_detalhada(r.dados["vidas"], r.dados["estipulantes"], tabua, cfg_base["data_referencia"])
    xp = experiencia(det, r.dados["sinistros"], r.dados["estipulantes"], cfg_base["data_referencia"])
    g = xp.groupby("id_estipulante")[["obitos_esperados", "obitos_observados"]].sum()
    f = carteira["verdade_estipulantes"].set_index("id_estipulante")["fator_verdadeiro"]
    esperado = (g["obitos_esperados"] * f).sum()
    total = (g["obitos_observados"].sum() + len(carteira["verdade_ibnr"])
             + (carteira["sinistros"]["status"] == "negado").sum())
    z = (total - esperado) / np.sqrt(esperado)
    assert abs(z) < 4, z


def test_ibnr_tem_aviso_apos_referencia(carteira, cfg_base):
    ref = pd.Timestamp(cfg_base["data_referencia"])
    assert (carteira["verdade_ibnr"]["data_aviso"] > ref).all()
    assert (carteira["sinistros"]["data_aviso"] <= ref).all()


def test_ida_e_volta_parquet_pelo_leitor_real(carteira, cfg_sint_pequeno, cfg_base, mapeamentos, tabua, tmp_path):
    """O dado sintético passa pelo MESMO leitor que o dado real (tipos 'string', pd.NA etc.)."""
    from vgpricing.io import ler_pasta, salvar_pasta
    suja, _ = injetar_sujeira(carteira, cfg_sint_pequeno)
    salvar_pasta(tabelas_de_entrada(suja), tmp_path)
    dados, probs = ler_pasta(tmp_path, mapeamentos, "parquet")
    assert probs.empty
    r = validar(dados, cfg_base)
    det = exposicao_detalhada(r.dados["vidas"], r.dados["estipulantes"], tabua, cfg_base["data_referencia"])
    assert det["obitos_esperados"].notna().all() and (det["exposicao"] > 0).all()
    # vidas sem sexo (V05) recebem a média M/F, não somem
    v05 = set(r.excecoes.loc[r.excecoes["codigo"] == "V05", "id_linha"])
    assert det["id_linha"].isin(v05).any()


def test_choque_pandemico_so_nos_dias_do_ano():
    """Intensidade x2 só em 2021: o risco no 2º semestre de 2021 dobra a intensidade, e o de 2022 volta ao normal."""
    from vgpricing.sintetico import _tempo_obito
    rng = np.random.default_rng(0)
    n = 200000
    t = _tempo_obito(rng, np.full(n, 0.5), np.zeros(n, int), 365, np.datetime64("2021-07-01"), {"ano": 2021, "fator": 2.0})
    d1 = (t < 184).mean()
    d2 = ((t >= 184) & (t < 365)).sum() / (t >= 184).sum()
    assert abs(d1 - (1 - np.exp(-0.5 * 2 * 184 / 365.25))) < 0.005
    assert abs(d2 - (1 - np.exp(-0.5 * 181 / 365.25))) < 0.005
