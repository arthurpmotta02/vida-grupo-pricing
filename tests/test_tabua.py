"""O arquivo de teste reproduz a ESTRUTURA observada no arquivo oficial da SUSEP
(títulos acima do cabeçalho, cabeçalho em linhas diferentes por aba, colunas vazias
intercaladas, nome de aba com espaço sobrando). Os VALORES são sintéticos."""
import numpy as np
import pandas as pd
import pytest

from vgpricing.config import RAIZ, carregar_config
from vgpricing.tabua import (carregar_tabua_susep, expectativa_abreviada, ler_aba_tabua, sha256_arquivo,
                             validar_qx)

IDADES = np.arange(0, 116)


def _q(nivel):
    q = np.minimum(nivel * 0.0003 * np.exp(0.09 * IDADES), 1.0)
    q[-1] = 1.0
    return q


def _lx(q, defasado=False):
    """lx correto, ou com o defeito do arquivo 2026 (construído com q_{x+1})."""
    qq = np.append(q[1:], 1.0) if defasado else q
    lx = np.empty_like(q)
    lx[0] = 1_000_000
    for i in range(1, len(q)):
        lx[i] = lx[i - 1] * (1 - qq[i - 1])
    return np.round(lx)


def _ex(q):
    return np.round(expectativa_abreviada(pd.Series(q)).to_numpy(), 1)


def _aba_layout_espacado(q, defasado=False, ex=None):
    """Como BR-EMSmt-2026-m: título, cabeçalho na linha 3, colunas vazias intercaladas."""
    linhas = [[None] * 11, ["Tabela A.1: título"] + [None] * 10, [None] * 11,
              ["Idade", None, "qx", None, "IC 95% inf", None, "IC 95% sup", None, "lx", None, "ex"]]
    ex = _ex(q) if ex is None else ex
    for i, (qi, li, ei) in enumerate(zip(q, _lx(q, defasado), ex)):
        linhas.append([i, None, qi, None, None, None, None, None, li, None, ei])
    return pd.DataFrame(linhas)


def _aba_layout_compacto(q, defasado=False):
    """Como BR-EMSmt-2026-f: título na linha 0, cabeçalho na linha 1."""
    linhas = [["Tabela A.2: título"] + [None] * 5, ["Idade", "qx", "IC 95% inf", "IC 95% sup", "lx", "ex"]]
    for i, (qi, li, ei) in enumerate(zip(q, _lx(q, defasado), _ex(q))):
        linhas.append([i, qi, None, None, li, ei])
    return pd.DataFrame(linhas)


def _arquivo(tmp_path, q_mt_m=None, q_mt_f=None, nome_aba_f="BR-EMSmt-2026-f "):
    caminho = tmp_path / "tabuas.xlsx"
    q_mt_m = _q(1.0) if q_mt_m is None else q_mt_m
    q_mt_f = _q(0.6) if q_mt_f is None else q_mt_f
    with pd.ExcelWriter(caminho) as w:
        pd.DataFrame([["Vigência"]]).to_excel(w, sheet_name="Vigência", index=False, header=False)
        _aba_layout_compacto(_q(0.8)).to_excel(w, sheet_name="BR-EMSsb-2026-m", index=False, header=False)
        _aba_layout_espacado(q_mt_m).to_excel(w, sheet_name="BR-EMSmt-2026-m", index=False, header=False)
        _aba_layout_compacto(_q(0.5)).to_excel(w, sheet_name="BR-EMSsb-2026-f", index=False, header=False)
        _aba_layout_compacto(q_mt_f).to_excel(w, sheet_name=nome_aba_f, index=False, header=False)
    return caminho


ABAS = {"M": "BR-EMSmt-2026-m", "F": "BR-EMSmt-2026-f"}
ABAS_SB = {"M": "BR-EMSsb-2026-m", "F": "BR-EMSsb-2026-f"}


def test_placeholder_e_marcado_e_valido(tabua):
    assert tabua.placeholder and "NÃO É BR-EMS" in tabua.nome
    assert validar_qx(tabua.qx) == []


def test_q_sexo_desconhecido_usa_media(tabua_constante):
    q = tabua_constante.q([40, 40, 40, 40], ["M", "F", None, pd.NA])
    assert np.allclose(q, [0.001, 0.002, 0.0015, 0.0015])


def test_cabecalho_localizado_nos_dois_layouts(tmp_path):
    arq = _arquivo(tmp_path)
    esp = ler_aba_tabua(arq, "BR-EMSmt-2026-m")
    comp = ler_aba_tabua(arq, "BR-EMSsb-2026-m")
    assert esp.index[0] == 0 and esp.index[-1] == 115
    assert esp.loc[40, "qx"] == pytest.approx(_q(1.0)[40])
    assert comp.loc[40, "qx"] == pytest.approx(_q(0.8)[40])


def test_carrega_tabua_completa_com_checagens(tmp_path):
    arq = _arquivo(tmp_path)
    t = carregar_tabua_susep(arq, "BR-EMSmt-2026", ABAS, sha256_arquivo(arq), ABAS_SB)
    assert not t.placeholder and t.sha256 == sha256_arquivo(arq)
    assert t.meta["abas"]["F"] == "BR-EMSmt-2026-f "             # achou apesar do espaço
    assert t.meta["checagem_mt_maior_que_sb"]["M"] > 1
    assert t.meta["conciliacao_lx"]["M"]["max_dif_rel"] < 0.01     # q e lx consistentes


def test_hash_divergente_bloqueia(tmp_path):
    arq = _arquivo(tmp_path)
    with pytest.raises(ValueError, match="SHA-256"):
        carregar_tabua_susep(arq, "x", ABAS, "0" * 64)


def test_troca_mt_sb_bloqueia(tmp_path):
    arq = _arquivo(tmp_path, q_mt_m=_q(0.7))                       # "mt" menor que sb (0,8)
    with pytest.raises(ValueError, match="troca de aba"):
        carregar_tabua_susep(arq, "x", ABAS, None, ABAS_SB)


def test_aba_inexistente_bloqueia(tmp_path):
    arq = _arquivo(tmp_path)
    with pytest.raises(ValueError, match="não encontrada"):
        carregar_tabua_susep(arq, "x", {"M": "BR-EMSmt-2030-m", "F": "BR-EMSmt-2026-f"})


def test_qx_arredondado_gera_aviso(tmp_path):
    q = _q(1.0)
    arq_ok = _arquivo(tmp_path)
    lx_ok = ler_aba_tabua(arq_ok, "BR-EMSmt-2026-m")["lx"]
    # q arredondado a 4 casas mas lx calculado com o q exato: a conciliação deve acusar
    df = _aba_layout_espacado(q)
    df.iloc[4:, 2] = np.round(q, 4)
    df.iloc[4:, 8] = lx_ok.to_numpy()
    caminho = tmp_path / "arred.xlsx"
    with pd.ExcelWriter(caminho) as w:
        df.to_excel(w, sheet_name="BR-EMSmt-2026-m", index=False, header=False)
        _aba_layout_compacto(_q(0.6)).to_excel(w, sheet_name="BR-EMSmt-2026-f", index=False, header=False)
    with pytest.warns(UserWarning, match="coluna lx"):
        t = carregar_tabua_susep(caminho, "x", ABAS)
    assert t.meta["conciliacao_lx"]["M"]["diagnostico"] == "inconsistente"


def test_tabua_invalida_bloqueia(tmp_path):
    q = _q(1.0)
    q[60] = q[59] / 2           # queda aos 60 (a corcova de 26-31 anos é permitida)
    arq = _arquivo(tmp_path, q_mt_m=q)
    with pytest.raises(ValueError, match="não crescente"):
        carregar_tabua_susep(arq, "x", ABAS)


def test_corcova_de_acidentes_e_permitida(tabua_constante):
    q = tabua_constante.qx.copy()
    q.loc[20:34, "M"] = np.linspace(0.0012, 0.0009, 15)       # queda entre 20 e 34 anos
    assert validar_qx(q) == []


def test_lx_defasado_e_diagnosticado_mas_nao_bloqueia(tmp_path):
    caminho = tmp_path / "def.xlsx"
    with pd.ExcelWriter(caminho) as w:
        _aba_layout_espacado(_q(1.0), defasado=True).to_excel(w, sheet_name="BR-EMSmt-2026-m", index=False, header=False)
        _aba_layout_compacto(_q(0.6), defasado=True).to_excel(w, sheet_name="BR-EMSmt-2026-f", index=False, header=False)
    with pytest.warns(UserWarning, match="defasado em 1 idade"):
        t = carregar_tabua_susep(caminho, "x", ABAS)
    assert t.meta["conciliacao_lx"]["M"]["diagnostico"] == "lx defasado em 1 idade"
    assert t.meta["conciliacao_ex"]["M"]["consistente"]


def test_ex_inconsistente_com_qx_bloqueia(tmp_path):
    q = _q(1.0)
    ex_de_outra_tabua = _ex(np.append(q[1:], 1.0))            # ex gerado por q deslocado
    caminho = tmp_path / "ex.xlsx"
    with pd.ExcelWriter(caminho) as w:
        _aba_layout_espacado(q, ex=ex_de_outra_tabua).to_excel(w, sheet_name="BR-EMSmt-2026-m", index=False, header=False)
        _aba_layout_compacto(_q(0.6)).to_excel(w, sheet_name="BR-EMSmt-2026-f", index=False, header=False)
    with pytest.raises(ValueError, match="e_x publicado"):
        carregar_tabua_susep(caminho, "x", ABAS)


ARQ_OFICIAL = RAIZ / "data/external/tabuas_br_ems.xlsx"


@pytest.mark.skipif(not ARQ_OFICIAL.exists(), reason="arquivo oficial da SUSEP não baixado")
def test_arquivo_oficial_br_ems_2026():
    """Roda só onde o arquivo oficial existe. Trava os achados documentados no relatório."""
    cfg = carregar_config("base")["tabua"]
    with pytest.warns(UserWarning, match="defasado em 1 idade"):
        t = carregar_tabua_susep(ARQ_OFICIAL, cfg["nome"], cfg["abas"], cfg["sha256"], cfg["abas_sobrevivencia"])
    assert t.qx.index.min() == 0 and t.qx.index.max() == 116
    assert all(t.meta["conciliacao_ex"][s]["consistente"] for s in "MF")
    assert all(t.meta["checagem_mt_maior_que_sb"][s] > 1 for s in "MF")
    assert t.qx.loc[40, "M"] == pytest.approx(0.00150710, abs=5e-9)


# ---- download (servidor HTTP local; não depende de internet)
import http.server
import threading

from vgpricing.tabua import baixar_tabua_susep


def _servidor(conteudo: bytes):
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(conteudo)

        def log_message(self, *a):
            pass
    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/arquivo"


def test_download_grava_e_confere_hash(tmp_path):
    arq = _arquivo(tmp_path)
    conteudo = arq.read_bytes()
    srv, url = _servidor(conteudo)
    try:
        destino = tmp_path / "baixado.xlsx"
        sha = baixar_tabua_susep(destino, url, sha256_esperado=sha256_arquivo(arq))
        assert sha == sha256_arquivo(arq) and destino.read_bytes() == conteudo
    finally:
        srv.shutdown()


def test_download_hash_diferente_preserva_arquivo_atual(tmp_path):
    destino = tmp_path / "atual.xlsx"
    destino.write_bytes(b"PK versao antiga")
    srv, url = _servidor(b"PK versao nova republicada")
    try:
        with pytest.raises(ValueError, match="mudou"):
            baixar_tabua_susep(destino, url, sha256_esperado="0" * 64)
        assert destino.read_bytes() == b"PK versao antiga"
    finally:
        srv.shutdown()


def test_download_pagina_html_e_recusada(tmp_path):
    srv, url = _servidor(b"<html>erro</html>")
    try:
        with pytest.raises(ValueError, match="não é um arquivo .xlsx"):
            baixar_tabua_susep(tmp_path / "x.xlsx", url)
    finally:
        srv.shutdown()


def test_download_sem_rede_da_mensagem_util(tmp_path):
    with pytest.raises(ConnectionError, match="Baixe manualmente"):
        baixar_tabua_susep(tmp_path / "x.xlsx", "http://127.0.0.1:1/nada", timeout=2)
