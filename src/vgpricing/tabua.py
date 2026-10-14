"""Tábua de mortalidade (q_x anual por idade e sexo).

Fonte oficial: arquivo da SUSEP com as versões da BR-EMS
("Tabuas BR-EMS 2010 2015 2021 2026.xlsx"). Estrutura observada no arquivo
(versões 2015 em diante): uma aba por tábua e sexo (ex.: "BR-EMSmt-2026-m"),
com linhas de título e o cabeçalho "Idade | qx | ... | lx | ex" em posição
que varia entre abas, às vezes com colunas vazias intercaladas. Por isso o
cabeçalho é localizado automaticamente em cada aba.

Garantias de auditoria:
- SHA-256 do arquivo travado em config/base.yaml (arquivo mudou -> cálculo bloqueia);
- validação de sanidade (idades consecutivas, 0 < q <= 1, crescente de 30 a 95);
- checagem mt > sb na mesma versão (protege contra troca de aba; o título de uma
  aba do arquivo oficial tem erro de digitação, então não se confia no texto);
- conciliação q_x vs 1 - l_{x+1}/l_x, reportada em `meta` (não bloqueia).

`tabua_placeholder()` é uma curva ILUSTRATIVA usada só em testes automáticos
e quando o arquivo oficial não está presente. NÃO é a BR-EMS.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
import urllib.request
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

URL_SUSEP = ("https://www.gov.br/susep/pt-br/arquivos/informacoes-tecnicas-e-planos-padroes/"
             "tabuas-biometricas-br-ems/tabuas-br-ems-2010-2015-2021-010721.xlsx/@@download/file")


@dataclass(frozen=True)
class Tabua:
    nome: str
    qx: pd.DataFrame                    # índice: idade inteira; colunas: "M", "F"
    fonte: str
    sha256: str | None = None
    placeholder: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def idade_max(self) -> int:
        return int(self.qx.index.max())

    def q(self, idade, sexo, prop_masculina=None) -> np.ndarray:
        """q_x anual vetorizado. Sexo fora de {M, F} usa a mistura p q_M + (1 - p) q_F, com p = proporção de homens
        do perfil (`prop_masculina`, escalar ou vetor; padrão 0,5). É exata para o número esperado de óbitos, que é
        linear em p."""
        idade = np.clip(np.asarray(idade, dtype=int), self.qx.index.min(), self.idade_max)
        sexo = pd.Series(np.asarray(sexo, dtype=object)).astype("string").fillna("").to_numpy(dtype=object)
        qm = self.qx["M"].to_numpy()[idade - self.qx.index.min()]
        qf = self.qx["F"].to_numpy()[idade - self.qx.index.min()]
        p = 0.5 if prop_masculina is None else np.asarray(prop_masculina, dtype=float)
        return np.where(sexo == "M", qm, np.where(sexo == "F", qf, p * qm + (1 - p) * qf))


# --------------------------------------------------------------------- utilidades

def sha256_arquivo(caminho: str | Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 16), b""):
            h.update(bloco)
    return h.hexdigest()


def baixar_tabua_susep(destino: str | Path = "data/external/tabuas_br_ems.xlsx", url: str = URL_SUSEP,
                       sha256_esperado: str | None = None, timeout: int = 60) -> str:
    """Baixa o arquivo oficial da SUSEP e devolve o SHA-256.

    - Envia User-Agent de navegador (alguns portais gov.br recusam o padrão do Python).
    - Confere que o conteúdo é um .xlsx (assinatura ZIP "PK"), não uma página HTML de erro.
    - Grava em arquivo temporário e só substitui o destino no fim (download parcial não
      corrompe a cópia existente).
    - Se `sha256_esperado` for informado e divergir, NÃO substitui o arquivo atual e levanta erro:
      a SUSEP republicou o arquivo e a nova versão precisa ser revisada antes de entrar em uso.
    """
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (vgpricing; tabua BR-EMS)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            conteudo = resp.read()
    except Exception as e:  # noqa: BLE001 - mensagem orientada ao usuário
        raise ConnectionError(
            f"Não foi possível baixar a tábua de {url} ({type(e).__name__}: {e}). "
            "Baixe manualmente na página da SUSEP (Tábuas biométricas BR-EMS) e salve em "
            f"{destino}.") from e
    if conteudo[:2] != b"PK":
        raise ValueError("O conteúdo baixado não é um arquivo .xlsx (provável página de erro ou link mudou).")
    sha = hashlib.sha256(conteudo).hexdigest()
    if sha256_esperado and sha != sha256_esperado:
        raise ValueError(f"Arquivo da SUSEP mudou: SHA-256 {sha[:12]}… difere do config "
                         f"({sha256_esperado[:12]}…). Arquivo atual mantido; revise a nova versão.")
    tmp = destino.with_suffix(".tmp")
    tmp.write_bytes(conteudo)
    tmp.replace(destino)
    return sha


def _norm(txt) -> str:
    """Normaliza texto para comparação: sem acento, minúsculo, sem espaços, travessões viram hífen."""
    s = unicodedata.normalize("NFKD", str(txt)).encode("ascii", "ignore").decode()
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"\s+", "", s).lower()


def inspecionar_arquivo_tabua(caminho: str | Path, n_linhas: int = 8) -> dict[str, pd.DataFrame]:
    """Primeiras linhas de cada aba, para conferência humana."""
    return pd.read_excel(caminho, sheet_name=None, header=None, nrows=n_linhas)


def validar_qx(qx: pd.DataFrame) -> list[str]:
    """Checagens mínimas de sanidade. Retorna lista de problemas (vazia = ok)."""
    problemas = []
    idades = qx.index.to_numpy()
    if not np.array_equal(idades, np.arange(idades.min(), idades.max() + 1)):
        problemas.append("idades não são inteiros consecutivos")
    for s in ("M", "F"):
        v = qx[s].to_numpy(dtype=float)
        if np.isnan(v).any():
            problemas.append(f"{s}: valores ausentes")
        if ((v <= 0) | (v > 1)).any():
            problemas.append(f"{s}: q_x fora de (0, 1]")
        # a partir de 35: antes disso, tábuas masculinas têm a "corcova de acidentes"
        # (q_x cai entre ~26 e ~31 anos na BR-EMS 2026 masculina), que é demografia, não erro
        faixa = (idades >= 35) & (idades <= 95)
        if (np.diff(v[faixa]) < 0).any():
            problemas.append(f"{s}: q_x não crescente entre 35 e 95 anos")
    return problemas


# --------------------------------------------------------------------- leitura

def _achar_aba(abas_disponiveis: list[str], alvo: str) -> str:
    candidatos = [a for a in abas_disponiveis if _norm(a) == _norm(alvo)]
    if len(candidatos) != 1:
        raise ValueError(f"Aba '{alvo}' não encontrada de forma única. Abas: {abas_disponiveis}")
    return candidatos[0]


def ler_aba_tabua(caminho: str | Path, aba: str, col_idade: str = "Idade", col_qx: str = "qx",
                  col_lx: str = "lx", col_ex: str = "ex") -> pd.DataFrame:
    """Lê uma aba localizando a linha de cabeçalho (primeira com 'Idade' e 'qx').
    Devolve DataFrame indexado por idade com qx e, se existirem, lx e ex."""
    bruto = pd.read_excel(caminho, sheet_name=aba, header=None)
    norm = bruto.map(lambda x: _norm(x) if pd.notna(x) else "")
    linha = None
    for i in range(len(norm)):
        valores = list(norm.iloc[i])
        if _norm(col_idade) in valores and _norm(col_qx) in valores:
            linha = i
            break
    if linha is None:
        raise ValueError(f"[{aba}] cabeçalho com '{col_idade}' e '{col_qx}' não encontrado")
    cab = list(norm.iloc[linha])
    j_idade, j_qx = cab.index(_norm(col_idade)), cab.index(_norm(col_qx))
    j_lx = cab.index(_norm(col_lx)) if _norm(col_lx) in cab else None
    j_ex = cab.index(_norm(col_ex)) if _norm(col_ex) in cab else None
    corpo = bruto.iloc[linha + 1:]
    idade = pd.to_numeric(corpo.iloc[:, j_idade], errors="coerce")
    ok = idade.notna()
    df = pd.DataFrame({"idade": idade[ok].astype(int),
                       "qx": pd.to_numeric(corpo.iloc[:, j_qx][ok], errors="coerce").astype(float)})
    if j_lx is not None:
        df["lx"] = pd.to_numeric(corpo.iloc[:, j_lx][ok], errors="coerce").astype(float)
    if j_ex is not None:
        df["ex"] = pd.to_numeric(corpo.iloc[:, j_ex][ok], errors="coerce").astype(float)
    return df.set_index("idade").sort_index()


def expectativa_abreviada(qx: pd.Series) -> pd.Series:
    """e_x abreviada (curtate): soma de l_{x+k}/l_x, k >= 1, calculada a partir do q_x."""
    q = qx.to_numpy(dtype=float)
    l = np.concatenate([[1.0], np.cumprod(1 - q)])
    cauda = np.cumsum(l[::-1])[::-1]               # cauda[i] = soma l_i..l_n
    with np.errstate(divide="ignore", invalid="ignore"):   # l_x = 0 após q = 1 na última idade
        e = np.where(l[:-1] > 0, cauda[1:] / l[:-1], 0.0)
    return pd.Series(e, index=qx.index)


def _conciliar_lx(df: pd.DataFrame, idade_min: int = 20, idade_max: int = 80) -> dict:
    """Compara q_x com 1 - l_{x+1}/l_x (idades 20-80) e testa a hipótese de lx defasado
    em uma idade (q_{x+1} = 1 - l_{x+1}/l_x), defeito observado no arquivo BR-EMS 2026."""
    if "lx" not in df or df["lx"].isna().all():
        return {"disponivel": False}
    q_impl = 1 - df["lx"].shift(-1) / df["lx"]
    faixa = df.index[(df.index >= idade_min) & (df.index <= idade_max)]
    rel = ((df.loc[faixa, "qx"] - q_impl.loc[faixa]) / q_impl.loc[faixa]).abs().dropna()
    rel_def = ((df["qx"].shift(-1).loc[faixa] - q_impl.loc[faixa]) / q_impl.loc[faixa]).abs().dropna()
    return {"disponivel": True, "faixa": [idade_min, idade_max],
            "max_dif_rel": float(rel.max()), "max_dif_rel_lx_defasado_1_idade": float(rel_def.max()),
            "diagnostico": ("consistente" if rel.max() <= 0.01 else
                            "lx defasado em 1 idade" if rel_def.max() <= 0.01 else "inconsistente")}


def _conciliar_ex(df: pd.DataFrame, idade_max: int = 90, tolerancia: float = 0.06) -> dict:
    """Compara o e_x publicado (arredondado a 0,1) com o e_x abreviado calculado do q_x.
    É a checagem decisiva de que a coluna qx é a que gerou a tábua publicada."""
    if "ex" not in df or df["ex"].isna().all():
        return {"disponivel": False}
    e_calc = expectativa_abreviada(df["qx"])
    faixa = df.index[df.index <= idade_max]
    dif = (df.loc[faixa, "ex"] - e_calc.loc[faixa]).abs().dropna()
    return {"disponivel": True, "max_dif_anos": float(dif.max()), "idade_max_dif": int(dif.idxmax()),
            "consistente": bool(dif.max() <= tolerancia)}


def carregar_tabua_susep(caminho: str | Path, nome: str, abas: dict[str, str],
                         sha256_esperado: str | None = None,
                         abas_sobrevivencia: dict[str, str] | None = None,
                         col_idade: str = "Idade", col_qx: str = "qx") -> Tabua:
    """Carrega a tábua de MORTE (mt) por sexo a partir do arquivo oficial.

    abas: {"M": "BR-EMSmt-2026-m", "F": "BR-EMSmt-2026-f"}
    abas_sobrevivencia (opcional, recomendado): mesmas abas da versão sb, usadas só para
    conferir que a tábua lida tem mortalidade MAIOR que a de sobrevivência (mt > sb).
    """
    caminho = Path(caminho)
    sha = sha256_arquivo(caminho)
    if sha256_esperado and sha != sha256_esperado:
        raise ValueError(f"SHA-256 da tábua difere do registrado no config ({sha[:12]}… vs "
                         f"{sha256_esperado[:12]}…). Revalide a versão antes de usar.")
    with pd.ExcelFile(caminho) as xf:
        disponiveis = xf.sheet_names
    lidos, meta = {}, {"abas": {}, "conciliacao_lx": {}}
    for s in ("M", "F"):
        aba = _achar_aba(disponiveis, abas[s])
        lidos[s] = ler_aba_tabua(caminho, aba, col_idade, col_qx)
        meta["abas"][s] = aba
        meta["conciliacao_lx"][s] = _conciliar_lx(lidos[s])
        meta.setdefault("conciliacao_ex", {})[s] = c_ex = _conciliar_ex(lidos[s])
        if c_ex["disponivel"] and not c_ex["consistente"]:
            raise ValueError(f"[{s}] e_x publicado não bate com o e_x calculado do q_x (dif. {c_ex['max_dif_anos']:.2f} "
                             f"anos na idade {c_ex['idade_max_dif']}): a coluna qx pode não ser a que gerou a tábua.")
    comuns = lidos["M"].index.intersection(lidos["F"].index)
    qx = pd.DataFrame({"M": lidos["M"].loc[comuns, "qx"], "F": lidos["F"].loc[comuns, "qx"]})
    qx.index.name = "idade"
    problemas = validar_qx(qx)
    if problemas:
        raise ValueError(f"Tábua lida com problemas: {problemas}")

    if abas_sobrevivencia:
        meta["checagem_mt_maior_que_sb"] = {}
        for s in ("M", "F"):
            sb = ler_aba_tabua(caminho, _achar_aba(disponiveis, abas_sobrevivencia[s]), col_idade, col_qx)["qx"]
            faixa = [i for i in range(20, 81) if i in sb.index and i in qx.index]
            razao = float(qx.loc[faixa, s].sum() / sb.loc[faixa].sum())
            meta["checagem_mt_maior_que_sb"][s] = razao
            if razao <= 1:
                raise ValueError(f"[{s}] tábua lida tem mortalidade <= sobrevivência (razão {razao:.3f} "
                                 "entre 20 e 80 anos): provável troca de aba mt/sb.")
    for s, c in meta["conciliacao_lx"].items():
        if c.get("disponivel") and c["diagnostico"] != "consistente":
            warnings.warn(f"[{s}] coluna lx: {c['diagnostico']} em relação ao qx (dif. máx. {c['max_dif_rel']:.1%} "
                          "entre 20 e 80 anos). O cálculo usa só o qx, que foi conferido contra o e_x publicado.")
    return Tabua(nome=nome, qx=qx, fonte=str(caminho), sha256=sha, placeholder=False, meta=meta)


# --------------------------------------------------------------------- placeholder e fachada

def tabua_placeholder() -> Tabua:
    """Curva ILUSTRATIVA de Gompertz-Makeham (mu_x = A + B c^x), q_x = 1 - exp(-integral).
    Só para testes automáticos. NÃO usar para preço. Feminino = 60% do masculino (hipótese)."""
    idades = np.arange(0, 117)
    A, B, c = 0.0002, 0.00002, 1.10
    integ = A + B * (c ** idades) * (c - 1) / np.log(c)
    qm = np.minimum(1 - np.exp(-integ), 1.0)
    qf = np.minimum(qm * 0.60, 1.0)
    qm[-1] = qf[-1] = 1.0
    qx = pd.DataFrame({"M": qm, "F": qf}, index=pd.Index(idades, name="idade"))
    return Tabua(nome="PLACEHOLDER-Gompertz-Makeham (NÃO É BR-EMS)", qx=qx,
                 fonte="tabua_placeholder()", placeholder=True,
                 meta={"A": A, "B": B, "c": c, "fator_feminino": 0.60})


def obter_tabua(cfg_base: dict, raiz: Path | None = None) -> Tabua:
    """Tábua oficial se o arquivo existir; senão placeholder (se permitido no config)."""
    from .config import caminho_repo
    t = cfg_base["tabua"]
    caminho = caminho_repo(t["arquivo"], raiz)
    if caminho.exists():
        return carregar_tabua_susep(caminho, t["nome"], t["abas"], t.get("sha256"),
                                    t.get("abas_sobrevivencia"), t.get("coluna_idade", "Idade"),
                                    t.get("coluna_qx", "qx"))
    if t.get("baixar_automaticamente", False):
        try:
            baixar_tabua_susep(caminho, t.get("url", URL_SUSEP), t.get("sha256"))
            return obter_tabua(cfg_base, raiz)
        except (ConnectionError, ValueError) as e:
            warnings.warn(f"Download automático da tábua falhou: {e}")
    if t.get("permitir_placeholder", False):
        warnings.warn(f"Arquivo oficial da tábua não encontrado em {caminho}. Usando PLACEHOLDER "
                      "(não é BR-EMS). Rode vgpricing.tabua.baixar_tabua_susep().")
        return tabua_placeholder()
    raise FileNotFoundError(f"Tábua oficial não encontrada: {caminho}")
