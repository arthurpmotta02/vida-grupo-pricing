"""Regras de qualidade de dados.

Cada regra tem código, descrição, severidade e ação:
- "excluir": o registro sai da base de precificação (e fica no log);
- "sinalizar": o registro fica, mas aparece no log para análise.
As regras rodam em ordem; registros excluídos por uma regra não são avaliados
pelas seguintes (evita contar o mesmo problema duas vezes).
Nada é corrigido automaticamente.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .exposicao import capital_exposto_periodos, idade_completa
from .schema import VALORES_SEXO

COLS_LOG = ["codigo", "tabela", "descricao", "severidade", "acao", "id_linha", "id_sinistro",
            "id_estipulante", "detalhe"]


@dataclass
class ResultadoValidacao:
    dados: dict[str, pd.DataFrame]
    excecoes: pd.DataFrame
    separados: dict[str, pd.DataFrame] = field(default_factory=dict)

    def resumo(self) -> pd.DataFrame:
        if self.excecoes.empty:
            return pd.DataFrame(columns=["codigo", "tabela", "descricao", "acao", "severidade", "registros"])
        return (self.excecoes.groupby(["codigo", "tabela", "descricao", "acao", "severidade"], as_index=False)
                .size().rename(columns={"size": "registros"}).sort_values("codigo"))


class _Log:
    def __init__(self):
        self.linhas: list[pd.DataFrame] = []

    def add(self, df: pd.DataFrame, mask, codigo, tabela, descricao, severidade, acao, detalhe=None):
        mask = pd.Series(mask, index=df.index).fillna(False).astype(bool)
        if not mask.any():
            return
        sub = df.loc[mask]
        reg = pd.DataFrame({
            "codigo": codigo, "tabela": tabela, "descricao": descricao, "severidade": severidade,
            "acao": acao,
            "id_linha": sub["id_linha"] if "id_linha" in sub else pd.NA,
            "id_sinistro": sub["id_sinistro"] if "id_sinistro" in sub else pd.NA,
            "id_estipulante": sub["id_estipulante"] if "id_estipulante" in sub else pd.NA,
            "detalhe": (detalhe.loc[mask].astype(str) if isinstance(detalhe, pd.Series) else (detalhe or "")),
        })
        self.linhas.append(reg[COLS_LOG])

    def df(self) -> pd.DataFrame:
        if not self.linhas:
            return pd.DataFrame(columns=COLS_LOG)
        return pd.concat(self.linhas, ignore_index=True)


def _aplicar(df, mask, log, codigo, tabela, descricao, severidade, acao, detalhe=None):
    log.add(df, mask, codigo, tabela, descricao, severidade, acao, detalhe)
    if acao == "excluir":
        return df.loc[~pd.Series(mask, index=df.index).fillna(False).astype(bool)]
    return df


def validar_estipulantes(e: pd.DataFrame, log: _Log) -> pd.DataFrame:
    t = "estipulantes"
    e = _aplicar(e, e.duplicated("id_estipulante", keep="first"), log, "E01", t,
                 "id_estipulante duplicado", "alta", "excluir")
    e = _aplicar(e, e["data_inicio_vigencia"].isna(), log, "E02", t,
                 "início de vigência ausente/inválido", "alta", "excluir")
    e = _aplicar(e, e["cnae_secao"].isna(), log, "E03", t, "seção CNAE ausente (sem fator de atividade)",
                 "media", "sinalizar")
    e = _aplicar(e, e["cnae_secao"].notna() & e["cnae_divisao"].isna(), log, "E04", t,
                 "divisão CNAE ausente (usa o fator da seção)", "baixa", "sinalizar")
    return e


def validar_vidas(v: pd.DataFrame, estip: pd.DataFrame, cfg: dict, data_ref: pd.Timestamp,
                  log: _Log) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = "vidas"
    c = cfg["validacao"]
    chave = ["id_estipulante", "id_vida", "data_nascimento", "sexo", "data_inicio", "data_fim", "capital"]
    v = v.sort_values("id_linha")
    v = _aplicar(v, v.duplicated(chave, keep="first"), log, "V01", t,
                 "linha duplicada", "media", "excluir")
    v = _aplicar(v, ~v["id_estipulante"].isin(estip["id_estipulante"]), log, "V02", t,
                 "estipulante inexistente", "alta", "excluir")
    v = _aplicar(v, v["data_nascimento"].isna() | (v["data_nascimento"] > v["data_inicio"]), log, "V03", t,
                 "nascimento ausente ou posterior ao início", "alta", "excluir")
    idade_ent = idade_completa(v["data_nascimento"], v["data_inicio"])
    imposs = (idade_ent < c["idade_minima_possivel"]) | (idade_ent > c["idade_maxima_possivel"])
    v = _aplicar(v, imposs, log, "V04", t, "idade de entrada impossível", "alta", "excluir",
                 detalhe="idade=" + idade_ent.astype("string"))
    v = _aplicar(v, ~v["sexo"].isin(VALORES_SEXO), log, "V05", t,
                 "sexo ausente/inválido (usa média M/F da tábua)", "media", "sinalizar")
    v = _aplicar(v, v["capital"].isna() | (v["capital"] <= 0), log, "V06", t,
                 "capital ausente ou não positivo", "alta", "excluir")
    v = _aplicar(v, v["data_fim"].notna() & (v["data_fim"] < v["data_inicio"]), log, "V07", t,
                 "data_fim anterior a data_inicio", "alta", "excluir")
    v = _aplicar(v, v["data_inicio"] > data_ref, log, "V08", t,
                 "início após a data de referência", "media", "excluir")
    fora = v["situacao"].isin(c["situacoes_fora_da_precificacao"])
    separados = v.loc[fora].copy()
    v = _aplicar(v, fora, log, "V09", t, "afastado/aposentado por invalidez: separado da precificação",
                 "info", "excluir", detalhe=v["situacao"])
    # sobreposição de segmentos da mesma vida no mesmo estipulante
    v = v.sort_values(["id_estipulante", "id_vida", "data_inicio", "id_linha"])
    fim = v["data_fim"].fillna(data_ref)
    chaves = [v["id_estipulante"], v["id_vida"]]
    fim_max_ant = fim.groupby(chaves, sort=False).cummax().groupby(chaves, sort=False).shift()
    sobrep = fim_max_ant.notna() & (v["data_inicio"] <= fim_max_ant)
    v = _aplicar(v, sobrep, log, "V10", t, "segmento sobreposto a outro da mesma vida", "alta", "excluir")
    g = v.groupby(["id_estipulante", "id_vida"])
    inconsist = (g["data_nascimento"].transform("nunique") > 1) | (g["sexo"].transform("nunique") > 1)
    v = _aplicar(v, inconsist, log, "V11", t, "nascimento/sexo divergentes entre segmentos da mesma vida",
                 "media", "sinalizar")
    idade_ent = idade_completa(v["data_nascimento"], v["data_inicio"])
    v = _aplicar(v, idade_ent > c["idade_maxima_aceitacao"], log, "V12", t,
                 "idade de entrada acima do limite de aceitação", "media", "sinalizar",
                 detalhe="idade=" + idade_ent.astype("string"))
    v = _aplicar(v, v["capital"] > c["capital_maximo_sinalizacao"], log, "V13", t,
                 "capital acima do limite de sinalização", "media", "sinalizar",
                 detalhe="capital=" + v["capital"].astype("string"))
    ini_vig = v["id_estipulante"].map(estip.set_index("id_estipulante")["data_inicio_vigencia"])
    v = _aplicar(v, v["data_inicio"] < ini_vig, log, "V14", t,
                 "início do segmento antes da vigência da apólice (exposição prévia é ignorada)",
                 "baixa", "sinalizar")
    return v.sort_values("id_linha").reset_index(drop=True), separados


def validar_sinistros(s: pd.DataFrame, vidas_bruto: pd.DataFrame, vidas_ok: pd.DataFrame,
                      separados: pd.DataFrame, cfg: dict, data_ref: pd.Timestamp, log: _Log) -> pd.DataFrame:
    t = "sinistros"
    c = cfg["validacao"]
    s = s.reset_index(drop=True)
    dup = s.duplicated("id_sinistro", keep="first") | \
        s.duplicated(["id_estipulante", "id_vida", "cobertura", "data_ocorrencia"], keep="first")
    s = _aplicar(s, dup, log, "S01", t, "sinistro duplicado", "alta", "excluir")
    s = _aplicar(s, s["data_ocorrencia"].isna() | (s["data_ocorrencia"] > data_ref), log, "S07", t,
                 "ocorrência ausente ou após a data de referência", "alta", "excluir")
    s = _aplicar(s, s["status"].isin(c["status_sinistro_excluidos"]), log, "S06", t,
                 "sinistro negado: fora da experiência", "info", "excluir")
    sep = s.set_index(["id_estipulante", "id_vida"]).index.isin(
        separados.set_index(["id_estipulante", "id_vida"]).index)
    s = _aplicar(s, sep, log, "S08", t, "sinistro de vida separada (afastado/invalidez)", "info", "excluir")
    existe = s.set_index(["id_estipulante", "id_vida"]).index.isin(
        vidas_bruto.set_index(["id_estipulante", "id_vida"]).index)
    s = _aplicar(s, ~existe, log, "S02", t,
                 "vida não encontrada na base de vidas (mantido até investigação)", "alta", "sinalizar")
    # cobertura vigente na ocorrência (contra a base limpa)
    vv = vidas_ok[["id_estipulante", "id_vida", "data_inicio", "data_fim", "capital"]].copy()
    vv["data_fim"] = vv["data_fim"].fillna(data_ref)
    m = s.reset_index().merge(vv, on=["id_estipulante", "id_vida"], how="left")
    m["cobre"] = (m["data_ocorrencia"] >= m["data_inicio"]) & (m["data_ocorrencia"] <= m["data_fim"])
    cob = m[m["cobre"]].groupby("index")["capital"].max()
    tem_cob = s.index.isin(cob.index)
    s = _aplicar(s, existe & ~tem_cob, log, "S03", t,
                 "ocorrência fora de período de cobertura na base limpa", "alta", "sinalizar")
    s = _aplicar(s, s["data_aviso"] < s["data_ocorrencia"], log, "S04", t,
                 "aviso anterior à ocorrência", "media", "sinalizar")
    valor = s["valor_indenizado"].fillna(0) + s["valor_reserva"].fillna(0)
    cap = pd.Series(cob, dtype=float).reindex(s.index)
    s = _aplicar(s, cap.notna() & (valor > cap * 1.0001), log, "S05", t,
                 "valor do sinistro acima do capital vigente", "alta", "sinalizar",
                 detalhe="valor=" + valor.round(2).astype("string") + " capital=" + cap.astype("string"))
    return s.reset_index(drop=True)


def conciliar_premios(p: pd.DataFrame, vidas_ok: pd.DataFrame, cfg: dict, data_ref, log: _Log) -> pd.DataFrame:
    t = "premios"
    p = p.reset_index(drop=True).copy()
    if "taxa_mensal_mil" not in p or p["taxa_mensal_mil"].isna().all():
        return p
    cap = capital_exposto_periodos(vidas_ok, p, data_ref, cfg["exposicao"]["dias_por_ano"])
    p["premio_teorico"] = cap * p["taxa_mensal_mil"] * 12 / 1000
    p["dif_rel"] = np.where(p["premio_teorico"] > 0, p["premio_emitido"] / p["premio_teorico"] - 1, np.nan)
    tol = cfg["validacao"]["tolerancia_conciliacao_premio"]
    _aplicar(p, p["dif_rel"].abs() > tol, log, "P01", t,
             f"prêmio emitido difere de taxa × capital-ano em mais de {tol:.0%}", "media", "sinalizar",
             detalhe="dif_rel=" + p["dif_rel"].round(4).astype("string"))
    return p


def validar(dados: dict[str, pd.DataFrame], cfg: dict) -> ResultadoValidacao:
    data_ref = pd.Timestamp(cfg["data_referencia"])
    log = _Log()
    e = validar_estipulantes(dados["estipulantes"], log)
    v, separados = validar_vidas(dados["vidas"], e, cfg, data_ref, log)
    out = {"estipulantes": e.reset_index(drop=True), "vidas": v}
    if "sinistros" in dados:
        out["sinistros"] = validar_sinistros(dados["sinistros"], dados["vidas"], v, separados, cfg, data_ref, log)
    if "premios" in dados:
        # afastados continuam faturados: a conciliação usa a base limpa + separados
        out["premios"] = conciliar_premios(dados["premios"], pd.concat([v, separados]), cfg, data_ref, log)
    return ResultadoValidacao(dados=out, excecoes=log.df(), separados={"vidas": separados})
