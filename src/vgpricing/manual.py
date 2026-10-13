"""Etapa 2: taxa manual.

Modelo de frequência de óbitos relativo à tábua, estimado como GLMM no R (glmmTMB, único pacote de
modelagem do projeto):

    N ~ Poisson(mu),  log mu = log E + beta_0 + beta_cap * log(capital / capital_ref)
                               [+ fatores candidatos] + u_secao + u_secao:divisao + u_estipulante

- Atividade segue a estrutura de Ohlsson (2008): seção CNAE (poucos níveis) como efeito FIXO e
  divisão dentro da seção (muitos níveis) como efeito ALEATÓRIO, isto é, credibilidade de cada
  divisão em torno da sua seção. Seções com pouca exposição são agrupadas em "OUTRAS".
- Capital entra como log(capital) contínuo: um parâmetro só, mais poder que faixas.
- Atividade e capital estão no modelo por decisão de negócio tomada ANTES de olhar os dados.
  Faixa etária, sexo e porte são candidatos: entram só com evidência (razão de verossimilhança,
  com correção de Holm para testes múltiplos).
- Estima-se em contagem de óbitos; o valor (R$) entra na renovação, com pooling.
- Só entra experiência maturada (até data_referencia - meses_maturacao).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .exposicao import exposicao_detalhada, idade_completa
from .glmm import ResultadoGLMM, ajustar_glmm
from .tabua import Tabua

CATEGORICAS_BASE = ["id_estipulante", "cnae_secao", "cnae_divisao"]


# --------------------------------------------------------------------------- base de modelagem

def _faixa(valores, faixa_cfg) -> pd.Series:
    return pd.cut(valores, bins=faixa_cfg["limites"], labels=faixa_cfg["rotulos"], right=False).astype(str)


def base_modelagem(vidas: pd.DataFrame, sinistros: pd.DataFrame, estipulantes: pd.DataFrame, tabua: Tabua,
                   data_referencia, cfg_manual: dict, dias_por_ano: float = 365.25,
                   janela: tuple | None = None, excluir_anos: tuple[int, ...] = ()) -> tuple[pd.DataFrame, dict]:
    """Células (estipulante x faixa etária x sexo x faixa de capital) com E, N e o log do capital
    médio da célula (ponderado pelos óbitos esperados, porque o efeito multiplica q_x).

    Sem `janela`: toda a experiência maturada (até data_referencia - meses_maturacao).
    Com `janela=(inicio, fim)`: só a experiência ocorrida entre as duas datas (separação treino/teste fora
    do tempo); `fim` não pode passar da data de maturação."""
    maturado = pd.Timestamp(data_referencia) - pd.DateOffset(months=cfg_manual["meses_maturacao"])
    inicio = None
    corte = maturado
    if janela is not None:
        inicio, corte = (pd.Timestamp(x) if x is not None else None for x in janela)
        corte = corte or maturado
        if corte > maturado:
            raise ValueError(f"fim da janela ({corte.date()}) depois da experiência maturada ({maturado.date()})")
    if excluir_anos:      # experiência em pedaços que pulam os anos excluídos
        partes, ini_p = [], inicio
        for ano in sorted(excluir_anos):
            fim_p = min(pd.Timestamp(f"{ano - 1}-12-31"), corte)
            if ini_p is None or fim_p >= ini_p:
                partes.append(exposicao_detalhada(vidas, estipulantes, tabua, fim_p, dias_por_ano, data_inicio=ini_p))
            depois = pd.Timestamp(f"{ano + 1}-01-01")
            ini_p = depois if ini_p is None else max(ini_p, depois)     # nunca antes do início da janela
        if ini_p <= corte:
            partes.append(exposicao_detalhada(vidas, estipulantes, tabua, corte, dias_por_ano, data_inicio=ini_p))
        if not partes:
            raise ValueError(f"sem experiência entre {inicio} e {corte.date()} fora dos anos excluídos {list(excluir_anos)}")
        det = pd.concat(partes, ignore_index=True)
    else:
        det = exposicao_detalhada(vidas, estipulantes, tabua, corte, dias_por_ano, data_inicio=inicio)
    fx = cfg_manual["faixas"]
    est = estipulantes[["id_estipulante", "cnae_secao", "cnae_divisao"]].copy()
    est["cnae_secao"] = est["cnae_secao"].astype("string").fillna("ND")
    est["cnae_divisao"] = est["cnae_divisao"].astype("string").fillna("ND")
    det = det.merge(est, on="id_estipulante", how="left")
    det["sexo"] = det["sexo"].astype("string").fillna("ND")
    det["faixa_idade"] = _faixa(det["idade"], fx["idade"])
    det["faixa_capital"] = _faixa(det["capital"], fx["capital"])
    porte = det.groupby("id_estipulante")[["exposicao"]].sum().join(
        det.groupby("id_estipulante")["ano_apolice"].nunique()).eval("exposicao / ano_apolice").rename("vidas_medias")
    det = det.merge(porte, on="id_estipulante")
    det["faixa_tamanho"] = _faixa(det["vidas_medias"], fx["tamanho"])
    det["E_logcap"] = det["obitos_esperados"] * np.log(det["capital"])

    s = sinistros[(sinistros["data_ocorrencia"] <= corte)
                  & ((sinistros["data_ocorrencia"] >= inicio) if inicio is not None else True)
                  & ~sinistros["data_ocorrencia"].dt.year.isin(list(excluir_anos))]
    m = s[["id_sinistro", "id_estipulante", "id_vida", "data_ocorrencia"]].merge(
        det[["id_estipulante", "id_vida", "ini", "fim", "faixa_idade", "sexo", "faixa_capital", "capital"]],
        on=["id_estipulante", "id_vida"], how="inner")
    m = m[(m["data_ocorrencia"] >= m["ini"]) & (m["data_ocorrencia"] <= m["fim"])].drop_duplicates("id_sinistro")
    sem_celula = sorted(set(s["id_sinistro"]) - set(m["id_sinistro"]))

    chave = ["id_estipulante", "faixa_idade", "sexo", "faixa_capital"]
    cel = det.groupby(chave, as_index=False, observed=True).agg(
        E=("obitos_esperados", "sum"), E_logcap=("E_logcap", "sum"), exposicao=("exposicao", "sum"),
        valor_esperado=("valor_esperado", "sum"))
    obs = m.groupby(chave, as_index=False, observed=True).agg(N=("id_sinistro", "count"),
                                                             valor_observado=("capital", "sum"))
    cel = cel.merge(obs, on=chave, how="left").fillna({"N": 0, "valor_observado": 0.0})
    cel["N"] = cel["N"].astype(int)
    cel = cel.merge(est, on="id_estipulante").merge(
        det[["id_estipulante", "faixa_tamanho"]].drop_duplicates(), on="id_estipulante")
    cel["log_capital_c"] = cel.pop("E_logcap") / cel["E"] - np.log(cfg_manual["capital_referencia"])
    exp_sec = cel.groupby("cnae_secao")["E"].sum()
    raras = sorted(exp_sec[exp_sec < cfg_manual["modelo"]["min_esperados_secao"]].index)
    cel["cnae_secao_original"] = cel["cnae_secao"]
    cel.loc[cel["cnae_secao"].isin(raras), "cnae_secao"] = "OUTRAS"
    n_nd = int((cel["sexo"] == "ND").sum())
    cel = cel[(cel["sexo"] != "ND") & (cel["E"] > 0)].reset_index(drop=True)
    info = {"inicio_experiencia": inicio.date().isoformat() if inicio is not None else None,
            "corte_experiencia": corte.date().isoformat(), "anos_excluidos": list(excluir_anos),
            "sinistros_no_periodo": int(len(s)),
            "sinistros_sem_celula": len(sem_celula), "ids_sem_celula": sem_celula[:50],
            "celulas_sexo_ND_removidas": n_nd, "celulas": int(len(cel)), "estipulantes": int(cel["id_estipulante"].nunique()),
            "obitos": int(cel["N"].sum()), "esperados": float(cel["E"].sum()),
            "secoes_agrupadas_em_OUTRAS": raras}
    return cel, info


# --------------------------------------------------------------------------- ajuste e seleção

def niveis_base(cel: pd.DataFrame, colunas: list[str]) -> dict[str, str]:
    """Nível base de cada fator categórico fixo: o de maior exposição (óbitos esperados)."""
    return {c: str(cel.groupby(c)["E"].sum().idxmax()) for c in colunas}


def _ajustar(cel, cfg_manual, fixos, testar_fixos=(), testar_aleatorios=(), aleatorios=None,
             familia="poisson") -> ResultadoGLMM:
    m = cfg_manual["modelo"]
    cats = ["cnae_secao"] + list(m["fixos_candidatos"])
    return ajustar_glmm(cel, fixos=fixos, aleatorios=m["aleatorios"] if aleatorios is None else aleatorios,
                        categoricas=CATEGORICAS_BASE + list(m["fixos_candidatos"]), numericas=["log_capital_c"],
                        niveis_base=niveis_base(cel, cats), testar_fixos=list(testar_fixos),
                        testar_aleatorios=list(testar_aleatorios), familia=familia)


def holm(p: pd.Series) -> pd.Series:
    """p-valores ajustados por Holm (controle do erro de família nos testes dos candidatos)."""
    p = pd.Series(p, dtype=float)
    ordem = p.sort_values().index
    m = len(p)
    aj = pd.Series(np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p.loc[ordem].to_numpy())), index=ordem)
    return aj.reindex(p.index)


def ajustar_taxa_manual(cel: pd.DataFrame, cfg_manual: dict) -> tuple[ResultadoGLMM, ResultadoGLMM, pd.DataFrame]:
    """1) Ajusta o modelo completo (obrigatórios + candidatos) e testa cada candidato e cada efeito
    aleatório por razão de verossimilhança. 2) Reajusta só com os candidatos com p <= alfa.
    Devolve (modelo_final, modelo_completo, testes)."""
    m = cfg_manual["modelo"]
    completo = _ajustar(cel, cfg_manual, m["fixos_obrigatorios"] + m["fixos_candidatos"],
                        testar_fixos=m["fixos_candidatos"], testar_aleatorios=m["aleatorios"])
    t = completo.testes.copy()
    fixo = t["tipo"] == "fixo"
    t["p_holm"] = np.nan
    t.loc[fixo, "p_holm"] = holm(t.loc[fixo, "p_valor"])
    t["decisao"] = np.where(~fixo, "estrutura (mantido)", np.where(t["p_holm"] <= m["alfa"], "entra", "não entra"))
    entram = t.loc[(t["tipo"] == "fixo") & (t["decisao"] == "entra"), "termo"].tolist()
    final = _ajustar(cel, cfg_manual, m["fixos_obrigatorios"] + entram)
    return final, completo, t


# --------------------------------------------------------------------------- comparações

def ajustar_comparacao(cel: pd.DataFrame, cfg_manual: dict, fixos: list[str], familia: str) -> ResultadoGLMM:
    """Modelo de comparação sem efeitos aleatórios (GLM Poisson ou Binomial Negativa), no mesmo glmmTMB."""
    return _ajustar(cel, cfg_manual, fixos, aleatorios=[], familia=familia)


def comparar_dispersao(cel: pd.DataFrame, cfg_manual: dict, res_glmm: ResultadoGLMM) -> pd.DataFrame:
    """Onde está a sobredispersão? Mesmos efeitos fixos do modelo final em três versões, todas no
    glmmTMB (log-verossimilhanças diretamente comparáveis)."""
    fixos = res_glmm.especificacao["fixos"]
    p = ajustar_comparacao(cel, cfg_manual, fixos, "poisson")
    nb = ajustar_comparacao(cel, cfg_manual, fixos, "nbinom2")
    linhas = [
        {"modelo": "GLM Poisson", "dispersão": "nenhuma", "loglik": p.loglik, "parametros": p.gl},
        {"modelo": "Binomial Negativa (NB2)", "dispersão": "por célula, independente", "loglik": nb.loglik,
         "parametros": nb.gl},
        {"modelo": "GLMM (efeitos de divisão e estipulante)", "dispersão": "por grupo e por atividade",
         "loglik": res_glmm.loglik, "parametros": res_glmm.gl},
    ]
    out = pd.DataFrame(linhas)
    out["AIC"] = -2 * out["loglik"] + 2 * out["parametros"]
    out["dif_AIC_vs_melhor"] = out["AIC"] - out["AIC"].min()
    return out


# --------------------------------------------------------------------------- tabela de fatores

def tabela_fatores(res: ResultadoGLMM, cfg_manual: dict, meta: dict) -> dict:
    """Fatores para o cotador. A/E base inclui a média marginal do efeito de estipulante
    (grupo novo); seção/divisão não vistas usam a média marginal do respectivo efeito."""
    v = res.variancias
    fat_cat = {}
    for c in cfg_manual["modelo"]["fixos_candidatos"]:
        if c in res.especificacao["fixos"]:
            base = res.especificacao["niveis_base"][c]
            niveis = {base: 1.0}
            for nome, b in zip(res.fixos["nome"], res.fixos["estimativa"]):
                if nome.startswith(c):
                    niveis[nome[len(c):]] = float(np.exp(b))
            fat_cat[c] = niveis
    base_sec = res.especificacao["niveis_base"]["cnae_secao"]
    secoes = {base_sec: 1.0}
    for nome, b in zip(res.fixos["nome"], res.fixos["estimativa"]):
        if nome.startswith("cnae_secao"):
            secoes[nome[len("cnae_secao"):]] = float(np.exp(b))
    peso = meta.get("esperados_por_secao") or {}
    if peso:
        tot = sum(peso.get(k, 0.0) for k in secoes)
        media_sec = sum(secoes[k] * peso.get(k, 0.0) for k in secoes) / tot if tot else 1.0
    else:
        media_sec = float(np.mean(list(secoes.values())))
    divisoes = res.aleatorios.get("cnae_secao:cnae_divisao", pd.DataFrame(columns=["nivel", "modo"]))
    return {
        "meta": {**meta, "formula": res.formula, "variancias_efeitos_aleatorios": v},
        "ae_base": float(np.exp(res.coef("(Intercept)") + v.get("id_estipulante", 0.0) / 2)),
        "capital": {"referencia": float(cfg_manual["capital_referencia"]),
                    "elasticidade": res.coef("log_capital_c")},
        "secao": secoes,
        "secao_nao_vista": float(secoes.get("OUTRAS", media_sec)),
        "divisao": {str(n): float(np.exp(u)) for n, u in zip(divisoes["nivel"], divisoes["modo"])},
        "divisao_nao_vista": float(np.exp(v.get("cnae_secao:cnae_divisao", 0.0) / 2)),
        "fatores_categoricos": fat_cat,
        "faixas": cfg_manual["faixas"],
    }


def tabela_fatores_ridge(coeficientes: pd.Series, prioris: dict | None, cel: pd.DataFrame, cfg_manual: dict,
                         meta: dict) -> dict:
    """Tabela de fatores (mesmo formato de `tabela_fatores`) a partir da ridge do glmnet:
    log mu = log E + b0 + (beta_priori + delta) * log_capital_c + log f_priori(seção) + gamma(seção).
    Sem priori, beta_priori = 0 e f_priori = 1. A ridge não tem efeito de grupo: a previsão já é a média
    da carteira, sem correção de grupo novo."""
    pri = prioris or {"elasticidade_capital": 0.0, "fator_secao": {}}
    b = coeficientes
    b = b.drop(".nula", errors="ignore")
    gam = {k[len("cnae_secao"):]: v for k, v in b.items() if k.startswith("cnae_secao")}
    secoes = sorted(set(cel["cnae_secao"].astype(str)))
    log_f = {s_: np.log(pri["fator_secao"].get(s_, 1.0)) + gam.get(s_, 0.0) for s_ in secoes}
    base = cel.groupby("cnae_secao")["E"].sum().idxmax()
    secao = {s_: float(np.exp(log_f[s_] - log_f[base])) for s_ in secoes}
    for s_, f in pri["fator_secao"].items():        # seções só da priori (sem experiência própria)
        if s_ not in secao:
            secao[s_] = float(f / np.exp(log_f[base]))
    return {
        "meta": {**meta, "modelo": "ridge (glmnet)" + (" com priori" if prioris else ""),
                 "elasticidade_priori": pri["elasticidade_capital"],
                 "desvio_elasticidade": float(b.get("log_capital_c", 0.0))},
        "ae_base": float(np.exp(b["(Intercept)"] + log_f[base])),
        "capital": {"referencia": float(cfg_manual["capital_referencia"]),
                    "elasticidade": float(pri["elasticidade_capital"] + b.get("log_capital_c", 0.0))},
        "secao": secao,
        "secao_nao_vista": float(1.0 / np.exp(log_f[base])),
        "divisao": {},
        "divisao_nao_vista": 1.0,
        "fatores_categoricos": {},
        "faixas": cfg_manual["faixas"],
    }


METODOS_TARIFA = ("ridge com priori SOA", "GLMM", "GLM", "plano atual (tábua × A/E)")


def ajustar_tabela(cel: pd.DataFrame, metodo: str, cfg_manual: dict, prioris: dict | None, meta: dict,
                   semente: int = 2026) -> dict:
    """Ajusta um dos métodos que podem virar tarifa e devolve a tabela de fatores no formato do cotador.
    Usada no notebook 03b (reajuste final com toda a experiência) e no backtest da renovação (05), para que os
    dois usem exatamente o mesmo método."""
    meta = {**meta, "modelo_escolhido": metodo,
            "esperados_por_secao": cel.groupby("cnae_secao")["E"].sum().round(3).to_dict()}
    if metodo == "GLMM":
        final, _, _ = ajustar_taxa_manual(cel, cfg_manual)
        tab = tabela_fatores(final, cfg_manual, {**meta, "efeitos_aleatorios_retirados": final.extra.get("aleatorios_removidos", [])})
    elif metodo == "GLM":
        res = ajustar_comparacao(cel, cfg_manual, cfg_manual["modelo"]["fixos_obrigatorios"], "poisson")
        tab = tabela_fatores(res, cfg_manual, meta)
    elif metodo == "ridge com priori SOA":
        from .elasticnet import ajustar_elastic_net
        from .prioris import offset_priori
        d = cel.copy()
        from .avaliacao import folds_por_grupo
        d["fold"] = folds_por_grupo(d["id_estipulante"], cfg_manual["validacao_cruzada"]["folds"], semente) + 1
        d["off_priori"] = offset_priori(d, prioris)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Variáveis com um único nível")   # pouca experiência: só a priori
            r = ajustar_elastic_net(d, {}, ["log_capital_c", "cnae_secao"], ["cnae_secao"], ["log_capital_c"], "off_priori",
                                    alpha=0.0, foldid="fold", semente=semente)
        tab = tabela_fatores_ridge(r.coeficientes, prioris, d, cfg_manual,
                                   {**meta, "lambda": r.lambda_escolhido,
                                    "formula": "N ~ log_capital_c + cnae_secao, offset = log E + priori SOA (ridge)"})
    elif metodo == "plano atual (tábua × A/E)":
        tab = {"meta": {**meta, "formula": "N = E x A/E da carteira"}, "ae_base": float(cel["N"].sum() / cel["E"].sum()),
               "capital": {"referencia": float(cfg_manual["capital_referencia"]), "elasticidade": 0.0},
               "secao": {}, "secao_nao_vista": 1.0, "divisao": {}, "divisao_nao_vista": 1.0,
               "fatores_categoricos": {}, "faixas": cfg_manual["faixas"]}
    else:
        raise ValueError(f"Método {metodo!r} não pode virar tarifa; opções: {METODOS_TARIFA}")
    tab["meta"].setdefault("formula", tab["meta"].get("formula", ""))
    tab["meta"]["modelo_escolhido"] = metodo
    return tab


def escolher_modelo(comparacao_vs_melhor: pd.DataFrame, ordem_preferencia: list[str]) -> tuple[str, list[str]]:
    """Regra de decisão: entre os candidatos empatados com o de menor deviance (IC da diferença contém zero),
    o primeiro na ordem de preferência. Devolve (escolhido, empatados)."""
    empatados = comparacao_vs_melhor.loc[~comparacao_vs_melhor["significativa"], "modelo"].tolist()
    for m in ordem_preferencia:
        if m in empatados:
            return m, empatados
    melhor = comparacao_vs_melhor.sort_values("deviance").iloc[0]["modelo"]
    return melhor, empatados


def _para_python(x):
    if isinstance(x, dict):
        return {str(k): _para_python(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_para_python(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def salvar_tabela(tabela: dict, caminho) -> None:
    import yaml
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        yaml.safe_dump(_para_python(tabela), f, allow_unicode=True, sort_keys=False)


def carregar_tabela(caminho) -> dict:
    import yaml
    with open(caminho, encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- taxa pura

def multiplicador_manual(df: pd.DataFrame, tabela: dict) -> tuple[np.ndarray, list[str]]:
    """Fator A/E da taxa manual para cada linha (vida ou pedaço de exposição).

    df precisa de: capital, cnae_secao, cnae_divisao e, se a tabela tiver fatores categóricos,
    as colunas correspondentes (idade para faixa_idade, sexo, faixa_tamanho).
    Devolve (multiplicador, avisos)."""
    avisos = []
    sec = df["cnae_secao"].astype(str)
    f_sec = sec.map(tabela["secao"])
    if f_sec.isna().any():
        avisos.append(f"seções sem experiência {sorted(sec[f_sec.isna()].unique())}: usada a média da carteira")
    chave = sec + ":" + df["cnae_divisao"].astype("string").fillna("ND").astype(str)
    f_div = chave.map(tabela["divisao"])
    if f_div.isna().any():
        avisos.append(f"divisões sem experiência {sorted(chave[f_div.isna()].unique())}: usada a média da seção")
    cap = tabela["capital"]
    mult = (tabela["ae_base"] * f_sec.fillna(tabela["secao_nao_vista"]).to_numpy()
            * f_div.fillna(tabela["divisao_nao_vista"]).to_numpy()
            * (df["capital"].to_numpy(float) / cap["referencia"]) ** cap["elasticidade"])
    fx = tabela["faixas"]
    for fator, mapa in tabela.get("fatores_categoricos", {}).items():
        if fator == "faixa_idade":
            col = _faixa(df["idade"], fx["idade"])
        elif fator in df.columns:
            col = df[fator].astype(str)
        else:
            avisos.append(f"{fator}: coluna ausente (usado 1)")
            continue
        valores = col.map(mapa)
        if valores.isna().any():
            avisos.append(f"{fator}: níveis sem fator {sorted(col[valores.isna()].unique())} (usado 1)")
        mult = mult * valores.fillna(1.0).to_numpy()
    return mult, avisos


def taxa_pura_manual(vidas: pd.DataFrame, cnae_secao: str, cnae_divisao: str | None, tabela: dict, tabua: Tabua,
                     data_calculo) -> dict:
    """Taxa pura anual e mensal por mil de capital para o censo de um grupo novo.

    custo esperado da vida = capital x q_x(idade na data, sexo) x A/E base x fator da seção
                             x fator da divisão x (capital / capital_ref)^elasticidade x fatores categóricos
    """
    v = vidas.copy()
    data = pd.Timestamp(data_calculo)
    v["idade"] = idade_completa(v["data_nascimento"], pd.Series(data, index=v.index)).astype(int)
    v["q"] = tabua.q(v["idade"].to_numpy(), v["sexo"].astype(object).to_numpy())
    v["cnae_secao"], v["cnae_divisao"] = cnae_secao, cnae_divisao
    v["faixa_tamanho"] = str(_faixa(pd.Series([len(v)]), tabela["faixas"]["tamanho"]).iloc[0])
    mult, avisos = multiplicador_manual(v, tabela)
    avisos = [a.replace("seções sem experiência", "seção sem experiência").replace("divisões sem experiência",
              "divisão sem experiência") for a in avisos]
    v["q_ajustado"] = v["q"] * mult
    v["custo_esperado"] = v["capital"] * v["q_ajustado"]
    taxa_anual = 1000 * v["custo_esperado"].sum() / v["capital"].sum()
    f_sec = tabela["secao"].get(str(cnae_secao), tabela["secao_nao_vista"])
    f_div = tabela["divisao"].get(f"{cnae_secao}:{cnae_divisao}", tabela["divisao_nao_vista"])
    return {"taxa_pura_anual_mil": float(taxa_anual), "taxa_pura_mensal_mil": float(taxa_anual / 12),
            "premio_puro_anual": float(v["custo_esperado"].sum()), "capital_total": float(v["capital"].sum()),
            "vidas": int(len(v)), "obitos_esperados": float(v["q_ajustado"].sum()),
            "fator_atividade": float(f_sec * f_div), "avisos": avisos, "detalhe": v}
