"""Etapa 4: diagnóstico de estrutura faltante com LightGBM.

Ajusta um gradient boosting com perda Poisson cujo ponto de partida (offset, `init_score`) é o log
dos óbitos esperados pela taxa manual. O modelo, portanto, só aprende o que a manual não explica.
- Se a deviance fora da amostra (grupos inteiros deixados de fora) não melhora com significância,
  a conclusão é que não falta estrutura relevante.
- Se melhora, as contribuições SHAP (calculadas pelo próprio LightGBM, TreeSHAP) mostram onde; o que
  aparecer vira candidato a teste no GLMM. O ML não entra na tarifa.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .exposicao import exposicao_detalhada
from .manual import multiplicador_manual
from .tabua import Tabua


def base_diagnostico(vidas: pd.DataFrame, sinistros: pd.DataFrame, estipulantes: pd.DataFrame, tabua: Tabua,
                     tabela_manual: dict, data_referencia, meses_maturacao: int = 12,
                     dias_por_ano: float = 365.25) -> pd.DataFrame:
    """Linhas = estipulante x ano-apólice x idade (5 anos) x sexo x log(capital) arredondado, com o
    esperado pela taxa manual (mu_manual) e os óbitos observados (N). Só experiência maturada."""
    corte = pd.Timestamp(data_referencia) - pd.DateOffset(months=meses_maturacao)
    det = exposicao_detalhada(vidas, estipulantes, tabua, corte, dias_por_ano)
    atrib = ["id_estipulante", "cnae_secao", "cnae_divisao"] + [c for c in ("tipo_capital", "contributario")
                                                                if c in estipulantes.columns]
    det = det.merge(estipulantes[atrib], on="id_estipulante", how="left")
    det["sexo"] = det["sexo"].astype("string").fillna("ND")
    mult, _ = multiplicador_manual(det, tabela_manual)
    det["mu_manual"] = det["obitos_esperados"] * mult
    det["E_tabua"] = det["obitos_esperados"]
    porte = (det.groupby("id_estipulante")["exposicao"].sum()
             / det.groupby("id_estipulante")["ano_apolice"].nunique()).rename("vidas_medias")
    det = det.merge(porte, on="id_estipulante")
    det["idade_q"] = (det["idade"] // 5) * 5
    det["logc_r"] = np.round(np.log(det["capital"]), 1)
    det["w_idade"] = det["mu_manual"] * det["idade"]
    det["w_logc"] = det["mu_manual"] * np.log(det["capital"])

    s = sinistros[sinistros["data_ocorrencia"] <= corte]
    m = s[["id_sinistro", "id_estipulante", "id_vida", "data_ocorrencia"]].merge(
        det[["id_estipulante", "id_vida", "ini", "fim", "ano_apolice", "idade_q", "sexo", "logc_r"]],
        on=["id_estipulante", "id_vida"])
    m = m[(m["data_ocorrencia"] >= m["ini"]) & (m["data_ocorrencia"] <= m["fim"])].drop_duplicates("id_sinistro")

    chave = ["id_estipulante", "ano_apolice", "idade_q", "sexo", "logc_r"]
    b = det.groupby(chave, as_index=False).agg(
        mu_manual=("mu_manual", "sum"), E=("E_tabua", "sum"), w_idade=("w_idade", "sum"), w_logc=("w_logc", "sum"),
        inicio_ano=("inicio_ano", "first"), vidas_medias=("vidas_medias", "first"),
        **{c: (c, "first") for c in atrib if c != "id_estipulante"})
    b = b.merge(m.groupby(chave).size().rename("N").reset_index(), on=chave, how="left").fillna({"N": 0})
    b["idade"] = b.pop("w_idade") / b["mu_manual"]
    b["log_capital"] = b.pop("w_logc") / b["mu_manual"]
    b["log_porte"] = np.log(b["vidas_medias"])
    b["ano_calendario"] = b["inicio_ano"].dt.year
    return b[(b["mu_manual"] > 0) & (b["sexo"] != "ND")].reset_index(drop=True)


def _matriz(df: pd.DataFrame, variaveis: list[str], categoricas: list[str],
            niveis: dict[str, list] | None = None) -> pd.DataFrame:
    """Matriz para o LightGBM. `niveis` fixa as categorias de cada variável (as do treino), para que treino
    e teste tenham a mesma codificação; nível não visto no treino vira ausente."""
    X = df[variaveis].copy()
    for c in categoricas:
        if c in X:
            cats = niveis[c] if niveis else sorted(X[c].astype(str).unique())
            X[c] = pd.Categorical(X[c].astype(str), categories=cats)
    return X


def _folds(grupos: pd.Series, k: int, semente: int) -> list[tuple[np.ndarray, np.ndarray]]:
    u = grupos.unique()
    rng = np.random.default_rng(semente)
    f = grupos.map(dict(zip(rng.permutation(u), np.arange(len(u)) % k))).to_numpy()
    return [(np.flatnonzero(f != i), np.flatnonzero(f == i)) for i in range(k)]


def ajustar_residual(df: pd.DataFrame, cfg: dict) -> dict:
    """LightGBM sobre o resíduo da taxa manual, com validação deixando grupos de fora.
    Devolve previsões fora da amostra (oof), número de rodadas, modelo final e comparação de deviance."""
    import lightgbm as lgb
    X = _matriz(df, cfg["variaveis"], cfg["categoricas"])
    niveis = {c: list(X[c].cat.categories) for c in cfg["categoricas"] if c in X}
    off = np.log(df["mu_manual"].to_numpy())
    dados = lgb.Dataset(X, df["N"].to_numpy(float), init_score=off, free_raw_data=False)
    folds = _folds(df["id_estipulante"], cfg["folds"], cfg["semente"])
    cv = lgb.cv(cfg["lightgbm"], dados, num_boost_round=cfg["rodadas_max"], folds=folds,
                callbacks=[lgb.early_stopping(cfg["parada_antecipada"], verbose=False)], return_cvbooster=True)
    rodadas = len(next(v for k, v in cv.items() if k.endswith("-mean")))
    oof = np.zeros(len(df))
    for (_, te), bst in zip(folds, cv["cvbooster"].boosters):
        oof[te] = bst.predict(X.iloc[te], raw_score=True, num_iteration=rodadas)
    mu_gbm = df["mu_manual"].to_numpy() * np.exp(oof)
    final = lgb.train(cfg["lightgbm"], dados, num_boost_round=rodadas)
    comp = comparar_deviance(df, df["mu_manual"].to_numpy(), mu_gbm, cfg["bootstrap"], cfg["semente"])
    return {"rodadas": rodadas, "oof_log_ajuste": oof, "mu_gbm_oof": mu_gbm, "modelo": final, "comparacao": comp,
            "X": X, "niveis": niveis, "variaveis": cfg["variaveis"], "categoricas": cfg["categoricas"]}


def prever(resultado: dict, df: pd.DataFrame) -> np.ndarray:
    """Previsão do LightGBM final em outra base (ex.: o teste), com a codificação do treino.
    Devolve mu = mu_manual x exp(ajuste)."""
    X = _matriz(df, resultado["variaveis"], resultado["categoricas"], resultado["niveis"])
    return df["mu_manual"].to_numpy() * np.exp(resultado["modelo"].predict(X, raw_score=True))


def comparar_deviance(df, mu_ref, mu_novo, n_boot: int, semente: int) -> dict:
    """Diferença de deviance fora da amostra (novo - referência), por grupo, com IC de 95% por bootstrap
    pareado (implementação única em `avaliacao.diferenca_deviance`)."""
    from .avaliacao import deviance_por_unidade, diferenca_deviance
    g = df["id_estipulante"].to_numpy()
    d_ref, d_novo = deviance_por_unidade(df["N"], mu_ref, g), deviance_por_unidade(df["N"], mu_novo, g)
    r = diferenca_deviance(d_novo, d_ref, n_boot, semente)
    return {"deviance_manual": float(d_ref.sum()), "deviance_gbm": float(d_novo.sum()), "diferenca": r["diferenca"],
            "ic95": (r["ic95_inf"], r["ic95_sup"]), "melhora_significativa": bool(r["ic95_sup"] < 0)}


def contribuicoes(modelo, X: pd.DataFrame) -> pd.DataFrame:
    """Contribuições SHAP (escala log) de cada variável, pelo TreeSHAP nativo do LightGBM.
    A soma das contribuições (mais o termo constante) é o ajuste em log sobre a taxa manual."""
    c = modelo.predict(X, pred_contrib=True)
    return pd.DataFrame(c[:, :-1], columns=X.columns, index=X.index).assign(constante=c[:, -1])


def importancia(contrib: pd.DataFrame, peso: np.ndarray) -> pd.Series:
    """Média ponderada (pelos esperados) do |SHAP| de cada variável: quanto ela mexe na mortalidade
    prevista, em log, em relação à taxa manual."""
    c = contrib.drop(columns="constante")
    return pd.Series(np.average(np.abs(c), axis=0, weights=peso), index=c.columns).sort_values(ascending=False)


def testar_candidato_glmm(df: pd.DataFrame, cfg_manual: dict, fixos_base: list[str], termos: list[str],
                          categoricas_extras: list[str] = (), numericas_extras: list[str] = ()) -> pd.DataFrame:
    """Fecha o ciclo: testa no GLMM da taxa manual (glmmTMB) um termo sugerido pelo diagnóstico.

    Roda direto nas linhas da base de diagnóstico (a Poisson é invariante à agregação), com a mesma
    estrutura de efeitos aleatórios da manual. `termos` usa sintaxe de fórmula do R; o teste é a razão
    de verossimilhança de retirar o último termo da lista."""
    from .glmm import ajustar_glmm
    from .manual import CATEGORICAS_BASE
    d = df.copy()
    d["log_capital_c"] = d["log_capital"] - np.log(cfg_manual["capital_referencia"])
    d["cnae_divisao"] = d["cnae_divisao"].astype("string").fillna("ND")
    base_sec = d.groupby("cnae_secao")["E"].sum().idxmax()
    r = ajustar_glmm(d, fixos=list(fixos_base) + list(termos), aleatorios=cfg_manual["modelo"]["aleatorios"],
                     categoricas=CATEGORICAS_BASE + list(categoricas_extras),
                     numericas=["log_capital_c"] + list(numericas_extras),
                     niveis_base={"cnae_secao": str(base_sec)}, testar_fixos=[termos[-1]])
    t = r.testes.copy()
    coef = r.fixos[r.fixos["nome"].str.contains(termos[-1].split(":")[0].split("(")[0], regex=False)]
    t["coeficientes"] = [dict(zip(coef["nome"], np.round(np.exp(coef["estimativa"]), 4)))] * len(t)
    return t


def tabela_interacoes(modelo, apenas_fortes: bool = False) -> pd.DataFrame:
    """Força das interações no LightGBM, na linha da tabela do framework da SOA (seção 4.2.3, pacote EIX):
    cada divisão cujo nó pai divide por outra variável forma um par; somam-se o ganho (`sumGain`) e o número de
    observações do nó filho (`sumCover`), e conta-se a frequência. `apenas_fortes=True` mantém só os pares em que
    o ganho do filho supera o do pai (a "interação forte" do EIX)."""
    t = modelo.trees_to_dataframe()
    n = t[t["split_feature"].notna()].set_index(["tree_index", "node_index"])
    filhos = n[n["parent_index"].notna()].reset_index()
    pais = n[["split_feature", "split_gain"]].rename(columns={"split_feature": "f_pai", "split_gain": "g_pai"})
    m = filhos.merge(pais, left_on=["tree_index", "parent_index"], right_index=True, how="inner")
    m = m[m["split_feature"] != m["f_pai"]]
    if apenas_fortes:
        m = m[m["split_gain"] > m["g_pai"]]
    m["par"] = [":".join(sorted(p)) for p in zip(m["split_feature"], m["f_pai"])]
    g = m.groupby("par").agg(sumGain=("split_gain", "sum"), sumCover=("count", "sum"), frequency=("split_gain", "size"))
    g["meanCover"], g["meanGain"] = g["sumCover"] / g["frequency"], g["sumGain"] / g["frequency"]
    return g.sort_values("sumGain", ascending=False).reset_index().rename(columns={"par": "interacao"})


def grafico_shap(ax, contrib: pd.DataFrame, df: pd.DataFrame, var: str, obs: str = "N", base: str = "mu_manual",
                 titulo: str | None = None):
    """Boxplot dos valores SHAP de uma variável por nível (SOA, seção 4.2.4): losango vermelho = SHAP médio;
    quadrado azul = log(observado / base) do nível, centrado na média geral."""
    niveis = sorted(df[var].astype(str).unique())
    x = df[var].astype(str).to_numpy()
    dados = [contrib[var].to_numpy()[x == n] for n in niveis]
    ax.boxplot(dados, showfliers=False, widths=0.6)
    pos = np.arange(1, len(niveis) + 1)
    ax.plot(pos, [d.mean() for d in dados], "D", color="#c00000", ms=5, label="SHAP médio")
    g = df.groupby(df[var].astype(str))[[obs, base]].sum().reindex(niveis)
    geral = np.log(df[obs].sum() / df[base].sum())
    ax.plot(pos, np.log(g[obs] / g[base]) - geral, "s", color="#1f4e79", ms=5, label="log(observado / base)")
    ax.axhline(0, color="grey", lw=0.6, ls=":")
    ax.set_xticks(pos, niveis, rotation=45, ha="right", fontsize=7)
    ax.set_title(titulo or var); ax.legend(fontsize=7)
