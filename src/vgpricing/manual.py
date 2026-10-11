"""Etapa 2: taxa manual.

Modelo de frequência de óbitos relativo à tábua:
    N_c ~ Poisson(mu_c),  log mu_c = log E_c + beta_0 + soma dos efeitos dos fatores
onde E_c = óbitos esperados pela BR-EMS na célula c. exp(beta) é um fator A/E relativo.

Três estimadores para a mesma estrutura:
1. GLM Poisson (statsmodels): referência, saída familiar para auditoria.
2. GLM Poisson com ridge (implementação própria): encolhe níveis pouco expostos para a média.
3. Poisson-Gama com efeito aleatório de estipulante (máxima verossimilhança marginal exata,
   implementação própria): u_i ~ Gama(média 1, variância 1/a). Separa o efeito dos fatores do
   efeito do grupo e estima a, que é o k de credibilidade em unidades de óbitos esperados:
       E[u_i | dados] = (a + N_i) / (a + M_i) = Z_i * N_i/M_i + (1 - Z_i),  Z_i = M_i / (M_i + a).
   É a ligação direta com a fórmula de Bühlmann usada na renovação (Etapa 3).

Por que contagem (e não valor) na taxa manual: estimar em R$ deixa os fatores reféns de poucos
sinistros de capital alto. O valor entra na renovação, com pooling de sinistros grandes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import digamma, gammaln

from .exposicao import exposicao_detalhada
from .tabua import Tabua


# --------------------------------------------------------------------------- base de modelagem

def _faixa(valores, faixa_cfg) -> pd.Series:
    return pd.cut(valores, bins=faixa_cfg["limites"], labels=faixa_cfg["rotulos"], right=False).astype(str)


def base_modelagem(vidas: pd.DataFrame, sinistros: pd.DataFrame, estipulantes: pd.DataFrame, tabua: Tabua,
                   data_referencia, cfg_manual: dict, dias_por_ano: float = 365.25) -> tuple[pd.DataFrame, dict]:
    """Tabela de células (estipulante x idade x sexo x setor x capital x tamanho) com E e N.

    Usa só a experiência maturada: exposição e ocorrências até data_referencia - meses_maturacao.
    Cada sinistro herda a célula do pedaço de exposição em que a vida estava na data do óbito.
    """
    corte = pd.Timestamp(data_referencia) - pd.DateOffset(months=cfg_manual["meses_maturacao"])
    det = exposicao_detalhada(vidas, estipulantes, tabua, corte, dias_por_ano)
    fx = cfg_manual["faixas"]
    det = det.merge(estipulantes[["id_estipulante", "setor"]], on="id_estipulante", how="left")
    det["sexo"] = det["sexo"].astype("string").fillna("ND")
    det["faixa_idade"] = _faixa(det["idade"], fx["idade"])
    det["faixa_capital"] = _faixa(det["capital"], fx["capital"])
    porte = det.groupby("id_estipulante").apply(
        lambda g: g["exposicao"].sum() / g["ano_apolice"].nunique(), include_groups=False).rename("vidas_medias")
    det = det.merge(porte, on="id_estipulante")
    det["faixa_tamanho"] = _faixa(det["vidas_medias"], fx["tamanho"])

    # sinistros -> pedaço de exposição vigente na ocorrência
    s = sinistros[sinistros["data_ocorrencia"] <= corte]
    m = s[["id_sinistro", "id_estipulante", "id_vida", "data_ocorrencia"]].merge(
        det[["id_estipulante", "id_vida", "ini", "fim", "faixa_idade", "sexo", "setor", "faixa_capital",
             "faixa_tamanho", "capital"]], on=["id_estipulante", "id_vida"], how="inner")
    m = m[(m["data_ocorrencia"] >= m["ini"]) & (m["data_ocorrencia"] <= m["fim"])].drop_duplicates("id_sinistro")
    sem_celula = sorted(set(s["id_sinistro"]) - set(m["id_sinistro"]))

    chave = ["id_estipulante", "faixa_idade", "sexo", "setor", "faixa_capital", "faixa_tamanho"]
    cel = det.groupby(chave, as_index=False, observed=True).agg(
        E=("obitos_esperados", "sum"), exposicao=("exposicao", "sum"), capital_ano=("capital_ano", "sum"),
        valor_esperado=("valor_esperado", "sum"))
    obs = m.groupby(chave, as_index=False, observed=True).agg(N=("id_sinistro", "count"), valor_observado=("capital", "sum"))
    cel = cel.merge(obs, on=chave, how="left").fillna({"N": 0, "valor_observado": 0.0})
    cel["N"] = cel["N"].astype(int)
    n_nd = int((cel["sexo"] == "ND").sum())
    cel = cel[(cel["sexo"] != "ND") & (cel["E"] > 0)].reset_index(drop=True)
    info = {"corte_experiencia": corte.date().isoformat(), "sinistros_no_periodo": int(len(s)),
            "sinistros_sem_celula": len(sem_celula), "ids_sem_celula": sem_celula[:50],
            "celulas_sexo_ND_removidas": n_nd, "celulas": int(len(cel)),
            "obitos": int(cel["N"].sum()), "esperados": float(cel["E"].sum())}
    return cel, info


# --------------------------------------------------------------------------- desenho

@dataclass
class Desenho:
    fatores: list[str]
    niveis: dict[str, list[str]]
    base: dict[str, str]
    colunas: list[str] = field(default_factory=list)
    completa: bool = False      # True: todas as dummies (ridge); False: codificação de referência

    @classmethod
    def criar(cls, df: pd.DataFrame, fatores: list[str], completa: bool = False) -> "Desenho":
        niveis, base = {}, {}
        for f in fatores:
            exp = df.groupby(f, observed=True)["E"].sum().sort_values(ascending=False)
            niveis[f] = sorted(exp.index.astype(str))
            base[f] = str(exp.index[0])           # nível de maior exposição
        d = cls(fatores, niveis, base, completa=completa)
        d.colunas = ["(intercepto)"] + [f"{f}[{n}]" for f in fatores for n in niveis[f]
                                         if completa or n != base[f]]
        return d

    def matriz(self, df: pd.DataFrame) -> np.ndarray:
        X = np.zeros((len(df), len(self.colunas)))
        X[:, 0] = 1.0
        pos = {c: j for j, c in enumerate(self.colunas)}
        for f in self.fatores:
            v = df[f].astype(str).to_numpy()
            for n in self.niveis[f]:
                c = f"{f}[{n}]"
                if c in pos:
                    X[v == n, pos[c]] = 1.0
        return X


def _deviance_poisson(N, mu) -> float:
    N, mu = np.asarray(N, float), np.asarray(mu, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(N > 0, N * np.log(N / mu), 0.0)
    return float(2 * np.sum(t - (N - mu)))


# --------------------------------------------------------------------------- 1. GLM Poisson

@dataclass
class ResultadoModelo:
    nome: str
    desenho: Desenho
    beta: np.ndarray
    ep: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    def mu(self, df: pd.DataFrame) -> np.ndarray:
        return df["E"].to_numpy() * np.exp(self.desenho.matriz(df) @ self.beta)

    def fatores(self) -> pd.DataFrame:
        """Fator multiplicativo de cada nível (base = 1) e A/E base (intercepto)."""
        linhas = []
        coef = dict(zip(self.desenho.colunas, self.beta))
        ep = dict(zip(self.desenho.colunas, self.ep)) if self.ep is not None else {}
        for f in self.desenho.fatores:
            b_base = coef.get(f"{f}[{self.desenho.base[f]}]", 0.0)
            for n in self.desenho.niveis[f]:
                b = coef.get(f"{f}[{n}]", 0.0) - b_base     # relativo ao nível base
                e = ep.get(f"{f}[{n}]", np.nan) if n != self.desenho.base[f] else 0.0
                linhas.append({"fator": f, "nivel": n, "base": n == self.desenho.base[f],
                               "relatividade": np.exp(b), "ep_log": e,
                               "ic_inf": np.exp(b - 1.96 * e) if e == e else np.nan,
                               "ic_sup": np.exp(b + 1.96 * e) if e == e else np.nan})
        out = pd.DataFrame(linhas)
        b0 = coef["(intercepto)"] + sum(coef.get(f"{f}[{self.desenho.base[f]}]", 0.0) for f in self.desenho.fatores)
        out.attrs["ae_base"] = float(np.exp(b0))
        return out


def ajustar_glm(df: pd.DataFrame, fatores: list[str]) -> ResultadoModelo:
    import statsmodels.api as sm
    d = Desenho.criar(df, fatores)
    X = d.matriz(df)
    mod = sm.GLM(df["N"].to_numpy(), X, family=sm.families.Poisson(), offset=np.log(df["E"].to_numpy()))
    r = mod.fit()
    gl = len(df) - X.shape[1]
    return ResultadoModelo("GLM Poisson", d, r.params, r.bse, {
        "deviance": float(r.deviance), "pearson_chi2": float(r.pearson_chi2), "gl": gl,
        "dispersao_pearson": float(r.pearson_chi2 / gl), "aic": float(r.aic), "loglik": float(r.llf),
        "statsmodels": r})


def teste_binomial_negativa(df: pd.DataFrame, fatores: list[str], res_poisson: ResultadoModelo) -> dict:
    """NB2 nas células (statsmodels) e teste da razão de verossimilhança contra a Poisson.
    O parâmetro alpha está na fronteira sob H0, então o p-valor usa a mistura 50:50 de qui-quadrados."""
    import statsmodels.api as sm
    from scipy.stats import chi2
    X = res_poisson.desenho.matriz(df)
    nb = sm.NegativeBinomial(df["N"].to_numpy(), X, offset=np.log(df["E"].to_numpy()), loglike_method="nb2")
    r = nb.fit(start_params=np.append(res_poisson.beta, 0.1), disp=0, maxiter=500)
    lr = 2 * (r.llf - res_poisson.extra["loglik"])
    p = 0.5 * chi2.sf(max(lr, 0), 1)
    return {"alpha_nb2": float(r.params[-1]), "ep_alpha": float(r.bse[-1]), "estatistica_lr": float(lr),
            "p_valor": float(p), "aic_nb": float(r.aic), "aic_poisson": res_poisson.extra["aic"]}


# --------------------------------------------------------------------------- 2. Ridge

def ajustar_ridge(df: pd.DataFrame, fatores: list[str], lam: float) -> ResultadoModelo:
    """Poisson penalizada: min  sum(mu - N log mu) + lam/2 * sum(beta_j^2), j != intercepto.
    Codificação completa (todas as dummies): o ridge encolhe cada nível para a média da carteira,
    sem privilegiar um nível base. Com lam = 0 cai na codificação de referência (= GLM)."""
    if lam == 0:
        r = ajustar_glm(df, fatores)
        r.nome, r.extra = "Ridge (lambda=0)", {"lambda": 0.0}
        return r
    d = Desenho.criar(df, fatores, completa=True)
    X, N, logE = d.matriz(df), df["N"].to_numpy(float), np.log(df["E"].to_numpy())
    pen = np.r_[0.0, np.full(X.shape[1] - 1, lam)]

    def f(b):
        eta = logE + X @ b
        mu = np.exp(eta)
        return float(np.sum(mu - N * eta) + 0.5 * np.sum(pen * b * b)), X.T @ (mu - N) + pen * b

    b0 = np.zeros(X.shape[1])
    b0[0] = np.log(N.sum() / np.exp(logE).sum())
    opt = minimize(f, b0, jac=True, method="L-BFGS-B", options={"maxiter": 2000, "gtol": 1e-8})
    return ResultadoModelo(f"Ridge (lambda={lam:g})", d, opt.x, None, {"lambda": lam, "convergiu": bool(opt.success)})


def _folds_por_grupo(df: pd.DataFrame, k: int, semente: int) -> np.ndarray:
    grupos = df["id_estipulante"].unique()
    rng = np.random.default_rng(semente)
    f = dict(zip(rng.permutation(grupos), np.arange(len(grupos)) % k))
    return df["id_estipulante"].map(f).to_numpy()


def validacao_cruzada(df: pd.DataFrame, ajustadores: dict, k: int = 5, semente: int = 2026) -> pd.DataFrame:
    """Deviance de Poisson fora da amostra, deixando GRUPOS inteiros de fora (simula cotar um grupo
    novo). Avaliada no total de cada grupo, que é o que a taxa manual precisa acertar."""
    fold = _folds_por_grupo(df, k, semente)
    linhas = []
    for nome, ajustar in ajustadores.items():
        dev = 0.0
        for i in range(k):
            tr, te = df[fold != i], df[fold == i]
            mu = ajustar(tr).mu(te) if ajustar is not None else te["E"].to_numpy()
            g = te.assign(mu=mu).groupby("id_estipulante")[["N", "mu"]].sum()
            dev += _deviance_poisson(g["N"], g["mu"])
        linhas.append({"modelo": nome, "deviance_fora_amostra": dev})
    out = pd.DataFrame(linhas)
    out["dif_vs_melhor"] = out["deviance_fora_amostra"] - out["deviance_fora_amostra"].min()
    return out


def escolher_lambda(df, fatores, lambdas, k=5, semente=2026) -> pd.DataFrame:
    aj = {lam: (lambda d, lam=lam: ajustar_ridge(d, fatores, lam)) for lam in lambdas}
    cv = validacao_cruzada(df, aj, k, semente).rename(columns={"modelo": "lambda"})
    return cv


# --------------------------------------------------------------------------- 3. Poisson-Gama

def _nll_poisson_gama(theta, X, N, logE, g_idx, n_grupos):
    b, loga = theta[:-1], theta[-1]
    a = np.exp(loga)
    eta = logE + X @ b
    mu = np.exp(eta)
    Ni = np.bincount(g_idx, weights=N, minlength=n_grupos)
    Mi = np.bincount(g_idx, weights=mu, minlength=n_grupos)
    ll = (np.sum(N * eta)
          + np.sum(gammaln(a + Ni) - gammaln(a) + a * np.log(a) - (a + Ni) * np.log(a + Mi)))
    w = ((a + Ni) / (a + Mi))[g_idx]                       # média a posteriori de u_i, por linha
    grad_b = X.T @ (N - w * mu)
    dll_da = np.sum(digamma(a + Ni) - digamma(a) + np.log(a) + 1 - np.log(a + Mi) - (a + Ni) / (a + Mi))
    return -ll, -np.r_[grad_b, a * dll_da]


def ajustar_poisson_gama(df: pd.DataFrame, fatores: list[str], grupo: str = "id_estipulante",
                         beta_inicial: np.ndarray | None = None) -> ResultadoModelo:
    d = Desenho.criar(df, fatores)
    X, N, logE = d.matriz(df), df["N"].to_numpy(float), np.log(df["E"].to_numpy())
    codigos, grupos = pd.factorize(df[grupo])
    b0 = beta_inicial if beta_inicial is not None else np.r_[np.log(N.sum() / np.exp(logE).sum()),
                                                             np.zeros(X.shape[1] - 1)]
    th0 = np.r_[b0, np.log(10.0)]
    args = (X, N, logE, codigos, len(grupos))
    opt = minimize(_nll_poisson_gama, th0, args=args, jac=True, method="L-BFGS-B",
                   options={"maxiter": 5000, "gtol": 1e-7})
    th = opt.x
    # erro-padrão: hessiana numérica do gradiente analítico
    h = 1e-5
    H = np.zeros((len(th), len(th)))
    for j in range(len(th)):
        e = np.zeros(len(th)); e[j] = h
        H[:, j] = (_nll_poisson_gama(th + e, *args)[1] - _nll_poisson_gama(th - e, *args)[1]) / (2 * h)
    H = (H + H.T) / 2
    cov = np.linalg.inv(H)
    ep = np.sqrt(np.clip(np.diag(cov), 0, None))
    a = float(np.exp(th[-1]))
    mu = np.exp(logE + X @ th[:-1])
    g = pd.DataFrame({"g": grupos[codigos], "N": N, "M": mu}).groupby("g")[["N", "M"]].sum()
    g["Z"] = g["M"] / (g["M"] + a)
    g["fator_grupo_posteriori"] = (a + g["N"]) / (a + g["M"])
    return ResultadoModelo("Poisson-Gama (efeito de estipulante)", d, th[:-1], ep[:-1], {
        "k": a, "ep_log_k": float(ep[-1]), "ic_k": (a * np.exp(-1.96 * ep[-1]), a * np.exp(1.96 * ep[-1])),
        "cv_efeito_grupo": float(1 / np.sqrt(a)), "loglik": float(-opt.fun), "convergiu": bool(opt.success),
        "grupos": g.reset_index().rename(columns={"g": grupo})})


# --------------------------------------------------------------------------- taxa pura manual

def tabela_fatores(res: ResultadoModelo, meta: dict) -> dict:
    """Exporta fatores para o cotador (YAML/JSON versionável)."""
    fat = res.fatores()
    return {"meta": {**meta, "modelo": res.nome, "ae_base": float(fat.attrs["ae_base"]),
                     "fatores_no_modelo": list(res.desenho.fatores)},
            "fatores": {f: {r.nivel: float(r.relatividade) for r in g.itertuples()}
                        for f, g in fat.groupby("fator", sort=False)},
            }


def taxa_pura_manual(vidas: pd.DataFrame, setor: str, tabela: dict, tabua: Tabua, data_calculo,
                     cfg_manual: dict) -> dict:
    """Taxa pura anual e mensal por mil de capital para um censo de vidas (cotação de grupo novo).

    Para cada vida: custo esperado = capital x q_x(idade na data de cálculo, sexo) x A/E base
    x produto dos fatores. Taxa = soma dos custos / soma dos capitais x 1000.
    """
    v = vidas.copy()
    data = pd.Timestamp(data_calculo)
    from .exposicao import idade_completa
    v["idade"] = idade_completa(v["data_nascimento"], pd.Series(data, index=v.index)).astype(int)
    v["q"] = tabua.q(v["idade"].to_numpy(), v["sexo"].astype(object).to_numpy())
    fx = cfg_manual["faixas"]
    v["faixa_idade"] = _faixa(v["idade"], fx["idade"])
    v["faixa_capital"] = _faixa(v["capital"], fx["capital"])
    v["setor"] = setor
    v["faixa_tamanho"] = str(_faixa(pd.Series([len(v)]), fx["tamanho"]).iloc[0])
    f = tabela["fatores"]
    mult = np.full(len(v), tabela["meta"]["ae_base"])
    nao_encontrados = {}
    for fator, mapa in f.items():
        col = v[fator].astype(str) if fator in v else pd.Series("", index=v.index)
        valores = col.map(mapa)
        if valores.isna().any():
            nao_encontrados[fator] = sorted(col[valores.isna()].unique())
            valores = valores.fillna(1.0)
        mult = mult * valores.to_numpy()
    v["q_ajustado"] = v["q"] * mult
    v["custo_esperado"] = v["capital"] * v["q_ajustado"]
    taxa_anual = 1000 * v["custo_esperado"].sum() / v["capital"].sum()
    return {"taxa_pura_anual_mil": float(taxa_anual), "taxa_pura_mensal_mil": float(taxa_anual / 12),
            "premio_puro_anual": float(v["custo_esperado"].sum()), "capital_total": float(v["capital"].sum()),
            "vidas": int(len(v)), "obitos_esperados": float(v["q_ajustado"].sum()),
            "niveis_sem_fator": nao_encontrados, "detalhe": v}


# --------------------------------------------------------------------------- seleção de fatores

def testes_lr_fatores(df: pd.DataFrame, fatores: list[str], completo: ResultadoModelo | None = None) -> pd.DataFrame:
    """Teste da razão de verossimilhança de cada fator no modelo Poisson-Gama (retira um fator
    por vez, mantendo os demais). Um fator só entra na tabela de preço com evidência nos dados."""
    from scipy.stats import chi2
    completo = completo or ajustar_poisson_gama(df, fatores)
    linhas = []
    for f in fatores:
        outros = [x for x in fatores if x != f]
        red = ajustar_poisson_gama(df, outros)
        lr = 2 * (completo.extra["loglik"] - red.extra["loglik"])
        gl = len(completo.desenho.niveis[f]) - 1
        linhas.append({"fator": f, "niveis": gl + 1, "estatistica_lr": lr, "gl": gl, "p_valor": chi2.sf(max(lr, 0), gl)})
    return pd.DataFrame(linhas)


def selecionar_fatores(df: pd.DataFrame, fatores: list[str], alfa: float = 0.05,
                       obrigatorios: tuple[str, ...] = ()) -> tuple[list[str], pd.DataFrame]:
    """Eliminação para trás: retira o fator menos significativo enquanto houver p > alfa.
    `obrigatorios` ficam no modelo por decisão de negócio, qualquer que seja o p-valor."""
    atuais, historico = list(fatores), []
    while True:
        lr = testes_lr_fatores(df, atuais)
        cand = lr[~lr["fator"].isin(obrigatorios)].sort_values("p_valor", ascending=False)
        if cand.empty or cand.iloc[0]["p_valor"] <= alfa:
            return atuais, pd.concat(historico + [lr.assign(passo=len(historico) + 1, acao="final")], ignore_index=True)
        sai = cand.iloc[0]["fator"]
        historico.append(lr.assign(passo=len(historico) + 1, acao=np.where(lr["fator"] == sai, "retirado", "")))
        atuais.remove(sai)
        if not atuais:
            return atuais, pd.concat(historico, ignore_index=True)


# --------------------------------------------------------------------------- persistência

def _para_python(x):
    """Converte tipos numpy/pandas em tipos nativos (YAML legível e sem tags de objeto)."""
    if isinstance(x, dict):
        return {str(k): _para_python(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_para_python(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, float) and np.isinf(x):
        return float("inf")
    return x


def salvar_tabela(tabela: dict, caminho) -> None:
    import yaml
    from pathlib import Path
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        yaml.safe_dump(_para_python(tabela), f, allow_unicode=True, sort_keys=False)


def carregar_tabela(caminho) -> dict:
    import yaml
    with open(caminho, encoding="utf-8") as f:
        return yaml.safe_load(f)
