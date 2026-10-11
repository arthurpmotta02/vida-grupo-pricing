"""Modelos de frequência de óbitos via R (glmmTMB), chamado a partir do Python.

glmmTMB é o único pacote de modelagem do projeto: ajusta o GLMM da taxa manual e também os
modelos de comparação (Poisson e Binomial Negativa sem efeitos aleatórios), com as
log-verossimilhanças na mesma base. O Python organiza os dados, chama `r/ajustar_glmm.R` com
`Rscript` e lê o resultado (troca por CSV/JSON).

Modelo (escala log, offset = óbitos esperados pela tábua):
    log mu = log E + beta_0 + beta' x + u_secao:divisao + u_estipulante
com u ~ Normal(0, sigma^2) independentes. O efeito de divisão é a credibilidade de cada divisão CNAE
em torno da sua seção (efeito fixo); o de estipulante é a heterogeneidade entre grupos.

Média condicional x média marginal: com efeito Normal na escala log, um nível desconhecido
(grupo novo, divisão não vista) tem fator esperado exp(sigma^2 / 2), não 1. A previsão para um
grupo novo usa essa correção; sem ela, a taxa manual sairia sistematicamente baixa.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAIZ

SCRIPT = RAIZ / "r" / "ajustar_glmm.R"


def r_disponivel() -> bool:
    if shutil.which("Rscript") is None:
        return False
    teste = subprocess.run(["Rscript", "-e", "suppressMessages({library(glmmTMB);library(actuar);library(jsonlite)})"],
                           capture_output=True, text=True)
    return teste.returncode == 0


@dataclass
class ResultadoGLMM:
    formula: str
    fixos: pd.DataFrame                      # nome, estimativa, ep
    vcov: np.ndarray
    variancias: dict[str, float]             # por termo aleatório
    aleatorios: dict[str, pd.DataFrame]      # nivel, modo, sd_condicional
    loglik: float
    aic: float
    gl: int
    convergiu: bool
    testes: pd.DataFrame
    especificacao: dict
    ic_variancias: pd.DataFrame | None = None
    extra: dict = field(default_factory=dict)

    def coef(self, nome: str, padrao: float = 0.0) -> float:
        s = self.fixos.set_index("nome")["estimativa"]
        return float(s.get(nome, padrao))


def ajustar_glmm(cel: pd.DataFrame, fixos: list[str], aleatorios: list[str],
                 categoricas: list[str], numericas: list[str], niveis_base: dict | None = None,
                 testar_fixos: list[str] | None = None, testar_aleatorios: list[str] | None = None,
                 familia: str = "poisson", timeout: int = 900) -> ResultadoGLMM:
    """Ajusta o modelo no R. `fixos` e `aleatorios` usam a sintaxe de fórmula do R, por exemplo
    fixos=["log_capital_c", "cnae_secao"], aleatorios=["(1|id_estipulante)"] (lista vazia = GLM).
    familia: "poisson" ou "nbinom2" (Binomial Negativa)."""
    if not r_disponivel():
        raise RuntimeError("R com glmmTMB, actuar e jsonlite não encontrado. Rode `Rscript r/instalar_pacotes.R`.")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cols = sorted(set(["N", "E", *categoricas, *numericas]))
        cel[cols].to_csv(tmp / "celulas.csv", index=False)
        spec = {"entrada": str(tmp / "celulas.csv"), "saida": str(tmp / "saida.json"),
                "fixos": list(fixos), "aleatorios": list(aleatorios),
                "categoricas": list(categoricas), "numericas": list(numericas),
                "niveis_base": niveis_base or {}, "testar_fixos": list(testar_fixos or []),
                "testar_aleatorios": list(testar_aleatorios or []), "familia": familia}
        (tmp / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
        r = subprocess.run(["Rscript", str(SCRIPT), str(tmp / "spec.json")], capture_output=True, text=True,
                           timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError(f"Falha no ajuste em R:\n{r.stderr[-3000:]}")
        out = json.loads((tmp / "saida.json").read_text(encoding="utf-8"))
    m = out["modelo"]
    _lista = lambda x: x if isinstance(x, list) else [x]
    fixos_df = pd.DataFrame({"nome": _lista(m["fixos"]["nome"]), "estimativa": _lista(m["fixos"]["estimativa"]),
                             "ep": _lista(m["fixos"]["ep"])})
    ale = {}
    for g, d in (m.get("aleatorios") or {}).items():
        ale[g] = pd.DataFrame({"nivel": _lista(d["nivel"]), "modo": _lista(d["modo"]),
                               "sd_condicional": _lista(d["sd_condicional"])})
    variancias = {k: float(v[0] if isinstance(v, list) else v) for k, v in (m.get("variancias") or {}).items()}
    testes = pd.DataFrame(out.get("testes") or [], columns=["termo", "tipo", "lr", "gl", "p_valor"])
    ic = m.get("ic_variancias")
    ic_df = None
    if ic:
        ic_df = pd.DataFrame({"termo": _lista(ic["termo"]), "var_inf": _lista(ic["var_inf"]),
                              "var_sup": _lista(ic["var_sup"])})
    return ResultadoGLMM(formula=m["formula"], fixos=fixos_df, vcov=np.atleast_2d(np.array(m["vcov_fixos"], float)),
                         variancias=variancias, aleatorios=ale, loglik=float(m["loglik"]), aic=float(m["aic"]),
                         gl=int(m["gl"]), convergiu=bool(m["convergiu"]), testes=testes, especificacao=spec,
                         ic_variancias=ic_df)


# --------------------------------------------------------------------------- previsão

def _x_fixos(res: ResultadoGLMM, df: pd.DataFrame) -> np.ndarray:
    """Preditor linear dos efeitos fixos, reconstruído a partir dos nomes do R
    (ex.: '(Intercept)', 'log_capital_c', 'faixa_idade30-39', 'sexoM')."""
    eta = np.zeros(len(df))
    cats = res.especificacao["categoricas"]
    for nome, b in zip(res.fixos["nome"], res.fixos["estimativa"]):
        if nome == "(Intercept)":
            eta += b
        elif nome in df.columns and nome not in cats:
            eta += b * df[nome].to_numpy(float)
        else:
            for c in cats:
                if nome.startswith(c) and c in res.especificacao["fixos"]:
                    eta += b * (df[c].astype(str).to_numpy() == nome[len(c):])
                    break
    return eta


def prever(res: ResultadoGLMM, df: pd.DataFrame, usar_efeito_grupo: bool = False) -> np.ndarray:
    """Óbitos previstos mu = E x exp(eta). Para cada termo aleatório, usa o modo do nível quando o
    nível é conhecido; quando não é (ou, para estipulante, quando usar_efeito_grupo=False, que é o
    caso da cotação de grupo novo), soma sigma^2/2 (média marginal)."""
    eta = np.log(df["E"].to_numpy(float)) + _x_fixos(res, df)
    for termo, var in res.variancias.items():
        cols = termo.split(":")
        if termo == "id_estipulante" and not usar_efeito_grupo:
            eta += var / 2
            continue
        chave = df[cols].astype(str).agg(":".join, axis=1) if len(cols) > 1 else df[cols[0]].astype(str)
        modos = res.aleatorios[termo].set_index("nivel")["modo"]
        u = chave.map(modos)
        eta += np.where(u.isna(), var / 2, u.fillna(0.0).to_numpy())
    return np.exp(eta)
