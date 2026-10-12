"""GLM de Poisson com penalização elastic net, via R (glmnet), com offset.

Uso no projeto (SOA, 2024, seção 4.3; CAS Monograph 5, seção 10.5):
- candidato na comparação de modelos (o melhor no teste do framework da SOA);
- "priori" de um estudo de mercado: a forma do estudo entra no offset, e os coeficientes penalizados medem o
  desvio da carteira em relação a ela. Penalização ridge (alpha = 0) = priori normal centrada no estudo; o
  lambda escolhido por validação cruzada equivale ao k da credibilidade.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAIZ


@dataclass
class ResultadoEN:
    coeficientes: pd.Series          # nome -> valor (escala log); "(Intercept)" incluído
    lambda_escolhido: float
    lambda_min: float
    lambda_1se: float
    caminho: pd.DataFrame            # lambda, deviance de validação cruzada, erro-padrão, coeficientes não nulos
    previsoes: dict[str, np.ndarray]  # "treino" e cada conjunto pedido
    especificacao: dict


def ajustar_elastic_net(treino: pd.DataFrame, prever: dict[str, pd.DataFrame], termos: list[str],
                        categoricas: list[str], numericas: list[str], offset: str, alpha: float = 0.0,
                        nfolds: int = 5, foldid: str | None = None, lambda_: str = "lambda.min",
                        semente: int = 2026, padronizar: bool = True, timeout: int = 1800) -> ResultadoEN:
    """`termos` usa sintaxe de fórmula do R (ex.: ["faixa_capital", "atividade", "sexo:faixa_idade"]).
    `offset` é o nome da coluna com o log do esperado (tábua + priori, se houver). `foldid`: coluna com a
    partição da validação cruzada (ex.: por estipulante); sem ela, partição aleatória em `nfolds`."""
    colunas = sorted(set(["N", offset, *categoricas, *numericas] + ([foldid] if foldid else [])))
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        treino[colunas].to_csv(t / "treino.csv", index=False)
        caminhos = {}
        for nome, df in prever.items():
            df[colunas if foldid is None else [c for c in colunas if c != foldid]].assign(
                **({foldid: 0} if foldid else {})).to_csv(t / f"{nome}.csv", index=False)
            caminhos[nome] = str(t / f"{nome}.csv")
        spec = {"treino": str(t / "treino.csv"), "prever": caminhos, "termos": termos, "categoricas": categoricas,
                "numericas": numericas, "offset": offset, "alpha": alpha, "nfolds": nfolds, "foldid": foldid,
                "lambda": lambda_, "semente": semente, "padronizar": padronizar, "pasta_saida": str(t),
                "saida": str(t / "saida.json")}
        (t / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
        r = subprocess.run(["Rscript", str(RAIZ / "r" / "ajustar_glmnet.R"), str(t / "spec.json")],
                           capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError(f"glmnet falhou:\n{r.stderr[-3000:]}")
        o = json.loads((t / "saida.json").read_text(encoding="utf-8"))
        prev = {nome: pd.read_csv(arq)["mu"].to_numpy() for nome, arq in (o.get("previsoes") or {}).items()}
        prev["treino"] = pd.read_csv(t / "prev_treino.csv")["mu"].to_numpy()
    c = o["coeficientes"]
    caminho = pd.DataFrame(o["caminho"])
    return ResultadoEN(coeficientes=pd.Series(c["valor"], index=c["nome"], dtype=float),
                       lambda_escolhido=float(o["lambda"]), lambda_min=float(o["lambda_min"]),
                       lambda_1se=float(o["lambda_1se"]), caminho=caminho, previsoes=prev, especificacao=spec)
