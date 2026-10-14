"""GAM de Poisson com offset via R (mgcv; Wood, 2017): termos suaves (splines penalizados) com a suavidade escolhida
por REML (fREML no `bam`, a versão para bases grandes). No projeto, referência e diagnóstico da forma de variáveis contínuas (o log do capital é linear?)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from .r import rodar_spec


def ajustar_gam(treino: pd.DataFrame, prever: dict[str, pd.DataFrame], suaves: list[str], categoricas: list[str],
                k: int = 6, timeout: int = 1800) -> dict:
    cols = sorted(set(["N", "E", *suaves, *categoricas]))
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        treino[cols].to_csv(t / "treino.csv", index=False)
        caminhos = {}
        for nome, df in prever.items():
            df[cols].to_csv(t / f"{nome}.csv", index=False)
            caminhos[nome] = str(t / f"{nome}.csv")
        spec = {"treino": str(t / "treino.csv"), "prever": caminhos, "suaves": suaves, "categoricas": categoricas,
                "numericas": suaves, "k": k, "pasta_saida": str(t)}
        o = rodar_spec("ajustar_gam.R", spec, t, timeout)
        o["previsoes"] = {n: pd.read_csv(a)["mu"].to_numpy() for n, a in (o.get("previsoes") or {}).items()}
    o["curvas"] = {s: pd.DataFrame(c) for s, c in (o.get("curvas") or {}).items()}
    return o
