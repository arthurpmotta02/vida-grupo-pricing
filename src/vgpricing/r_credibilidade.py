"""Conferência do Bühlmann-Straub em Python contra o actuar::cm (R)."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from .config import RAIZ


def buhlmann_straub_actuar(df: pd.DataFrame, grupo: str, periodo: str, x: str, w: str) -> dict:
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        d = df[[grupo, periodo, x, w]].rename(columns={grupo: "grupo", periodo: "periodo", x: "x", w: "w"})
        d = d[d["w"] > 0]
        d.to_csv(t / "e.csv", index=False)
        r = subprocess.run(["Rscript", str(RAIZ / "r" / "buhlmann_straub.R"), str(t / "e.csv"), str(t / "s.json")],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-2000:])
        o = json.loads((t / "s.json").read_text())
    o["k"] = o["s2"] / o["a"] if o["a"] > 0 else float("inf")
    o["Z"] = pd.Series(o.pop("cred"), index=[str(g) for g in o["grupos"]])
    return o
