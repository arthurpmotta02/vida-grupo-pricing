"""Ponte única Python -> R (Rscript). Os scripts ficam em `r/` e trocam dados por CSV e JSON (sem rpy2).

Pacotes R usados, um por tarefa: stats::glm (GLM), glmmTMB (GLMM), glmnet (elastic net e AGLM), mgcv (GAM) e
actuar (Bühlmann-Straub). Cotar com uma tabela pronta não exige R; estimar exige.
"""
from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from pathlib import Path

from .config import RAIZ

PACOTES = ("glmmTMB", "glmnet", "mgcv", "actuar", "jsonlite")


@lru_cache(maxsize=1)
def disponivel() -> bool:
    teste = "suppressMessages({" + ";".join(f"library({p})" for p in PACOTES) + "})"
    try:
        return subprocess.run(["Rscript", "-e", teste], capture_output=True, timeout=120).returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def rodar(script: str, *args, timeout: int = 1800) -> None:
    if not disponivel():
        raise RuntimeError(f"R com {', '.join(PACOTES)} não encontrado. Rode `Rscript r/instalar_pacotes.R`.")
    r = subprocess.run(["Rscript", str(RAIZ / "r" / script), *map(str, args)], capture_output=True, text=True,
                       timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{script} falhou:\n{r.stderr[-3000:]}")


def rodar_spec(script: str, spec: dict, pasta: Path, timeout: int = 1800) -> dict:
    """Grava a especificação em JSON na pasta temporária, roda o script e devolve o JSON de saída."""
    pasta = Path(pasta)
    spec = {**spec, "saida": str(pasta / "saida.json")}
    (pasta / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    rodar(script, pasta / "spec.json", timeout=timeout)
    return json.loads((pasta / "saida.json").read_text(encoding="utf-8"))
