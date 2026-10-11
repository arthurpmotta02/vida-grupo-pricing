"""Leitura de configuração YAML. Caminhos relativos são resolvidos a partir da raiz do repositório."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

RAIZ = Path(__file__).resolve().parents[2]


def carregar_config(nome: str, raiz: Path | None = None) -> dict[str, Any]:
    """Lê config/<nome>.yaml."""
    raiz = raiz or RAIZ
    caminho = raiz / "config" / f"{nome}.yaml"
    with open(caminho, encoding="utf-8") as f:
        return yaml.safe_load(f)


def caminho_repo(relativo: str | Path, raiz: Path | None = None) -> Path:
    p = Path(relativo)
    return p if p.is_absolute() else (raiz or RAIZ) / p
