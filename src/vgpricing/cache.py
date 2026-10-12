"""Cache simples de resultados demorados (como o `bUseCache` do framework da SOA): reexecutar um notebook não
recalcula o que já foi calculado com os mesmos dados. Apague a pasta para forçar o recálculo."""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path


def em_cache(nome: str, funcao, pasta: Path, chave: str = "", invalidar: bool = False):
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(chave.encode()).hexdigest()[:10]
    arq = pasta / f"{nome}_{h}.pkl"
    if arq.exists() and not invalidar:
        with open(arq, "rb") as f:
            return pickle.load(f)
    resultado = funcao()
    with open(arq, "wb") as f:
        pickle.dump(resultado, f)
    return resultado
