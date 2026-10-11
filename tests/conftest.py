import copy

import numpy as np
import pandas as pd
import pytest

from vgpricing.config import carregar_config
from vgpricing.tabua import Tabua, tabua_placeholder


@pytest.fixture(scope="session")
def cfg_base():
    return carregar_config("base")


@pytest.fixture(scope="session")
def mapeamentos():
    return carregar_config("colunas")


@pytest.fixture(scope="session")
def cfg_sint_pequeno():
    c = copy.deepcopy(carregar_config("sintetico"))
    c["n_estipulantes"] = 30
    c["tamanho"] = {"min": 30, "max": 1500}
    return c


@pytest.fixture(scope="session")
def tabua():
    return tabua_placeholder()


@pytest.fixture
def tabua_constante():
    """q_x = 0,001 (M) e 0,002 (F) em todas as idades: facilita contas à mão."""
    idades = np.arange(0, 117)
    qx = pd.DataFrame({"M": 0.001, "F": 0.002}, index=pd.Index(idades, name="idade"))
    return Tabua(nome="teste", qx=qx, fonte="teste")


@pytest.fixture(scope="session")
def carteira(cfg_sint_pequeno, tabua):
    from vgpricing.sintetico import gerar_carteira
    return gerar_carteira(cfg_sint_pequeno, tabua)
