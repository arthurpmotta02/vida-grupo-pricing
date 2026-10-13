import numpy as np
import pandas as pd
import pytest

from vgpricing.glmm import r_disponivel

pytestmark = pytest.mark.skipif(not r_disponivel(), reason="R não disponível")


def _dados(n_niveis=30, por_nivel=40, efeito_sd=0.3, semente=1):
    rng = np.random.default_rng(semente)
    ef = rng.normal(0, efeito_sd, n_niveis)
    linhas = []
    for j in range(n_niveis):
        for _ in range(por_nivel if j % 3 else 3):                  # 1/3 dos níveis com pouca experiência
            E = rng.uniform(0.5, 3)
            linhas.append({"nivel": f"L{j:02d}", "E": E, "N": rng.poisson(E * np.exp(ef[j])), "ef": ef[j]})
    d = pd.DataFrame(linhas)
    d["log_E"] = np.log(d["E"])
    return d


def test_ridge_sem_padronizar_encolhe_mais_os_niveis_raros():
    """Com padronizar=False (opção, não o padrão do glmnet), cada coeficiente tem a mesma penalidade e os níveis
    com pouca experiência ficam mais perto do offset, como na credibilidade."""
    from vgpricing.elasticnet import ajustar_elastic_net
    d = _dados()
    r = ajustar_elastic_net(d, {}, ["nivel"], ["nivel"], [], "log_E", alpha=0.0, nfolds=5, padronizar=False)
    b = r.coeficientes.drop("(Intercept)")
    contagem = d.groupby("nivel").size()
    raros = contagem[contagem == 3].index
    # níveis raros (3 linhas) ficam mais encolhidos que os demais em relação ao efeito verdadeiro
    verdade = d.groupby("nivel")["ef"].first()
    fator = lambda idx: np.abs(b[[f"nivel{i}" for i in idx]].to_numpy()).mean() / np.abs(verdade[idx]).mean()
    assert fator(raros) < fator(contagem.index.difference(raros))
    assert r.previsoes["treino"].shape[0] == len(d)


def test_priori_correta_no_offset_melhora_teste_quando_ha_pouca_experiencia():
    from vgpricing.avaliacao import deviance_poisson
    from vgpricing.elasticnet import ajustar_elastic_net
    d = _dados(por_nivel=6, semente=3)
    teste = _dados(por_nivel=200, semente=3).assign(N=lambda x: np.random.default_rng(9).poisson(x["E"] * np.exp(x["ef"])))
    for df in (d, teste):
        df["log_E_priori"] = df["log_E"] + df["ef"]                 # estudo de mercado que acertou a forma
    sem = ajustar_elastic_net(d, {"teste": teste}, ["nivel"], ["nivel"], [], "log_E", alpha=0.0, nfolds=5)
    com = ajustar_elastic_net(d, {"teste": teste}, ["nivel"], ["nivel"], [], "log_E_priori", alpha=0.0, nfolds=5)
    assert deviance_poisson(teste["N"], com.previsoes["teste"]) < deviance_poisson(teste["N"], sem.previsoes["teste"])


def test_fator_de_um_nivel_e_uma_coluna_so():
    """Pouca experiência: a seção tem um nível só. O fator sai (com aviso) e o ajuste roda com uma coluna."""
    from vgpricing.elasticnet import ajustar_elastic_net
    d = _dados().assign(secao="OUTRAS", lc=lambda x: np.random.default_rng(0).normal(0, 1, len(x)))
    with pytest.warns(UserWarning, match="único nível"):
        r = ajustar_elastic_net(d, {}, ["lc", "secao"], ["secao"], ["lc"], "log_E", alpha=0.0, nfolds=3)
    assert "lc" in r.coeficientes.index and np.isfinite(r.coeficientes).all()
