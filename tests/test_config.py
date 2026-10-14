"""Confere que os arquivos de configuração têm as chaves que o código e os notebooks usam (uma edição errada no
YAML quebraria um notebook sem quebrar nenhum outro teste)."""
import pytest

from vgpricing.config import carregar_config

OBRIGATORIAS = {
    "modelagem": {"semente_particao", "fracao_teste", "minimo_esperados_nivel", "elastic_net", "lightgbm_soa",
                  "lightgbm_carteira"},
    "manual": {"meses_maturacao", "capital_referencia", "faixas", "modelo", "validacao_cruzada", "fora_do_tempo", "decisao"},
    "renovacao": {"anos_experiencia_max", "pooling", "credibilidade", "reajuste", "regra_contratual", "backtest"},
    "prioris": {"capital_referencia_usd", "soa_para_cnae"},
}


@pytest.mark.parametrize("nome", sorted(OBRIGATORIAS))
def test_chaves_obrigatorias(nome):
    assert OBRIGATORIAS[nome] <= set(carregar_config(nome))


def test_blocos_de_modelagem():
    c = carregar_config("modelagem")
    assert {"alpha", "lambda", "folds", "semente", "interacoes_do_lightgbm", "padronizar"} <= set(c["elastic_net"])
    for b in ("lightgbm_soa", "lightgbm_carteira"):
        assert {"parametros", "rodadas_max", "parada_antecipada", "folds", "semente", "bootstrap"} <= set(c[b])
        assert c[b]["parametros"]["objective"] == "poisson"
    assert set(carregar_config("manual")["decisao"]["ordem_preferencia"]) <= {
        "ridge com priori SOA", "AGLM com priori SOA", "GLMM", "GLM", "plano atual (tábua × A/E)"}
