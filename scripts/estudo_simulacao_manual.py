"""Estudo de simulação da taxa manual: a complexidade muda o preço?

Gera R carteiras sintéticas independentes (mesmas hipóteses, sementes diferentes) e compara:
- erro dos fatores de setor contra a verdade do gerador (RMSE em log);
- cobertura dos intervalos de 95% dos fatores de setor (GLM ignora o agrupamento por estipulante);
- estimativa de k contra o valor implícito na hipótese do gerador;
- deviance fora da amostra deixando grupos inteiros de fora.

Uso:  python scripts/estudo_simulacao_manual.py [R] [inicio] [fim]
      -> resultados/estudo_simulacao_manual.csv (cada réplica também é gravada à parte,
         então o estudo pode ser rodado em lotes e retomado)
"""
import copy
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
warnings.filterwarnings("ignore", message=".*coluna lx.*")

from vgpricing import manual as M  # noqa: E402
from vgpricing.config import carregar_config  # noqa: E402
from vgpricing.sintetico import gerar_carteira, injetar_sujeira, tabelas_de_entrada  # noqa: E402
from vgpricing.tabua import obter_tabua  # noqa: E402
from vgpricing.validacao import validar  # noqa: E402


def rel_setor(res, base="servicos"):
    f = res.fatores()
    f = f[f["fator"] == "setor"].set_index("nivel")
    r = f["relatividade"] / f.loc[base, "relatividade"]
    # IC só faz sentido direto quando a base do modelo é a mesma da comparação
    ok = f.loc[base, "base"]
    return r, (f["ic_inf"] if ok else None), (f["ic_sup"] if ok else None)


def main(R=12, inicio=0, fim=None):
    base = carregar_config("base", RAIZ); cs0 = carregar_config("sintetico", RAIZ); cm = carregar_config("manual", RAIZ)
    tab = obter_tabua(base, RAIZ)
    F = cm["fatores"]
    verdade = pd.Series({s: v["fator_mortalidade"] for s, v in cs0["setores"].items()})
    verdade = verdade / verdade["servicos"]
    k_verdade = 1 / (np.exp(cs0["sigma_efeito_grupo"] ** 2) - 1)
    partes = RAIZ / "resultados" / "estudo_partes"
    partes.mkdir(parents=True, exist_ok=True)
    for rep in range(inicio, min(fim or R, R)):
        if (partes / f"rep_{rep:02d}.csv").exists():
            continue
        linhas = []
        t = time.time()
        cs = copy.deepcopy(cs0); cs["semente"] = 1000 + rep
        cart = gerar_carteira(cs, tab)
        suja, _ = injetar_sujeira(cart, cs)
        r = validar(tabelas_de_entrada(suja), base)
        cel, info = M.base_modelagem(r.dados["vidas"], r.dados["sinistros"], r.dados["estipulantes"], tab,
                                     base["data_referencia"], cm)
        glm = M.ajustar_glm(cel, F)
        pg = M.ajustar_poisson_gama(cel, F, beta_inicial=glm.beta)
        pg_setor = M.ajustar_poisson_gama(cel, ["setor"])
        cvl = M.escolher_lambda(cel, F, cm["ridge"]["lambdas"], cm["ridge"]["folds"], cm["semente_cv"])
        lam = float(cvl.loc[cvl["deviance_fora_amostra"].idxmin(), "lambda"])
        rid = M.ajustar_ridge(cel, F, lam)
        cv = M.validacao_cruzada(cel, {
            "BR-EMS pura": None,
            "GLM (todos os fatores)": lambda d: M.ajustar_glm(d, F),
            f"Ridge": lambda d: M.ajustar_ridge(d, F, lam),
            "Poisson-Gama (todos)": lambda d: M.ajustar_poisson_gama(d, F),
            "Poisson-Gama (só setor)": lambda d: M.ajustar_poisson_gama(d, ["setor"]),
        }, cm["ridge"]["folds"], cm["semente_cv"])
        modelos = {"GLM (todos os fatores)": glm, "Ridge": rid, "Poisson-Gama (todos)": pg,
                   "Poisson-Gama (só setor)": pg_setor}
        for nome, res in modelos.items():
            rel, li, ls = rel_setor(res)
            lv = verdade.reindex(rel.index)
            err = np.log(rel) - np.log(lv)
            cob = np.nan
            if li is not None and res.ep is not None:      # ridge não tem erro-padrão
                m = rel.index != "servicos"
                cob = float(((li[m] <= lv[m]) & (lv[m] <= ls[m])).mean())
            linhas.append({"rep": rep, "modelo": nome, "rmse_log_setor": float(np.sqrt((err ** 2).mean())),
                           "cobertura_ic95_setor": cob,
                           "deviance_fora_amostra": float(cv.set_index("modelo").loc[nome, "deviance_fora_amostra"]),
                           "k": res.extra.get("k", np.nan),
                           "k_ic_cobre_verdade": (res.extra["ic_k"][0] <= k_verdade <= res.extra["ic_k"][1])
                           if "ic_k" in res.extra else np.nan,
                           "lambda": lam if nome == "Ridge" else np.nan, "obitos": info["obitos"]})
        linhas.append({"rep": rep, "modelo": "BR-EMS pura", "rmse_log_setor": float(np.sqrt((np.log(verdade) ** 2).mean())),
                       "deviance_fora_amostra": float(cv.set_index("modelo").loc["BR-EMS pura", "deviance_fora_amostra"]),
                       "obitos": info["obitos"]})
        pd.DataFrame(linhas).to_csv(partes / f"rep_{rep:02d}.csv", index=False)
        print(f"rep {rep + 1}/{R}: {time.time() - t:.0f}s, óbitos={info['obitos']}, k_PG={pg.extra['k']:.1f}, lambda={lam}", flush=True)
    arquivos = sorted(partes.glob("rep_*.csv"))
    out = pd.concat([pd.read_csv(a) for a in arquivos], ignore_index=True)
    out["k_verdade"] = k_verdade
    destino = RAIZ / "resultados" / "estudo_simulacao_manual.csv"
    out.to_csv(destino, index=False)
    print("salvo em", destino)
    resumo = out.groupby("modelo").agg(rmse_log_setor=("rmse_log_setor", "mean"),
                                       cobertura_ic95=("cobertura_ic95_setor", "mean"),
                                       deviance_media=("deviance_fora_amostra", "mean"),
                                       k_medio=("k", "mean"), k_cobertura=("k_ic_cobre_verdade", "mean"))
    print(resumo.round(4).to_string()); print("k verdade (lognormal sigma=%.2f): %.2f" % (cs0["sigma_efeito_grupo"], k_verdade))


if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:]]
    main(*a) if a else main()
