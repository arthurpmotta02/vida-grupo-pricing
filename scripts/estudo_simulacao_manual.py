"""Estudo de simulação da taxa manual: a complexidade muda o preço?

Gera R carteiras sintéticas independentes (mesmas hipóteses, sementes diferentes) e compara:
- erro dos fatores de setor contra a verdade do gerador (RMSE em log);
- cobertura dos intervalos de 95% dos fatores de setor (GLM ignora o agrupamento por estipulante);
- erro dos fatores de faixa de capital contra a verdade (gradiente de renda do gerador);
- subsídio cruzado: erro de estrutura do prêmio puro manual em R$, grupo a grupo;
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
warnings.filterwarnings("ignore", message=".*limite superior da grade.*")

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
from vgpricing import verdade as V  # noqa: E402


def rel_nivel(res, fator, base):
    f = res.fatores()
    f = f[f["fator"] == fator].set_index("nivel")
    if f.empty:                                   # fator fora do modelo: todos os níveis valem 1
        return None, None, None
    r = f["relatividade"] / f.loc[base, "relatividade"]
    ok = bool(f.loc[base, "base"]) and res.ep is not None
    return r, (f["ic_inf"] if ok else None), (f["ic_sup"] if ok else None)


def main(R=12, inicio=0, fim=None):
    base = carregar_config("base", RAIZ); cs0 = carregar_config("sintetico", RAIZ); cm = carregar_config("manual", RAIZ)
    tab = obter_tabua(base, RAIZ)
    F = cm["fatores"]
    ver_setor = pd.Series({s: v["fator_mortalidade"] for s, v in cs0["setores"].items()})
    ver_setor = ver_setor / ver_setor["servicos"]
    k_verdade = 1 / (np.exp(cs0["sigma_efeito_grupo"] ** 2) - 1)
    lam_grade = cm["encolhimento"]["lambdas"]
    obrig = tuple(cm["fatores_obrigatorios"])
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
        det = V.detalhe_com_verdade(r.dados["vidas"], r.dados["estipulantes"], tab, base["data_referencia"], cm,
                                    cart["verdade_vidas"], cs)
        base_cap = M.Desenho.criar(cel, ["faixa_capital"]).base["faixa_capital"]
        ver_cap = V.relatividades_capital(det, base_cap)
        custo_v = V.custo_estrutural_por_grupo(det)

        sel, _ = M.selecionar_fatores(cel, F)
        sel_obr, _ = M.selecionar_fatores(cel, F, obrigatorios=obrig)
        lam_obr, _ = M.calibrar_encolhimento(cel, sel_obr, list(obrig), lam_grade, cm["encolhimento"]["folds"],
                                             cm["semente_cv"])
        pen_obr = {f: v for f, v in lam_obr.items() if v > 0} or None
        cvl = M.escolher_lambda(cel, F, cm["ridge"]["lambdas"], cm["ridge"]["folds"], cm["semente_cv"])
        lam = float(cvl.loc[cvl["deviance_fora_amostra"].idxmin(), "lambda"])

        aj = {
            "GLM (todos)": lambda d: M.ajustar_glm(d, F),
            "NB2 (todos)": lambda d: M.ajustar_nb2(d, F),
            "Ridge (todos)": lambda d: M.ajustar_ridge(d, F, lam),
            "Poisson-Gama (todos)": lambda d: M.ajustar_poisson_gama(d, F),
            "Poisson-Gama (selecionados)": lambda d: M.ajustar_poisson_gama(d, sel),
            "Poisson-Gama (setor + capital obrigatórios, encolhidos)": lambda d: M.ajustar_poisson_gama(d, sel_obr, penalizacao=pen_obr),
        }
        cv = M.validacao_cruzada(cel, {"BR-EMS pura": None, **aj}, cm["ridge"]["folds"], cm["semente_cv"]).set_index("modelo")
        for nome, ajustar in aj.items():
            res = ajustar(cel)
            rs, li, ls = rel_nivel(res, "setor", "servicos")
            if rs is None:                                    # setor fora do modelo
                rs = pd.Series(1.0, index=ver_setor.index)
            lv = ver_setor.reindex(rs.index)
            cob = np.nan
            if li is not None:
                m = rs.index != "servicos"
                cob = float(((li[m] <= lv[m]) & (lv[m] <= ls[m])).mean())
            rc, _, _ = rel_nivel(res, "faixa_capital", base_cap)
            rc = pd.Series(1.0, index=ver_cap.index) if rc is None else rc.reindex(ver_cap.index)
            linhas.append({"rep": rep, "modelo": nome,
                           "rmse_log_setor": float(np.sqrt(((np.log(rs) - np.log(lv)) ** 2).mean())),
                           "cobertura_ic95_setor": cob,
                           "rmse_log_capital": float(np.sqrt(((np.log(rc) - np.log(ver_cap)) ** 2).mean())),
                           "subsidio_cruzado": M.subsidio_cruzado(cel, res, custo_v),
                           "deviance_fora_amostra": float(cv.loc[nome, "deviance_fora_amostra"]),
                           "k": res.extra.get("k", np.nan),
                           "k_ic_cobre_verdade": (res.extra["ic_k"][0] <= k_verdade <= res.extra["ic_k"][1])
                           if "ic_k" in res.extra else np.nan,
                           "lambda_setor": lam_obr.get("setor", np.nan) if "obrigatórios" in nome else np.nan,
                           "lambda_capital": lam_obr.get("faixa_capital", np.nan) if "obrigatórios" in nome else np.nan,
                           "fatores": "+".join(res.desenho.fatores), "obitos": info["obitos"]})
        tabua_pura = M.ResultadoModelo("BR-EMS pura", M.Desenho.criar(cel, []), np.array([0.0]))
        linhas.append({"rep": rep, "modelo": "BR-EMS pura",
                       "rmse_log_setor": float(np.sqrt((np.log(ver_setor) ** 2).mean())),
                       "rmse_log_capital": float(np.sqrt((np.log(ver_cap) ** 2).mean())),
                       "subsidio_cruzado": M.subsidio_cruzado(cel, tabua_pura, custo_v),
                       "deviance_fora_amostra": float(cv.loc["BR-EMS pura", "deviance_fora_amostra"]),
                       "obitos": info["obitos"]})
        pd.DataFrame(linhas).to_csv(partes / f"rep_{rep:02d}.csv", index=False)
        print(f"rep {rep + 1}/{R}: {time.time() - t:.0f}s, óbitos={info['obitos']}, selecionados={sel}, "
              f"lambdas={lam_obr}", flush=True)
    arquivos = sorted(partes.glob("rep_*.csv"))
    out = pd.concat([pd.read_csv(a) for a in arquivos], ignore_index=True)
    out["k_verdade"] = k_verdade
    destino = RAIZ / "resultados" / "estudo_simulacao_manual.csv"
    out.to_csv(destino, index=False)
    print("salvo em", destino, "| réplicas:", out["rep"].nunique())
    resumo = out.groupby("modelo").agg(rmse_log_setor=("rmse_log_setor", "mean"),
                                       cobertura_ic95=("cobertura_ic95_setor", "mean"),
                                       rmse_log_capital=("rmse_log_capital", "mean"),
                                       subsidio_cruzado=("subsidio_cruzado", "mean"),
                                       deviance_media=("deviance_fora_amostra", "mean"),
                                       k_medio=("k", "mean"))
    print(resumo.sort_values("deviance_media").round(4).to_string())


if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:]]
    main(*a) if a else main()
