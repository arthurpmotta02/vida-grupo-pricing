"""Gerador de carteira sintética de Vida em Grupo com verdade conhecida.

O que é simulado (todos os parâmetros em config/sintetico.yaml, rotulados como hipóteses):
- estipulantes de 30 a 5.000 vidas, 3 a 5 anos de histórico, setor e efeito de grupo;
- vidas com entradas, saídas (rotatividade), envelhecimento e mudança de capital;
- óbitos com hazard = q_x(tábua) × fator do setor × fator do grupo;
- eventos catastróficos (várias mortes na mesma data em um grupo);
- atraso de aviso (gera IBNR na data de referência), sinistros negados e pendentes;
- prêmio histórico com taxa manual e reajuste anual pela regra contratual por faixa;
- sujeira de cadastro injetada com registro exato do que foi sujo (para testar a validação).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .exposicao import idade_completa
from .tabua import Tabua

DIA_ANO = 365.25


def _idade_np(nasc: np.ndarray, data: np.ndarray) -> np.ndarray:
    return idade_completa(pd.Series(nasc), pd.Series(data)).to_numpy(dtype=int)


def _reajuste_faixa(sinistralidade: float, faixas: list[dict]) -> float:
    for f in faixas:
        if sinistralidade <= float(f["ate"]):
            return float(f["reajuste"])
    return float(faixas[-1]["reajuste"])


class _Grupo:
    """Estado de um estipulante durante a simulação."""

    def __init__(self, rng, cfg, setor, tipo_cap):
        self.rng, self.cfg, self.setor, self.tipo_cap = rng, cfg, setor, tipo_cap
        c = cfg["capital"]
        self.valor_uniforme = rng.choice(c["uniforme_valores"])
        self.multiplo = rng.choice(c["multiplos"])
        self.prox_id = 0

    def novas(self, k: int, data_entrada: np.ndarray) -> dict:
        rng, cfg, s = self.rng, self.cfg, self.cfg["setores"][self.setor]
        ci = cfg["idade"]
        idade = np.clip(rng.normal(s["idade_media"], ci["sd"], k), ci["min_entrada"], ci["max_entrada"])
        nasc = data_entrada - (idade * DIA_ANO).astype("timedelta64[D]")
        sexo = np.where(rng.random(k) < s["prop_masc"], "M", "F")
        c = cfg["capital"]
        salario = c["salario_mediana"] * np.exp(rng.normal(0, c["salario_sigma"], k))
        nivel = rng.choice(len(c["escalonado_niveis"]), size=k, p=c["escalonado_probs"])
        ids = np.arange(self.prox_id, self.prox_id + k)
        self.prox_id += k
        e = cfg.get("gradiente_renda", {}).get("elasticidade", 0.0)
        mult_renda = (salario / c["salario_mediana"]) ** (-e) / np.exp((e * c["salario_sigma"]) ** 2 / 2)
        return {"vid": ids, "nasc": nasc.astype("datetime64[D]"), "sexo": sexo,
                "salario": salario, "nivel": nivel, "mult_renda": mult_renda}

    def capital(self, salario, nivel):
        c = self.cfg["capital"]
        if self.tipo_cap == "uniforme":
            cap = np.full(len(salario), float(self.valor_uniforme))
        elif self.tipo_cap == "multiplo_salarial":
            cap = np.round(salario * self.multiplo, -2)
        else:
            cap = np.asarray(c["escalonado_niveis"], dtype=float)[nivel]
        return np.minimum(cap, c["teto"])


def _concat(a: dict, b: dict) -> dict:
    return {k: np.concatenate([a[k], b[k]]) for k in a}


def _filtrar(a: dict, m: np.ndarray) -> dict:
    return {k: v[m] for k, v in a.items()}


def gerar_carteira(cfg: dict, tabua: Tabua) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(cfg["semente"])
    data_ref = np.datetime64(cfg["data_referencia"], "D")
    ref_ts = pd.Timestamp(cfg["data_referencia"])
    setores = list(cfg["setores"])
    pesos = np.array([cfg["setores"][s]["peso"] for s in setores], dtype=float)
    pesos /= pesos.sum()
    tipos = list(cfg["capital"]["tipos"])
    p_tipos = np.array([cfg["capital"]["tipos"][t] for t in tipos], dtype=float)
    p_tipos /= p_tipos.sum()
    sig = cfg["sigma_efeito_grupo"]
    lam = -np.log(1 - cfg["rotatividade_anual"])
    cs, cp = cfg["sinistros"], cfg["premio"]

    estip, verdade, pecas, sinis, premios = [], [], [], [], []
    n_sin = 0
    for e in range(cfg["n_estipulantes"]):
        eid = f"EST{e:04d}"
        setor = setores[rng.choice(len(setores), p=pesos)]
        tipo = tipos[rng.choice(len(tipos), p=p_tipos)]
        n_anos = int(rng.integers(cfg["anos_historico"][0], cfg["anos_historico"][1] + 1))
        mes = int(rng.integers(1, 13))
        inicio = pd.Timestamp(ref_ts.year - n_anos, mes, 1)
        tamanho = int(round(np.exp(rng.uniform(np.log(cfg["tamanho"]["min"]), np.log(cfg["tamanho"]["max"])))))
        f_grupo = float(np.exp(rng.normal(-sig ** 2 / 2, sig)))
        f_setor = float(cfg["setores"][setor]["fator_mortalidade"])
        fator = f_setor * f_grupo
        g = _Grupo(rng, cfg, setor, tipo)

        ativos = g.novas(tamanho, np.full(tamanho, np.datetime64(inicio.date(), "D")))
        ativos["capital"] = g.capital(ativos["salario"], ativos["nivel"])
        fator_reaj, taxa_mensal_final = 1.0, np.nan
        sin_ano_anterior = None
        k = 0
        while True:
            ps = inicio + pd.DateOffset(years=k)
            if ps > ref_ts:
                break
            pe_cheio = inicio + pd.DateOffset(years=k + 1) - pd.Timedelta(days=1)
            pe = min(pe_cheio, ref_ts)
            ps64, pe64 = np.datetime64(ps.date(), "D"), np.datetime64(pe.date(), "D")
            L = int((pe64 - ps64).astype(int)) + 1

            if k > 0:  # aniversário: reajuste salarial e reajuste contratual da taxa
                ativos["salario"] = ativos["salario"] * (1 + cfg["capital"]["reajuste_salarial_anual"])
                ativos["capital"] = g.capital(ativos["salario"], ativos["nivel"])
                prem_ant, ocorr_ant = sin_ano_anterior
                avisados = sum(v for (a, v) in ocorr_ant if a <= ps64)
                sinistr = (avisados + cp["ibnr_regra_pct_premio"] * prem_ant) / prem_ant if prem_ant > 0 else 0.0
                fator_reaj *= 1 + _reajuste_faixa(sinistr, cp["faixas_reajuste"])

            # taxa manual sobre o censo no aniversário (tábua pura, sem setor nem grupo)
            idade_ps = _idade_np(ativos["nasc"], np.full(len(ativos["vid"]), ps64))
            q_ps = tabua.q(idade_ps, ativos["sexo"])
            taxa_pura_mil = 1000 * np.sum(q_ps * ativos["capital"]) / np.sum(ativos["capital"])
            taxa_anual_mil = taxa_pura_mil * fator_reaj / (1 - cp["carregamento_total"])

            # entradas no ano
            n_alvo = len(ativos["vid"]) * np.exp(rng.normal(0, cfg["crescimento_anual_sd"])) if k > 0 else len(ativos["vid"])
            n_ent = rng.poisson(n_alvo * lam * L / DIA_ANO) + max(0, int(round(n_alvo - len(ativos["vid"]))))
            off_ent = rng.integers(0, L, n_ent)
            ent = g.novas(n_ent, ps64 + off_ent.astype("timedelta64[D]"))
            ent["capital"] = g.capital(ent["salario"], ent["nivel"])
            todos = _concat(ativos, ent)
            offset = np.concatenate([np.zeros(len(ativos["vid"]), int), off_ent])
            n = len(offset)
            ini = ps64 + offset.astype("timedelta64[D]")
            idade = _idade_np(todos["nasc"], ini)
            q = np.minimum(tabua.q(idade, todos["sexo"]) * fator * todos["mult_renda"], 0.999)
            mu = -np.log(1 - q)
            td = rng.exponential(1 / mu) * DIA_ANO
            te = rng.exponential(1 / lam, n) * DIA_ANO
            R = L - offset
            if n >= 10 and rng.random() < cs["prob_evento_catastrofico"] * L / DIA_ANO:
                dia = int(rng.integers(0, L))
                elegiveis = np.flatnonzero(offset <= dia)
                kk = min(len(elegiveis), int(rng.integers(cs["mortes_evento_catastrofico"][0],
                                                          cs["mortes_evento_catastrofico"][1] + 1)))
                esc = rng.choice(elegiveis, kk, replace=False)
                td[esc] = np.minimum(td[esc], dia - offset[esc] + 0.5)
            morreu = td < np.minimum(te, R)
            saiu = (te < np.minimum(td, R)) & ~morreu
            fica = ~morreu & ~saiu
            t_fim = np.where(morreu, td, np.where(saiu, te, R - 1))
            fim = ini + np.floor(np.minimum(t_fim, R - 1)).astype(int).astype("timedelta64[D]")
            pecas.append(pd.DataFrame({
                "id_estipulante": eid, "vid": todos["vid"], "data_nascimento": todos["nasc"],
                "sexo": todos["sexo"], "data_inicio": ini, "data_fim": fim, "capital": todos["capital"],
                "em_vigor": fica & (pe64 == data_ref), "mult_renda": todos["mult_renda"]}))

            # prêmio do ano: taxa x capital-ano
            cap_ano = np.sum(todos["capital"] * ((fim - ini).astype(int) + 1) / DIA_ANO)
            premio = taxa_anual_mil / 1000 * cap_ano
            premios.append((eid, ps, pe, premio, taxa_anual_mil / 12))
            taxa_mensal_final = taxa_anual_mil / 12

            # sinistros
            ocorr_ano = []
            for i in np.flatnonzero(morreu):
                atraso = int(round(np.exp(rng.normal(np.log(cs["atraso_aviso_mediana_dias"]), cs["atraso_aviso_sigma"]))))
                aviso = fim[i] + np.timedelta64(atraso, "D")
                negado = rng.random() < cs["prob_negado"]
                sinis.append({"id_sinistro": f"SIN{n_sin:07d}", "id_estipulante": eid, "vid": todos["vid"][i],
                              "cobertura": "MQC", "data_ocorrencia": fim[i], "data_aviso": aviso,
                              "capital": todos["capital"][i], "negado": negado,
                              "dias_pag": int(rng.integers(5, 31))})
                if not negado:
                    ocorr_ano.append((aviso, todos["capital"][i]))
                n_sin += 1
            sin_ano_anterior = (premio, ocorr_ano)
            ativos = _filtrar(todos, fica)
            k += 1

        estip.append({"id_estipulante": eid, "setor": setor, "data_inicio_vigencia": inicio,
                      "data_fim_vigencia": pd.NaT, "tipo_capital": tipo,
                      "contributario": bool(rng.random() < 0.3), "taxa_vigente_mensal_mil": taxa_mensal_final})
        verdade.append({"id_estipulante": eid, "setor": setor, "fator_setor": f_setor, "fator_grupo": f_grupo,
                        "fator_verdadeiro": fator, "tamanho_inicial": tamanho, "anos": n_anos, "tipo_capital": tipo})

    todas_pecas = pd.concat(pecas, ignore_index=True)
    verdade_vidas = todas_pecas.groupby(["id_estipulante", "vid"], as_index=False)["mult_renda"].first()
    verdade_vidas["id_vida"] = verdade_vidas["id_estipulante"] + "-V" + verdade_vidas.pop("vid").astype(str).str.zfill(6)
    vidas = _montar_segmentos(todas_pecas.drop(columns="mult_renda"))
    # afastados: escolhidos entre vidas em vigor sem sinistro
    em_vigor = vidas.loc[vidas["data_fim"].isna(), ["id_estipulante", "id_vida"]].drop_duplicates()
    n_af = int(round(cfg["proporcao_afastados"] * len(em_vigor)))
    af = em_vigor.sample(n=n_af, random_state=int(rng.integers(1 << 31)))
    chave_af = set(zip(af["id_estipulante"], af["id_vida"]))
    vidas["situacao"] = [("afastado" if (e, v) in chave_af else "ativo")
                         for e, v in zip(vidas["id_estipulante"], vidas["id_vida"])]
    vidas.insert(0, "id_linha", np.arange(1, len(vidas) + 1))

    s = pd.DataFrame(sinis)
    s["id_vida"] = s["id_estipulante"] + "-V" + s.pop("vid").astype(str).str.zfill(6)
    for c in ("data_ocorrencia", "data_aviso"):
        s[c] = pd.to_datetime(s[c])
    ibnr = s["data_aviso"] > ref_ts
    verdade_ibnr = s[ibnr].copy()
    s = s[~ibnr].copy()
    pago = s["data_aviso"] + pd.to_timedelta(s.pop("dias_pag"), unit="D")
    neg = s.pop("negado")
    s["status"] = np.where(neg, "negado", np.where(pago <= ref_ts, "pago", "pendente"))
    s["data_pagamento"] = pago.where(s["status"] == "pago")
    cap = s.pop("capital")
    s["valor_indenizado"] = np.where(s["status"] == "pago", cap, 0.0)
    s["valor_reserva"] = np.where(s["status"] == "pendente", cap, 0.0)
    s = s.reset_index(drop=True)
    s.insert(0, "id_linha", np.arange(1, len(s) + 1))
    verdade_ibnr["id_vida"] = verdade_ibnr["id_vida"]

    premios_df = pd.DataFrame(premios, columns=["id_estipulante", "inicio_periodo", "fim_periodo",
                                                "premio_emitido", "taxa_mensal_mil"])
    premios_df.insert(0, "id_linha", np.arange(1, len(premios_df) + 1))
    est_df = pd.DataFrame(estip)
    est_df.insert(0, "id_linha", np.arange(1, len(est_df) + 1))
    return {"estipulantes": est_df, "vidas": vidas, "sinistros": s, "premios": premios_df,
            "verdade_estipulantes": pd.DataFrame(verdade),
            "verdade_vidas": verdade_vidas[["id_estipulante", "id_vida", "mult_renda"]],
            "verdade_ibnr": verdade_ibnr.drop(columns=["dias_pag", "negado"], errors="ignore").reset_index(drop=True)}


def _montar_segmentos(p: pd.DataFrame) -> pd.DataFrame:
    """Une pedaços contíguos da mesma vida com o mesmo capital."""
    p = p.sort_values(["id_estipulante", "vid", "data_inicio"]).reset_index(drop=True)
    mesmo = ((p["id_estipulante"] == p["id_estipulante"].shift()) & (p["vid"] == p["vid"].shift())
             & (p["capital"] == p["capital"].shift())
             & (p["data_inicio"] == p["data_fim"].shift() + np.timedelta64(1, "D")))
    p["seg"] = (~mesmo).cumsum()
    s = p.groupby("seg").agg(id_estipulante=("id_estipulante", "first"), vid=("vid", "first"),
                             data_nascimento=("data_nascimento", "first"), sexo=("sexo", "first"),
                             data_inicio=("data_inicio", "min"), data_fim=("data_fim", "max"),
                             capital=("capital", "first"), em_vigor=("em_vigor", "last"))
    s["data_fim"] = s["data_fim"].where(~s.pop("em_vigor"))
    s["id_vida"] = s["id_estipulante"] + "-V" + s.pop("vid").astype(str).str.zfill(6)
    for c in ("data_nascimento", "data_inicio", "data_fim"):
        s[c] = pd.to_datetime(s[c])
    return s.reset_index(drop=True)[["id_estipulante", "id_vida", "data_nascimento", "sexo",
                                     "data_inicio", "data_fim", "capital"]]


def injetar_sujeira(dados: dict[str, pd.DataFrame], cfg: dict) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Suja vidas e sinistros nas proporções do config. Devolve (dados_sujos, verdade_sujeira),
    onde verdade_sujeira lista exatamente qual registro recebeu qual problema e qual regra
    de validação deve capturá-lo."""
    rng = np.random.default_rng(cfg["semente"] + 1)
    sj = cfg["sujeira"]
    v = dados["vidas"].copy()
    s = dados["sinistros"].copy()
    verdade = []

    # --- vidas: conjuntos disjuntos de linhas (um problema por linha)
    v = v.reset_index(drop=True)
    livre = (v["situacao"] == "ativo").to_numpy().copy()
    ordem = rng.permutation(len(v))

    def pegar(frac, filtro=None):
        cand = ordem[livre[ordem]]
        if filtro is not None:
            cand = cand[filtro(v.iloc[cand]).to_numpy()]
        esc = cand[:int(round(frac * len(v)))]
        livre[esc] = False
        return esc

    idx = pegar(sj["nascimento_ausente"]);   v.loc[idx, "data_nascimento"] = pd.NaT
    verdade += [("vidas", "V03", v.at[i, "id_linha"]) for i in idx]
    idx = pegar(sj["idade_impossivel"]);     v.loc[idx, "data_nascimento"] = pd.Timestamp("1900-01-01")
    verdade += [("vidas", "V04", v.at[i, "id_linha"]) for i in idx]
    idx = pegar(sj["sexo_ausente"]);         v.loc[idx, "sexo"] = pd.NA
    verdade += [("vidas", "V05", v.at[i, "id_linha"]) for i in idx]
    idx = pegar(sj["capital_zero"]);         v.loc[idx, "capital"] = 0.0
    verdade += [("vidas", "V06", v.at[i, "id_linha"]) for i in idx]
    idx = pegar(sj["fim_antes_inicio"], lambda d: d["data_fim"].notna())
    v.loc[idx, "data_fim"] = v.loc[idx, "data_inicio"] - pd.Timedelta(days=1)
    verdade += [("vidas", "V07", v.at[i, "id_linha"]) for i in idx]
    idx = pegar(sj["vidas_duplicadas"])
    dup = v.loc[idx].copy()
    dup["id_linha"] = np.arange(v["id_linha"].max() + 1, v["id_linha"].max() + 1 + len(dup))
    verdade += [("vidas", "V01", i) for i in dup["id_linha"]]
    v = pd.concat([v, dup], ignore_index=True)

    # --- sinistros
    validos = s.index[s["status"] != "negado"].to_numpy()
    perm_s = rng.permutation(validos)
    k1 = int(round(sj["sinistro_vida_inexistente"] * len(s)))
    k2 = int(round(sj["aviso_antes_ocorrencia"] * len(s)))
    k3 = int(round(sj["sinistro_duplicado"] * len(s)))
    i1, i2, i3 = perm_s[:k1], perm_s[k1:k1 + k2], perm_s[k1 + k2:k1 + k2 + k3]
    s.loc[i1, "id_vida"] = s.loc[i1, "id_estipulante"] + "-X" + pd.Series(range(len(i1)), index=i1).astype(str)
    verdade += [("sinistros", "S02", s.at[i, "id_sinistro"]) for i in i1]
    s.loc[i2, "data_aviso"] = s.loc[i2, "data_ocorrencia"] - pd.to_timedelta(rng.integers(1, 31, len(i2)), unit="D")
    verdade += [("sinistros", "S04", s.at[i, "id_sinistro"]) for i in i2]
    dup_s = s.loc[i3].copy()
    dup_s["id_sinistro"] = [f"SIN-D{j:06d}" for j in range(len(dup_s))]
    dup_s["id_linha"] = np.arange(s["id_linha"].max() + 1, s["id_linha"].max() + 1 + len(dup_s))
    verdade += [("sinistros", "S01", x) for x in dup_s["id_sinistro"]]
    s = pd.concat([s, dup_s], ignore_index=True)

    out = dict(dados)
    out["vidas"], out["sinistros"] = v, s
    return out, pd.DataFrame(verdade, columns=["tabela", "codigo", "chave"])


def tabelas_de_entrada(dados: dict) -> dict:
    """Só as 4 tabelas que uma seguradora teria (sem as de verdade)."""
    return {k: dados[k] for k in ("estipulantes", "vidas", "sinistros", "premios")}
