"""
O modelo de placar bate só chutar números?

Na Copa 2022 (64 jogos, via 538) comparo o modelo de produção contra baselines
bobos que usam só a direção do mercado e a frequência histórica de placares. Se o
modelo não render mais pontos, toda a maquinaria probabilística não está valendo
de nada. Todas as opções recebem o mesmo 1X2 (538) e o mesmo lambda; só muda a
regra de escolher o placar, pra isolar o que o modelo de placar agrega. A
frequência de placares vem de ~49k jogos internacionais, tirando 2022+ pra não
vazar a própria Copa.

Uso: uv run python -m pesquisa.estudos.teste_calibracao
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np

from bolao.backtest import calcular_rps
from pesquisa.experimento.comparacao import placar_previsto, pontos_bolao
from pesquisa.experimento.mercado import carregar_mercado_538

_RESULTS = Path("pesquisa/dados/international_results.csv")


def freq_placares() -> tuple[Counter, Counter, Counter, Counter]:
    """
    Frequência histórica de placares (pré-2022, internacionais), na perspectiva
    vencedor x perdedor: geral, vitórias (w,l), empates (k,k) e o placar modal.
    """
    geral: Counter = Counter()      # (vencedor, perdedor) e (k,k)
    vitorias: Counter = Counter()   # (gols_vencedor, gols_perdedor)
    empates: Counter = Counter()    # k de k x k
    casa_fora: Counter = Counter()  # (gols_casa, gols_fora) cru
    with open(_RESULTS, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["date"][:4] >= "2022":
                continue
            try:
                gc, gf = int(r["home_score"]), int(r["away_score"])
            except (ValueError, KeyError):
                continue
            casa_fora[(gc, gf)] += 1
            if gc == gf:
                empates[gc] += 1
                geral[(gc, gc)] += 1
            else:
                venc, perd = max(gc, gf), min(gc, gf)
                vitorias[(venc, perd)] += 1
                geral[(venc, perd)] += 1
    return geral, vitorias, empates, casa_fora


def main() -> None:
    geral, vitorias, empates, casa_fora = freq_placares()
    placar_modal = casa_fora.most_common(1)[0][0]          # placar cru mais comum
    vit_modal = vitorias.most_common(1)[0][0]              # (w,l) vitória mais comum
    emp_modal = empates.most_common(1)[0][0]               # k do empate mais comum

    n_total = sum(casa_fora.values())
    print(f"Frequência histórica de placares (pré-2022, {n_total} jogos):")
    for s, c in casa_fora.most_common(6):
        print(f"   {s[0]}x{s[1]}: {c/n_total:5.1%}")
    print(f"   -> vitória modal (venc x perd): {vit_modal[0]}x{vit_modal[1]} | "
          f"empate modal: {emp_modal}x{emp_modal}\n")

    jogos = carregar_mercado_538()

    # Distribuição empírica oriented-to-favorite para o chutador (E[pts] analítico)
    tot_v = sum(vitorias.values())
    tot_e = sum(empates.values())
    p_emp_hist = tot_e / (tot_v + tot_e)

    def orientar(jg, venc_perd, k_empate, classe):
        """Converte (venc,perd)/empate para placar (time1,time2) dado o favorito."""
        fav1 = jg.prob1 >= jg.prob2
        if classe == "E":
            return (k_empate, k_empate)
        venc, perd = venc_perd
        # vencedor = favorito (time1 se fav1, senão time2)
        return (venc, perd) if fav1 else (perd, venc)

    # cada estratégia é uma função jogo -> placar (g1,g2)
    def s_modelo(jg):
        return placar_previsto((jg.prob1, jg.prob_empate, jg.prob2), jg.lambda_total)

    def s_fav_1x0(jg):
        return (1, 0) if jg.prob1 >= jg.prob2 else (0, 1)

    def s_modal_global(jg):
        # sempre o placar cru mais comum, orientado ao favorito
        return orientar(jg, vit_modal, emp_modal, "C") if placar_modal[0] != placar_modal[1] \
            else (emp_modal, emp_modal)

    def s_resultado_empirico(jg):
        # direção do mercado -> placar modal daquela classe
        classe = "C" if (jg.prob1 >= jg.prob_empate and jg.prob1 >= jg.prob2) else \
                 "F" if (jg.prob2 >= jg.prob_empate and jg.prob2 >= jg.prob1) else "E"
        return orientar(jg, vit_modal, emp_modal, classe)

    def s_sempre_1x1(jg):
        return (1, 1)

    estrategias = {
        "MODELO (produção)": s_modelo,
        "Favorito 1x0": s_fav_1x0,
        "Placar modal global": s_modal_global,
        "Direção mercado + placar empírico": s_resultado_empirico,
        "Sempre 1x1": s_sempre_1x1,
    }

    reais = [(j.gols1, j.gols2) for j in jogos]
    n = len(jogos)

    # Pontos jogo-a-jogo de cada estratégia
    pts: dict[str, list[int]] = {}
    crav: dict[str, int] = {}
    res: dict[str, int] = {}
    for nome, fn in estrategias.items():
        pj = [pontos_bolao(fn(j), real) for j, real in zip(jogos, reais, strict=True)]
        pts[nome] = pj
        crav[nome] = sum(1 for x in pj if x == 3)
        res[nome] = sum(1 for x in pj if x == 1)

    # Chutador "esperto": E[pts] sob freq. empírica orientada ao favorito.
    # Distribuição: P(empate)=p_emp_hist repartido por empates; resto por vitórias
    # (vencedor = favorito). Calcula E[pts] analítico por jogo.
    def e_pts_chutador(jg, real):
        e = 0.0
        for k, c in empates.items():
            p = p_emp_hist * c / tot_e
            e += p * pontos_bolao((k, k), real)
        for (venc, perd), c in vitorias.items():
            p = (1 - p_emp_hist) * c / tot_v
            placar = (venc, perd) if jg.prob1 >= jg.prob2 else (perd, venc)
            e += p * pontos_bolao(placar, real)
        return e
    e_chut = [e_pts_chutador(j, real) for j, real in zip(jogos, reais, strict=True)]

    # RPS do mercado (referência de calibração do 1X2)
    gols_res = {"C": (1, 0), "E": (0, 0), "F": (0, 1)}
    rps_mkt = np.mean([
        calcular_rps(j.prob1, j.prob_empate, j.prob2, *gols_res[j.resultado])
        for j in jogos
    ])

    # relatório
    print(f"{'-'*70}")
    print(f"  Teste de placar: Copa 2022 ({n} jogos, sem vazamento)")
    print(f"{'-'*70}")
    print(f"  {'Estratégia':<36}{'pts':>5}{'/jogo':>7}{'crav':>6}{'res':>5}")
    print(f"  {'-'*36}{'-'*5}{'-'*7}{'-'*6}{'-'*5}")
    ordenado = sorted(pts, key=lambda k: -sum(pts[k]))
    for nome in ordenado:
        t = sum(pts[nome])
        print(f"  {nome:<36}{t:>5}{t/n:>7.2f}{crav[nome]:>6}{res[nome]:>5}")
    print(f"  {'Chutador (E[pts] freq. empírica)':<36}{sum(e_chut):>5.0f}{sum(e_chut)/n:>7.2f}{'-':>6}{'-':>5}")
    print(f"  {'-'*70}")
    print(f"  RPS do mercado (1X2): {rps_mkt:.4f}   (piso do palpite uniforme: 0.222)")
    print(f"{'-'*70}\n")

    # bootstrap pareado: MODELO vs melhor baseline
    baselines = {k: v for k, v in pts.items() if k != "MODELO (produção)"}
    melhor_base = max(baselines, key=lambda k: sum(baselines[k]))
    a = np.array(pts["MODELO (produção)"], dtype=float)
    b = np.array(baselines[melhor_base], dtype=float)
    diff = a - b  # positivo = modelo melhor
    rng = np.random.default_rng(42)
    medias = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(10_000)])
    lo, hi = np.percentile(medias, [2.5, 97.5])

    print(f"  MODELO vs melhor baseline ('{melhor_base}'):")
    print(f"    delta pontos/jogo (modelo - baseline): {diff.mean():+.3f}  "
          f"IC95% [{lo:+.3f}, {hi:+.3f}]")
    print(f"    P(modelo rende mais) = {(medias > 0).mean():.1%}")
    if lo > 0:
        print("    -> o modelo agrega sobre chutar o placar óbvio.")
    elif hi < 0:
        print("    -> o modelo é pior que chutar o placar óbvio.")
    else:
        print("    -> diferença não é significativa (IC cruza zero): "
              "o modelo não bate chutar o placar óbvio.")
    print()


if __name__ == "__main__":
    main()
