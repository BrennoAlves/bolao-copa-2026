"""
Recalibrar o volume de gols (lambda) valeria pontos?

O diagnóstico achou um viés leve: lambda previsto um pouco abaixo dos gols reais, e
quase toda aposta com um lado zerado. Mas +0.21 gol/jogo em ~37 jogos fica dentro
do ruído. Aqui resolvo com número: replay sobre os jogos já encerrados,
re-apostando o pick max-EV com o lambda_total multiplicado por um fator e pontuando
contra o placar real. Probabilidades e placar vêm do registro; só o lambda muda. Só
recomenda recalibrar se o ganho passar do ruído pareado.

Uso:
    uv run python -m pesquisa.estudos.calibrar_lambda
    uv run python -m pesquisa.estudos.calibrar_lambda --fatores 1.0,1.1,1.2,1.3
"""
from __future__ import annotations

import argparse
import statistics
import sys

from loguru import logger

from bolao.backtest import listar_registros_backtest
from bolao.modelo import eh_mata_mata, predizer
from bolao.resultados import calcular_pontos


def _pontos_sob_fator(reg, fator: float) -> tuple[int, bool, str]:
    """Re-aposta o pick max-EV com lambda_total escalado e pontua vs o placar real.

    Retorna (pontos, é_cravada, label_apostado). Usa as probabilidades do modelo
    (pós-Polymarket) do registro; só o lambda_total muda.
    """
    lambda_base = reg.lambda_casa + reg.lambda_fora
    pred = predizer(
        time_casa=reg.time_casa,
        time_fora=reg.time_fora,
        prob_casa=reg.prob_casa_modelo,
        prob_empate=reg.prob_empate_modelo,
        prob_fora=reg.prob_fora_modelo,
        lambda_total=lambda_base * fator,
        mata_mata=eh_mata_mata(reg.kickoff_utc),
    )
    pick = pred.melhor_placar
    assert reg.gols_reais_casa is not None and reg.gols_reais_fora is not None
    pts, _tipo = calcular_pontos(
        pick.gols_casa, pick.gols_fora, reg.gols_reais_casa, reg.gols_reais_fora
    )
    return pts, (pts == 3), pick.label


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep de lambda sobre o backtest")
    parser.add_argument("--fatores", type=str, default="1.00,1.05,1.10,1.15",
                        help="fatores de lambda_total separados por vírgula")
    args = parser.parse_args()
    fatores = [float(x) for x in args.fatores.split(",") if x.strip()]
    if 1.0 not in fatores:
        fatores = [1.0, *fatores]  # 1.0 é a baseline da comparação pareada

    logger.disable("bolao.modelo")
    regs = [r for r in listar_registros_backtest() if r.completo]
    if not regs:
        print("Nenhum registro de backtest encerrado no cache.")
        sys.exit(1)
    n = len(regs)

    # Pontos por jogo sob cada fator (guardados p/ erro-padrão e diff pareada)
    pts_por_fator: dict[float, list[int]] = {}
    crav_por_fator: dict[float, int] = {}
    match_baseline = 0  # f=1.0 reproduz o pick realmente apostado?
    for f in fatores:
        col = []
        crav = 0
        for r in regs:
            pts, is_crav, label = _pontos_sob_fator(r, f)
            col.append(pts)
            crav += int(is_crav)
            if f == 1.0 and label == r.label_apostado:
                match_baseline += 1
        pts_por_fator[f] = col
        crav_por_fator[f] = crav

    base = pts_por_fator[1.0]
    base_total = sum(base)
    real_total = sum(r.pontos_ganhos or 0 for r in regs)

    print("\n" + "=" * 70)
    print(f"  Calibração de lambda: replay sobre {n} jogo(s) encerrado(s) (sem vazamento)")
    print(f"  reconstrução f=1.0: {base_total} pts | apostas reais logadas: {real_total} pts "
          f"| pick idêntico em {match_baseline}/{n}")
    print("-" * 70)
    print(f"  {'fator':>8}{'pontos':>8}{'cravadas':>10}{'E[pts]/jogo':>13}{'delta vs 1.0 (pareado)':>24}")
    for f in fatores:
        col = pts_por_fator[f]
        total = sum(col)
        media = total / n
        if f == 1.0:
            delta_txt = "-  (baseline)"
        else:
            diffs = [a - b for a, b in zip(col, base, strict=True)]
            md = statistics.mean(diffs)
            sd = statistics.pstdev(diffs) if n > 1 else 0.0
            se = sd / (n ** 0.5) if n > 0 else 0.0
            z = md / se if se > 0 else 0.0
            sig = "*" if abs(z) >= 1.96 else ""
            delta_txt = f"{md * n:+.0f} pts ({md:+.3f}/jg, z={z:+.2f}){sig}"
        print(f"  {f:>8.2f}{total:>8}{crav_por_fator[f]:>10}{media:>13.3f}{delta_txt:>22}")
    print("-" * 70)

    # veredito
    melhor_f = max(fatores, key=lambda f: sum(pts_por_fator[f]))
    if melhor_f == 1.0:
        veredito = "nenhum fator supera 1.0: NÃO recalibrar, o viés é ruído neste recorte."
    else:
        col = pts_por_fator[melhor_f]
        diffs = [a - b for a, b in zip(col, base, strict=True)]
        md = statistics.mean(diffs)
        se = (statistics.pstdev(diffs) / (n ** 0.5)) if n > 1 else 0.0
        z = md / se if se > 0 else 0.0
        if abs(z) >= 1.96:
            veredito = (f"lambda x {melhor_f:.2f} rende {md * n:+.0f} pts com abs(z)={abs(z):.2f} >= 1.96: "
                        "ganho acima do ruído; vale aplicar o fator no fit do lambda.")
        else:
            veredito = (f"melhor fator lambda x {melhor_f:.2f} ({md * n:+.0f} pts) mas abs(z)={abs(z):.2f} < 1.96: "
                        "dentro do ruído; não recalibrar com este n (reavaliar com mais jogos).")
    print(f"  -> {veredito}")
    print("  (lembrando: é um torneio só, olhando pra trás, não vale como expectativa.)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
