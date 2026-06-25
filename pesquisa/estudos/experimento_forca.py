"""
Queria saber qual prior de força rende mais pontos no bolão, sem vazamento.

Na Copa 2022 (64 jogos) comparo, jogo a jogo: Mercado (forecast pré-jogo do
538/SPI), Elo (rating computado do histórico, na véspera) e a mistura dos dois,
varrendo o peso do Elo. Olho duas coisas: pontos do bolão (3 placar exato, 1
resultado), que é o que importa, via o `predizer` da produção com o lambda do 538;
e o RPS do 1X2 (menor é melhor), que não depende do lambda. A melhor opção vs
mercado sai com bootstrap pareado.

Uso:
    uv run python -m pesquisa.estudos.experimento_forca
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

from loguru import logger

from bolao.backtest import calcular_rps
from pesquisa.experimento.comparacao import (
    Prob1X2,
    blend,
    bootstrap_diff,
    casar,
    placar_previsto,
    pontos_bolao,
)
from pesquisa.experimento.dados import carregar_partidas
from pesquisa.experimento.elo import preparar_elo
from pesquisa.experimento.mercado import carregar_mercado_538

# Pesos do Elo na mistura a varrer (0 = só mercado, 1 = só Elo).
_PESOS_BLEND = (0.0, 0.15, 0.30, 0.50, 0.70, 1.0)

# calcular_rps recebe gols; traduzimos o resultado num placar mínimo equivalente.
_GOLS_DO_RESULTADO = {"C": (1, 0), "E": (0, 0), "F": (0, 1)}


@dataclass(frozen=True)
class Metricas:
    """Resultado agregado de uma opção sobre os jogos avaliados."""

    rps: float
    pts_total: int
    pts_por_jogo: list[int]  # ponto a ponto, para o bootstrap pareado
    cravadas: int
    resultados: int


def _avaliar(
    probs: list[Prob1X2], lambdas: list[float], resultados: list[str], reais: list[tuple[int, int]]
) -> Metricas:
    """Calcula RPS e pontos do bolão de uma opção sobre todos os jogos."""
    rps = sum(
        calcular_rps(pc, pe, pf, *_GOLS_DO_RESULTADO[r])
        for (pc, pe, pf), r in zip(probs, resultados, strict=True)
    ) / len(probs)

    pts_por_jogo = [
        pontos_bolao(placar_previsto(p, lam), real)
        for p, lam, real in zip(probs, lambdas, reais, strict=True)
    ]
    return Metricas(
        rps=rps,
        pts_total=sum(pts_por_jogo),
        pts_por_jogo=pts_por_jogo,
        cravadas=sum(1 for x in pts_por_jogo if x == 3),
        resultados=sum(1 for x in pts_por_jogo if x == 1),
    )


def comparar(ano: int = 2022) -> None:
    """Roda a comparação e imprime o relatório com veredito."""
    partidas = carregar_partidas()
    preparo = preparar_elo(partidas, ano)
    mercado = carregar_mercado_538()

    # predizer loga em INFO a cada jogo; silencia durante o lote pesado
    logger.disable("bolao.modelo")

    casados, faltantes = casar(mercado, preparo.snapshots, preparo.params_empate)
    if faltantes:
        print(f"  [aviso] {len(faltantes)} jogos do 538 sem par no Elo:")
        for j in faltantes:
            print(f"      {j.data} {j.time1} x {j.time2}")

    resultados = [c.jogo.resultado for c in casados]
    reais = [(c.jogo.gols1, c.jogo.gols2) for c in casados]
    lambdas = [c.jogo.lambda_total for c in casados]
    probs_mercado = [c.probs_mercado for c in casados]
    probs_elo = [c.probs_elo for c in casados]
    n = len(casados)

    # Monta as opções: mercado, misturas, Elo puro.
    opcoes: list[tuple[str, list[Prob1X2]]] = [("Mercado (538 SPI)", probs_mercado)]
    for w in _PESOS_BLEND:
        if w == 0.0:
            continue
        nome = "Elo puro" if w == 1.0 else f"Mistura (Elo {w:.0%})"
        opcoes.append((nome, [blend(m, e, w) for m, e in zip(probs_mercado, probs_elo, strict=True)]))

    metricas = {nome: _avaliar(probs, lambdas, resultados, reais) for nome, probs in opcoes}

    print(f"\n{'-' * 72}")
    print(f"  Experimento força de time: Copa {ano}  ({n} jogos, sem vazamento)")
    print(f"{'-' * 72}")
    print(f"  {'Opção':<22} {'pts':>4} {'pts/jogo':>9} {'cravadas':>9} {'result.':>8} {'RPS':>8}")
    print(f"  {'-' * 22} {'-' * 4} {'-' * 9} {'-' * 9} {'-' * 8} {'-' * 8}")
    for nome, _ in opcoes:
        m = metricas[nome]
        print(
            f"  {nome:<22} {m.pts_total:>4} {m.pts_total / n:>9.2f} "
            f"{m.cravadas:>9} {m.resultados:>8} {m.rps:>8.4f}"
        )

    print(f"  {'-' * 72}")

    # Veredito: melhor opção POR PONTOS (o objetivo), vs mercado, com bootstrap.
    base = metricas["Mercado (538 SPI)"]
    melhor_nome = max(metricas, key=lambda k: metricas[k].pts_total)
    melhor = metricas[melhor_nome]
    if melhor_nome != "Mercado (538 SPI)":
        # bootstrap_diff: positivo = b melhor; aqui b = melhor opção (queremos mais pontos).
        media, lo, hi, p_b = bootstrap_diff(
            [-x for x in base.pts_por_jogo], [-x for x in melhor.pts_por_jogo]
        )
        print(f"  Melhor por pontos: {melhor_nome} ({melhor.pts_total} vs mercado {base.pts_total})")
        print(f"  delta médio de pontos/jogo (melhor-mercado): {media:+.3f}  IC95% [{lo:+.3f}, {hi:+.3f}]")
        print(f"  P(melhor rende mais que mercado) = {p_b:.1%}")
        veredito = (
            f"{melhor_nome} AGREGA pontos sobre o mercado" if lo > 0
            else "diferença NÃO é significativa (IC cruza zero); mercado já basta"
        )
        print(f"  -> {veredito}")
    else:
        print("  -> O mercado puro rende o máximo de pontos; nenhuma mistura supera.")
    print(f"{'-' * 72}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest do prior de força")
    parser.add_argument("--ano", type=int, default=2022, help="ano da Copa a avaliar")
    args = parser.parse_args()
    comparar(args.ano)


if __name__ == "__main__":
    main()
