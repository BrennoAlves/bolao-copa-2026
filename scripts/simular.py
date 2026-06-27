"""Simulador Monte Carlo do bolão: qual estratégia maximiza P(campeão)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from bolao.config import carregar_config
from bolao.fontes import ajustar_probabilidades, buscar_odds, buscar_probabilidades
from bolao.modelo import predizer
from bolao.simulador import Adversario, JogoSimulado, simular


def _parse_rival(texto: str, aderencia: float) -> Adversario:
    """Converte "Nome:pontos" em Adversario."""
    try:
        nome, pontos = texto.rsplit(":", 1)
        return Adversario(nome=nome.strip(), pontos=float(pontos), aderencia=aderencia)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Rival inválido: '{texto}', use o formato \"Nome:pontos\""
        ) from e


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulador Monte Carlo do bolão"
    )
    parser.add_argument("--meus-pontos", type=float, required=True, help="Sua pontuação atual")
    parser.add_argument(
        "--rival", action="append", required=True,
        help='Adversário no formato "Nome:pontos" (repita para cada um)',
    )
    parser.add_argument("--jogos", type=int, default=8, help="Quantos próximos jogos simular (padrão: 8)")
    parser.add_argument("--sims", type=int, default=10_000, help="Número de simulações (padrão: 10000)")
    parser.add_argument(
        "--aderencia", type=float, default=0.7,
        help="Prob. de cada adversário palpitar o placar óbvio (padrão: 0.7)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Semente para reprodutibilidade")
    args = parser.parse_args()

    config = carregar_config()
    adversarios = [_parse_rival(r, args.aderencia) for r in args.rival]

    # busca e prediz os próximos jogos
    todos = sorted(buscar_odds(config.odds_api_key), key=lambda j: j.kickoff_utc)
    proximos = todos[: args.jogos]
    if not proximos:
        logger.error("Nenhum jogo futuro encontrado.")
        sys.exit(1)

    jogos = []
    for jogo in proximos:
        poly = buscar_probabilidades(jogo.time_casa, jogo.time_fora)
        prob_casa, prob_empate, prob_fora = ajustar_probabilidades(
            jogo.prob_casa, jogo.prob_empate, jogo.prob_fora, poly
        )
        pred = predizer(
            time_casa=jogo.time_casa,
            time_fora=jogo.time_fora,
            prob_casa=prob_casa,
            prob_empate=prob_empate,
            prob_fora=prob_fora,
            lambda_total=jogo.lambda_total,
            top_n=36,  # grade completa para o simulador
        )
        jogos.append(JogoSimulado.de_predicao(pred))

    # simula
    resultados = simular(
        jogos=jogos,
        meus_pontos=args.meus_pontos,
        adversarios=adversarios,
        n_sims=args.sims,
        seed=args.seed,
    )

    # relatório
    lider = max(a.pontos for a in adversarios)
    posicao = "liderando" if args.meus_pontos > lider else (
        "empatado" if args.meus_pontos == lider else f"{lider - args.meus_pontos:.0f} pts atrás"
    )

    print("\n" + "=" * 60)
    print(f"  Monte Carlo: {args.sims:,} simulações".replace(",", "."))
    print(f"  Você: {args.meus_pontos:.0f} pts ({posicao}) | {len(adversarios)} adversários | {len(jogos)} jogos")
    print("-" * 60)
    print(f"  {'Estratégia':14} {'P(campeão)':>11} {'E[pontos]':>10}")
    for r in resultados:
        marca = " (melhor)" if r is resultados[0] else ""
        print(f"  {r.estrategia:14} {r.prob_campeao:>10.1%} {r.pontos_esperados:>9.2f}{marca}")
    print("-" * 60)

    melhor = resultados[0]
    print(f"  Palpites da estratégia '{melhor.estrategia}':")
    for jogo, palpite in zip(jogos, melhor.palpites, strict=False):
        print(f"     {jogo.time_casa} {palpite} {jogo.time_fora}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
