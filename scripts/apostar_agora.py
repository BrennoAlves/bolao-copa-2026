"""Aposta manual de um jogo específico agora (testes ou emergência)."""
from __future__ import annotations

import sys
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from bolao.apostador import apostar
from bolao.config import carregar_config
from bolao.fontes import (
    ajustar_probabilidades,
    buscar_jogo_por_times,
    buscar_odds,
    buscar_probabilidades,
)
from bolao.modelo import eh_mata_mata, predizer
from bolao.notificador import notificar_falha, notificar_sucesso


def _imprimir_predicao(predicao, time_casa: str, time_fora: str, kickoff) -> None:
    tz_br = ZoneInfo("America/Sao_Paulo")
    hora = kickoff.astimezone(tz_br).strftime("%d/%m às %H:%M BRT")

    print("\n" + "=" * 54)
    print("  Bolão")
    print(f"  {time_casa}  x  {time_fora}")
    print(f"  Kickoff: {hora}")
    print("-" * 54)
    print("  Probabilidades")
    print(f"  {'Casa:':8} {predicao.prob_casa:.1%}")
    print(f"  {'Empate:':8} {predicao.prob_empate:.1%}")
    print(f"  {'Fora:':8} {predicao.prob_fora:.1%}")
    print("-" * 54)
    print("  Top 5 placares (cravada = +3 pts, resultado = +1 pt)")
    for i, p in enumerate(predicao.placares[:5]):
        print(f"  {i+1}. {p.label:12}  {p.probabilidade:.1%}  E={p.pontos_esperados:.2f} pts")
    print("-" * 54)
    aposta = predicao.melhor_placar
    print(f"  Aposta: {aposta.label}  (E={aposta.pontos_esperados:.2f} pts)")
    print(f"  Confiança: {predicao.confianca}")
    print("=" * 54 + "\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Apostador manual do bolão")
    parser.add_argument("time_casa", nargs="?", help="Nome do time da casa")
    parser.add_argument("time_fora", nargs="?", help="Nome do time visitante")
    parser.add_argument("--listar", action="store_true", help="Lista todos os jogos futuros disponíveis")
    parser.add_argument("--apenas-predizer", action="store_true", help="Só mostra predição, não aposta")
    args = parser.parse_args()

    config = carregar_config()

    # lista jogos
    if args.listar:
        jogos = buscar_odds(config.odds_api_key)
        tz_br = ZoneInfo("America/Sao_Paulo")
        print(f"\n{'#':>3}  {'Kickoff':16}  {'Casa':20}  {'Fora':20}")
        print("-" * 65)
        for i, j in enumerate(jogos, 1):
            k = j.kickoff_utc.astimezone(tz_br).strftime("%d/%m %H:%M BRT")
            print(f"{i:>3}  {k:16}  {j.time_casa:20}  {j.time_fora:20}")
        print()
        return

    # precisa dos nomes dos times
    if not args.time_casa or not args.time_fora:
        parser.print_help()
        sys.exit(1)

    # busca dados do jogo
    jogo = buscar_jogo_por_times(config.odds_api_key, args.time_casa, args.time_fora)
    if jogo is None:
        logger.error("Jogo não encontrado. Use --listar para ver jogos disponíveis.")
        sys.exit(1)

    # ajusta com Polymarket
    poly = buscar_probabilidades(jogo.time_casa, jogo.time_fora)
    prob_casa, prob_empate, prob_fora = ajustar_probabilidades(
        jogo.prob_casa, jogo.prob_empate, jogo.prob_fora, poly
    )

    # predição
    mata_mata = eh_mata_mata(jogo.kickoff_utc)
    if mata_mata:
        print("  Mata-mata: placar previsto até o fim da prorrogação")
    predicao = predizer(
        time_casa=jogo.time_casa,
        time_fora=jogo.time_fora,
        prob_casa=prob_casa,
        prob_empate=prob_empate,
        prob_fora=prob_fora,
        lambda_total=jogo.lambda_total,
        mata_mata=mata_mata,
    )
    _imprimir_predicao(predicao, jogo.time_casa, jogo.time_fora, jogo.kickoff_utc)

    if args.apenas_predizer:
        return

    # confirma aposta
    resposta = input(f"  Apostar '{predicao.melhor_placar.label}'? [s/N] ").strip().lower()
    if resposta != "s":
        print("  Aposta cancelada.")
        return

    resultado = apostar(
        email=config.bolao_email,
        password=config.bolao_password,
        subleague=config.bolao_subleague,
        predicao=predicao,
    )

    tz_br = ZoneInfo("America/Sao_Paulo")
    kickoff_local = jogo.kickoff_utc.astimezone(tz_br)

    if resultado.sucesso:
        print("  Palpite registrado com sucesso.")
        notificar_sucesso(config.cc_api_url, config.cc_token, jogo.time_casa, jogo.time_fora, kickoff_local, predicao)
    else:
        print(f"  Falha: {resultado.mensagem}")
        notificar_falha(config.cc_api_url, config.cc_token, jogo.time_casa, jogo.time_fora, kickoff_local, predicao, resultado.mensagem)


if __name__ == "__main__":
    main()
