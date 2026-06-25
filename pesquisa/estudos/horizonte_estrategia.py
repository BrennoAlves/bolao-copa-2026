"""
Ligar tarde vs. agora: dá pra recuperar o gap diferenciando só na reta final, ou
tem que fugir do consenso desde já?

Aqui o objeto não é a estratégia mas a política de QUANDO virar a chave, sobre
todo o horizonte de jogos com odds. Roda o Monte Carlo
(`comparar_politicas_horizonte`) a partir dos standings e perfis reais da subliga,
nas mesmas simulações: nunca (max_ev) joga max E[pontos] em todos (o daemon hoje);
tarde joga max-EV até o gatilho da reta final e otimiza no sufixo; agora
diferencia desde o 1º jogo.

O gatilho 'tarde' replica `_modo_reta_final`: dispara no 1º mata-mata em que restam
<= JOGOS_PARA_MODO_FINAL jogos no horizonte. Se a reta final não cabe no horizonte
com odds, 'tarde' vira 'nunca' (não há o que diferenciar ainda). O horizonte é só
o que a Odds API postou (uma fase por vez), não o torneio inteiro.

Uso:
    uv run python -m pesquisa.estudos.horizonte_estrategia              # todos os jogos com odds
    uv run python -m pesquisa.estudos.horizonte_estrategia --jogos 12 --sims 20000
    uv run python -m pesquisa.estudos.horizonte_estrategia --scrape     # força raspar
"""
from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime

from loguru import logger

from bolao.backtest import RegistroBacktest, listar_registros_backtest
from bolao.config import Config, carregar_config
from bolao.fontes.odds_api import buscar_odds
from bolao.fontes.site import DadosBolao, carregar_dados_bolao, raspar_bolao
from bolao.modelo import eh_mata_mata, predizer
from bolao.nomes import casar_identidade
from bolao.simulador import (
    Adversario,
    JogoSimulado,
    comparar_politicas_horizonte,
    estimar_perfis,
)


def _resolver_meu_nome(config: Config, dados: DadosBolao) -> str | None:
    """Meu nome na classificação: BOLAO_NOME (exato) ou prefixo do e-mail."""
    nomes = [n for n, *_ in dados.classificacao]
    return casar_identidade(nomes, config.bolao_email, config.bolao_nome)


def _horizonte_de_registros(n: int) -> tuple[list[JogoSimulado], list[RegistroBacktest]]:
    """
    Fallback offline quando a Odds API está sem crédito: monta o horizonte com os
    `n` registros de backtest mais recentes (probabilidades já gravadas). A
    competitividade vem do último lote real, não dos jogos exatos por vir.
    """
    regs = [
        r for r in listar_registros_backtest()
        if r.prob_casa_modelo + r.prob_empate_modelo + r.prob_fora_modelo > 0.5  # exclui stubs (prob=0)
    ]
    regs = sorted(regs, key=lambda r: r.kickoff_utc)[-n:]
    jogos = [
        JogoSimulado.de_predicao(predizer(
            time_casa=r.time_casa, time_fora=r.time_fora,
            prob_casa=r.prob_casa_modelo, prob_empate=r.prob_empate_modelo,
            prob_fora=r.prob_fora_modelo, lambda_total=r.lambda_casa + r.lambda_fora,
            mata_mata=eh_mata_mata(r.kickoff_utc), top_n=36,
        ))
        for r in regs
    ]
    return jogos, regs


def _indice_gatilho(proximos: list, jogos_para_modo_final: int) -> int:
    """
    Índice (0-based, jogos ordenados por kickoff) do 1º jogo em que a política
    'tarde' passa a diferenciar; replica `_modo_reta_final`: mata-mata E
    restantes (do jogo em diante) <= limiar. Retorna len(proximos) se não chega.
    """
    n = len(proximos)
    for g, jogo in enumerate(proximos):
        restantes = n - g
        if eh_mata_mata(jogo.kickoff_utc) and restantes <= jogos_para_modo_final:
            return g
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Política tarde vs agora, P(1º) na subliga")
    parser.add_argument("--jogos", type=int, default=0, help="teto do horizonte (0 = todos com odds)")
    parser.add_argument("--sims", type=int, default=10_000, help="simulações Monte Carlo")
    parser.add_argument("--scrape", action="store_true", help="força raspar o site")
    parser.add_argument("--de-registros", type=int, default=0, dest="de_registros",
                        help="fallback offline: horizonte dos N registros mais recentes (sem Odds API)")
    parser.add_argument("--gatilho", type=int, default=-1,
                        help="força o índice da reta final (default: calculado pelos kickoffs)")
    args = parser.parse_args()

    config = carregar_config()
    logger.disable("bolao.modelo")  # silencia o log por-jogo durante o lote

    # dados reais do bolão
    dados = raspar_bolao(config) if args.scrape else carregar_dados_bolao()
    if dados is None:
        dados = raspar_bolao(config)
    if dados is None or not dados.classificacao:
        print("Sem dados do bolão (cache vazio e scrape falhou).")
        sys.exit(1)

    meu_nome = _resolver_meu_nome(config, dados)
    if meu_nome is None:
        print("Não achei seu nome na classificação. Standings:",
              [n for n, *_ in dados.classificacao])
        sys.exit(1)

    membros = set(config.bolao_membros)
    if not membros:
        print("Defina BOLAO_MEMBROS no .env para restringir à subliga.")
        sys.exit(1)
    perfis = estimar_perfis(dados, meu_nome)
    meus_pontos = 0.0
    adversarios: list[Adversario] = []
    for nome, _ap, _av, pts in dados.classificacao:
        if nome not in membros:
            continue  # ignora o resto da liga, só a subliga importa
        if nome == meu_nome:
            meus_pontos = float(pts)
        else:
            adversarios.append(Adversario(nome=nome, pontos=float(pts), perfil=perfis.get(nome)))

    if not adversarios:
        print("Sem adversários da subliga na classificação, nada a comparar.")
        sys.exit(1)
    com_perfil = sum(1 for a in adversarios if a.perfil and a.perfil.n_jogos >= 3)

    if args.de_registros > 0:
        jogos_sim, proximos = _horizonte_de_registros(args.de_registros)
        fonte = f"{len(jogos_sim)} registros recentes (proxy, Odds API sem crédito)"
    else:
        todos = sorted(buscar_odds(config.odds_api_key), key=lambda j: j.kickoff_utc)
        proximos = [j for j in todos if j.kickoff_utc > datetime.now(UTC)]
        if args.jogos > 0:
            proximos = proximos[: args.jogos]
        jogos_sim = [
            JogoSimulado.de_predicao(predizer(
                time_casa=j.time_casa, time_fora=j.time_fora,
                prob_casa=j.prob_casa, prob_empate=j.prob_empate, prob_fora=j.prob_fora,
                lambda_total=j.lambda_total, mata_mata=eh_mata_mata(j.kickoff_utc), top_n=36,
            ))
            for j in proximos
        ]
        fonte = f"{len(jogos_sim)} jogos c/ odds"

    if not jogos_sim:
        print("Nenhum jogo no horizonte (Odds API sem crédito? tente --de-registros 16).")
        sys.exit(1)

    idx_gatilho = (
        args.gatilho if args.gatilho >= 0
        else _indice_gatilho(proximos, config.jogos_para_modo_final)
    )
    resultados = comparar_politicas_horizonte(
        jogos_sim, meus_pontos, adversarios, idx_gatilho, n_sims=args.sims, seed=42
    )
    por_nome = {r.nome: r for r in resultados}
    nunca = por_nome["nunca (max_ev)"]
    tarde = por_nome["tarde (gatilho atual)"]
    agora = por_nome["agora"]

    # relatório
    lider_pts = max((a.pontos for a in adversarios), default=0.0)
    gap = meus_pontos - lider_pts
    pos = "liderando" if gap > 0 else ("empatado" if gap == 0 else f"{-gap:.0f} atrás do líder")
    n = len(jogos_sim)
    if idx_gatilho >= n:
        gatilho_txt = (f"a reta final NÃO chega neste horizonte; 'tarde' joga max-EV "
                       f"em todos os {n} (igual a 'nunca')")
    else:
        gatilho_txt = (f"'tarde' joga max-EV nos primeiros {idx_gatilho} e diferencia "
                       f"nos últimos {n - idx_gatilho}")

    print("\n" + "=" * 70)
    print(f"  Horizonte: {dados.subleague_nome} | você ({meu_nome}): "
          f"{meus_pontos:.0f} pts ({pos})")
    print(f"  {len(adversarios)} adversários ({com_perfil} c/ perfil empírico >=3 jogos) "
          f"| {fonte} | {args.sims:,} sims".replace(",", "."))
    print(f"  Gatilho da reta final (JOGOS_PARA_MODO_FINAL={config.jogos_para_modo_final}): {gatilho_txt}")
    print("-" * 70)
    print(f"  {'Política':24}{'P(1º)':>9}{'P(top-3)':>10}{'pos.méd':>9}{'E[pontos]':>11}")
    for r in (nunca, tarde, agora):
        marca = " <<" if r is max(resultados, key=lambda x: x.prob_campeao) else ""
        print(f"  {r.nome:24}{r.prob_campeao:>8.1%}{r.prob_top3:>10.1%}"
              f"{r.pos_media:>9.2f}{r.pontos_esperados:>11.2f}{marca}")
    print("-" * 70)

    # veredito
    delta_pp = (agora.prob_campeao - tarde.prob_campeao) * 100
    delta_nunca_pp = (agora.prob_campeao - nunca.prob_campeao) * 100
    if agora.prob_campeao < 0.02:
        veredito = (f"P(1º) ínfima em qualquer política ({agora.prob_campeao:.1%}): o gap é "
                    "grande p/ os jogos com odds agora; rodar de novo quando saírem as próximas fases.")
    elif idx_gatilho >= n:
        veredito = (f"o gatilho nem dispara no horizonte; a escolha real é max-EV vs diferenciar agora: "
                    f"{nunca.prob_campeao:.1%} -> {agora.prob_campeao:.1%} ({delta_nunca_pp:+.1f} pp).")
    elif delta_pp >= 2.0:
        veredito = (f"esperar custa: diferenciar agora dá P(1º) {tarde.prob_campeao:.1%} -> "
                    f"{agora.prob_campeao:.1%} ({delta_pp:+.1f} pp) vs ligar só na reta final. Antecipar vale.")
    else:
        veredito = (f"esperar quase não custa ({delta_pp:+.1f} pp): com este horizonte o gatilho atual "
                    f"recupera quase tanto quanto diferenciar agora.")
    print(f"  -> {veredito}")

    print("\n  Palpites de 'agora' que fogem do consenso (max_ev):")
    divergiu = False
    for jogo, p_now, p_max in zip(jogos_sim, agora.palpites, nunca.palpites, strict=True):
        if p_now != p_max:
            divergiu = True
            print(f"     {jogo.time_casa} {p_now} {jogo.time_fora}  (max_ev: {p_max})")
    if not divergiu:
        print("     (nenhum: 'agora' coincide com max_ev neste horizonte)")
    print("  (horizonte = só os jogos com odds postadas, não o torneio inteiro.)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
