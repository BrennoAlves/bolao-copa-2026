"""Simulador do bolão com dados reais (standings e picks do site)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from bolao.config import carregar_config
from bolao.fontes import ajustar_probabilidades, buscar_odds, buscar_probabilidades
from bolao.fontes.site import DadosBolao, obter_dados_bolao
from bolao.modelo import predizer
from bolao.nomes import casar_identidade
from bolao.simulador import Adversario, JogoSimulado, simular


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulador do bolão com dados reais do site"
    )
    parser.add_argument(
        "--jogos", type=int, default=8,
        help="Quantos próximos jogos simular (padrão: 8)",
    )
    parser.add_argument(
        "--sims", type=int, default=10_000,
        help="Número de simulações Monte Carlo (padrão: 10000)",
    )
    parser.add_argument(
        "--so-scrape", action="store_true",
        help="Apenas raspa os dados do site, sem simular",
    )
    parser.add_argument(
        "--meu-nome", type=str, default=None,
        help="Seu nome na classificação (padrão: primeiro campo do email)",
    )
    args = parser.parse_args()

    config = carregar_config()

    # HTTP primeiro (rápido, sem browser); cai no scraping Playwright se falhar
    dados = obter_dados_bolao(config)
    if dados is None:
        print("Não foi possível obter dados do bolão. Verifique as credenciais.")
        sys.exit(1)

    _imprimir_standings(dados)

    if args.so_scrape:
        return

    # identifica o usuário na classificação: BOLAO_NOME (exato) ou prefixo do
    # e-mail, igual aos outros scripts de pesquisa (evita meus_pontos=0 quando
    # o nome de exibição não começa com o e-mail).
    meu_nome = args.meu_nome
    if meu_nome is None:
        nomes = [n for n, *_ in dados.classificacao]
        meu_nome = casar_identidade(nomes, config.bolao_email, config.bolao_nome)
    if meu_nome is None:
        print("Não achei seu nome na classificação. Use --meu-nome para especificar.")
        print("Standings:", [n for n, *_ in dados.classificacao])
        sys.exit(1)

    meus_pontos, adversarios = _extrair_adversarios(dados, meu_nome)

    if not adversarios:
        print(f"Nenhum adversário encontrado (seu nome na classificação: '{meu_nome}').")
        print("Use --meu-nome para especificar.")
        sys.exit(1)

    # busca e prediz os próximos jogos
    print(f"\nBuscando os próximos {args.jogos} jogos...")
    todos = sorted(buscar_odds(config.odds_api_key), key=lambda j: j.kickoff_utc)
    proximos = todos[: args.jogos]

    if not proximos:
        print("Nenhum jogo futuro encontrado.")
        sys.exit(1)

    jogos_sim = []
    for jogo in proximos:
        try:
            poly = buscar_probabilidades(jogo.time_casa, jogo.time_fora)
            prob_casa, prob_empate, prob_fora = ajustar_probabilidades(
                jogo.prob_casa, jogo.prob_empate, jogo.prob_fora, poly
            )
        except Exception:
            prob_casa, prob_empate, prob_fora = jogo.prob_casa, jogo.prob_empate, jogo.prob_fora

        pred = predizer(
            time_casa=jogo.time_casa,
            time_fora=jogo.time_fora,
            prob_casa=prob_casa,
            prob_empate=prob_empate,
            prob_fora=prob_fora,
            lambda_total=jogo.lambda_total,
            top_n=36,
        )
        jogos_sim.append(JogoSimulado.de_predicao(pred))

    # calibra aderência histórica pelos picks raspados
    adversarios = _calibrar_aderencia(adversarios, dados)

    # simula
    print(f"Rodando Monte Carlo ({args.sims:,} simulações)...".replace(",", "."))
    resultados = simular(
        jogos=jogos_sim,
        meus_pontos=meus_pontos,
        adversarios=adversarios,
        n_sims=args.sims,
    )

    # relatório
    lider_pts = max(a.pontos for a in adversarios)
    posicao = "liderando" if meus_pontos > lider_pts else (
        "empatado" if meus_pontos == lider_pts else f"{lider_pts - meus_pontos:.0f} pts atrás"
    )

    print("\n" + "=" * 60)
    print(f"  Simulação com dados reais | {dados.subleague_nome}")
    print(f"  Você ({meu_nome}): {meus_pontos:.0f} pts ({posicao}) | {len(adversarios)} adversários | {len(jogos_sim)} jogos")
    print("-" * 60)
    print(f"  {'Estratégia':14} {'P(campeão)':>11} {'E[pontos]':>10}")
    for r in resultados:
        marca = " (melhor)" if r is resultados[0] else ""
        print(f"  {r.estrategia:14} {r.prob_campeao:>10.1%} {r.pontos_esperados:>9.2f}{marca}")
    print("-" * 60)

    melhor = resultados[0]
    print(f"\n  Palpites da estratégia '{melhor.estrategia}':")
    for jogo, palpite in zip(jogos_sim, melhor.palpites, strict=False):
        print(f"     {jogo.time_casa} {palpite} {jogo.time_fora}")
    print("=" * 60 + "\n")


def _imprimir_standings(dados: DadosBolao) -> None:
    print(f"\n  Classificação {dados.subleague_nome} (atualizada {dados.scraped_at.strftime('%d/%m %H:%M')}):")
    for i, (nome, ap, av, pts) in enumerate(dados.classificacao, 1):
        marcador = ">" if i == dados.minha_posicao else "  "
        print(f"  {marcador} {i:2}. {nome:<20} {pts:>3} pts  (AP={ap}, AV={av})")


def _extrair_adversarios(
    dados: DadosBolao,
    meu_nome: str,
) -> tuple[float, list[Adversario]]:
    """Separa minha pontuação da lista de adversários."""
    meus_pontos = 0.0
    adversarios: list[Adversario] = []

    for nome, _ap, _av, pts in dados.classificacao:
        if nome.lower() == meu_nome.lower():
            meus_pontos = float(pts)
        else:
            adversarios.append(Adversario(nome=nome, pontos=float(pts)))

    return meus_pontos, adversarios


def _calibrar_aderencia(
    adversarios: list[Adversario],
    dados: DadosBolao,
) -> list[Adversario]:
    """Aderência = fração de jogos em que o adversário cravou o placar mais votado."""
    if not dados.palpites_por_jogo:
        return adversarios

    picks_modais: dict[str, tuple[int, int]] = {}
    for palpite_jogo in dados.palpites_por_jogo:
        contagem: dict[tuple[int, int], int] = {}
        for pick in palpite_jogo.picks.values():
            contagem[pick] = contagem.get(pick, 0) + 1
        if contagem:
            picks_modais[palpite_jogo.jogo_id] = max(contagem, key=lambda p: contagem[p])

    if not picks_modais:
        return adversarios

    calibrados = []
    for adv in adversarios:
        coincidencias = 0
        total = 0
        for palpite_jogo in dados.palpites_por_jogo:
            if adv.nome not in palpite_jogo.picks:
                continue
            total += 1
            modal = picks_modais.get(palpite_jogo.jogo_id)
            if modal and palpite_jogo.picks[adv.nome] == modal:
                coincidencias += 1

        aderencia = coincidencias / total if total >= 3 else 0.7
        calibrados.append(Adversario(
            nome=adv.nome,
            pontos=adv.pontos,
            aderencia=aderencia,
        ))
        logger.debug(
            "Aderência calibrada | {n}: {a:.1%} ({c}/{t} jogos)",
            n=adv.nome,
            a=aderencia,
            c=coincidencias,
            t=total,
        )

    return calibrados


if __name__ == "__main__":
    main()
