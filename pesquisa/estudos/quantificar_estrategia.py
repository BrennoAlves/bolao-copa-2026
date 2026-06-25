"""
P(1º lugar) na subliga por estratégia, com a multidão real.

Antes de ligar a diferenciação no daemon eu queria o número. Roda o Monte Carlo
(`simular`) sobre os próximos N jogos, partindo dos standings da subliga e dos
perfis empíricos dos adversários (`estimar_perfis`, dos picks raspados), e compara
P(1º) em cada estratégia: max_ev (o que o daemon faz hoje, cola no consenso);
espelhar (copia o palpite modal da multidão); diferenciar (melhor E[pontos] fora do
modal); otimizada (hill-climbing maximizando P(1º) direto).

Só vale ligar a diferenciação se otimizada/diferenciar passar do max_ev de forma
clara. Isso aqui são só os próximos N jogos, e num pool pequeno a variância é alta.

Uso:
    uv run python -m pesquisa.estudos.quantificar_estrategia            # próximos 8 jogos
    uv run python -m pesquisa.estudos.quantificar_estrategia --jogos 5 --sims 20000
    uv run python -m pesquisa.estudos.quantificar_estrategia --scrape   # força raspar
"""
from __future__ import annotations

import argparse
import sys

from loguru import logger

from bolao.config import Config, carregar_config
from bolao.fontes.odds_api import buscar_odds
from bolao.fontes.site import DadosBolao, carregar_dados_bolao, raspar_bolao
from bolao.modelo import eh_mata_mata, predizer
from bolao.nomes import casar_identidade
from bolao.simulador import Adversario, JogoSimulado, estimar_perfis, simular


def _resolver_meu_nome(config: Config, dados: DadosBolao) -> str | None:
    """Meu nome na classificação: BOLAO_NOME (exato) ou prefixo do e-mail."""
    nomes = [n for n, *_ in dados.classificacao]
    return casar_identidade(nomes, config.bolao_email, config.bolao_nome)


def main() -> None:
    parser = argparse.ArgumentParser(description="P(1º) por estratégia na subliga")
    parser.add_argument("--jogos", type=int, default=8, help="próximos N jogos (padrão 8)")
    parser.add_argument("--sims", type=int, default=10_000, help="simulações Monte Carlo")
    parser.add_argument("--scrape", action="store_true", help="força raspar o site")
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

    com_perfil = sum(1 for a in adversarios if a.perfil and a.perfil.n_jogos >= 3)

    todos = sorted(buscar_odds(config.odds_api_key), key=lambda j: j.kickoff_utc)
    proximos = todos[: args.jogos]
    if not proximos:
        print("Nenhum jogo futuro encontrado.")
        sys.exit(1)

    jogos_sim: list[JogoSimulado] = []
    for j in proximos:
        pred = predizer(
            time_casa=j.time_casa,
            time_fora=j.time_fora,
            prob_casa=j.prob_casa,
            prob_empate=j.prob_empate,
            prob_fora=j.prob_fora,
            lambda_total=j.lambda_total,
            mata_mata=eh_mata_mata(j.kickoff_utc),
            top_n=36,
        )
        jogos_sim.append(JogoSimulado.de_predicao(pred))

    resultados = simular(jogos_sim, meus_pontos, adversarios, n_sims=args.sims, seed=42)
    por_nome = {r.estrategia: r for r in resultados}
    base = por_nome.get("max_ev")

    # relatório
    lider_pts = max((a.pontos for a in adversarios), default=0.0)
    gap = meus_pontos - lider_pts
    pos = "liderando" if gap > 0 else ("empatado" if gap == 0 else f"{-gap:.0f} atrás")

    print("\n" + "=" * 64)
    print(f"  P(1º) no {dados.subleague_nome} | você ({meu_nome}): "
          f"{meus_pontos:.0f} pts ({pos})")
    print(f"  {len(adversarios)} adversários ({com_perfil} c/ perfil empírico >=3 jogos) "
          f"| {len(jogos_sim)} jogos | {args.sims:,} sims".replace(",", "."))
    print("-" * 64)
    print(f"  {'Estratégia':14}{'P(1º)':>10}{'E[pontos]':>12}{'delta vs max_ev':>16}")
    for r in resultados:
        delta = "" if not base else f"{(r.prob_campeao - base.prob_campeao) * 100:+.1f} pp"
        marca = " <<" if r is resultados[0] else ""
        print(f"  {r.estrategia:14}{r.prob_campeao:>9.1%}{r.pontos_esperados:>12.2f}"
              f"{delta:>14}{marca}")
    print("-" * 64)

    melhor = resultados[0]
    if base is not None:
        ganho = (melhor.prob_campeao - base.prob_campeao) * 100
        print(f"  Melhor: '{melhor.estrategia}': P(1º) {base.prob_campeao:.1%} -> "
              f"{melhor.prob_campeao:.1%} ({ganho:+.1f} pp vs max_ev)")
        if melhor.estrategia == "max_ev":
            veredito = "max_ev já é o melhor; diferenciar não ajuda neste cenário."
        elif ganho >= 2.0:
            veredito = f"diferenciar ajuda (+{ganho:.1f} pp); vale ligar no daemon."
        else:
            veredito = (f"ganho pequeno (+{ganho:.1f} pp), dentro do ruído; "
                        "decidir com mais jogos/sims antes de ligar.")
        print(f"  -> {veredito}")

    print("\n  Palpites da melhor estratégia:")
    for jogo, pick in zip(jogos_sim, melhor.palpites, strict=True):
        modal = jogo.labels[int(jogo.probs.argmax())]
        flag = "" if pick == modal else f"  (consenso: {modal})"
        print(f"     {jogo.time_casa} {pick} {jogo.time_fora}{flag}")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
