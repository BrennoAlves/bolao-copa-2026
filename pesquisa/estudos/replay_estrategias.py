"""
Replay sobre os jogos REAIS já encerrados, sem modelo de desfecho.

Pego o placar real de cada jogo e os picks reais da turma (raspados) e pergunto:
que posição eu teria hoje na subliga se tivesse jogado cada estratégia, com os
adversários fixos nos picks que eles de fato deram?

As estratégias não precisam do grid do modelo: eu (real) usa meus palpites de
verdade; espelhar copia o palpite modal da turma a cada jogo; favorito 1x0 aposta
sempre 1x0 no lado majoritário; diferenciar pega o 2º placar mais comum (sai do
modal mas continua plausível).

Resultado e picks são fatos, então não vaza nada. É um campeonato só, olhando pra
trás, mostra o que teria acontecido, não diz qual estratégia ganha no longo prazo.

Uso: uv run python -m pesquisa.estudos.replay_estrategias
"""
from __future__ import annotations

import sys
from collections import Counter

from bolao.config import carregar_config
from bolao.fontes.site import PalpiteJogo, carregar_dados_bolao
from bolao.resultados import calcular_pontos


def _modal(picks: dict[str, tuple[int, int]]) -> tuple[int, int]:
    return Counter(picks.values()).most_common(1)[0][0]


def _favorito_1x0(picks: dict[str, tuple[int, int]]) -> tuple[int, int]:
    """1x0 no lado majoritário (sem odds históricas, o consenso vira o favorito)."""
    casa = sum(1 for gc, gf in picks.values() if gc > gf)
    fora = sum(1 for gc, gf in picks.values() if gc < gf)
    return (1, 0) if casa >= fora else (0, 1)


def _diferenciar(picks: dict[str, tuple[int, int]]) -> tuple[int, int]:
    """2º placar mais comum (sai do modal, mas continua plausível)."""
    comuns = Counter(picks.values()).most_common(2)
    return comuns[1][0] if len(comuns) > 1 else comuns[0][0]


def main() -> None:
    config = carregar_config()
    membros = set(config.bolao_membros)
    if not membros:
        print("Defina BOLAO_MEMBROS no .env para restringir à subliga.")
        sys.exit(1)
    meu_alvo = config.bolao_email.split("@")[0].lower()

    dados = carregar_dados_bolao()
    if dados is None:
        print("Sem dados no cache, rode o scrape antes.")
        sys.exit(1)

    meu_nome = next((m for m in membros if meu_alvo.startswith(m.split()[0].lower())), None)
    if meu_nome is None:
        print("Não resolvi seu nome entre os membros:", membros)
        sys.exit(1)

    # Jogos encerrados (com placar real) e dedup por conteúdo de picks.
    jogos: list[PalpiteJogo] = []
    vistos: set = set()
    for pj in dados.palpites_por_jogo:
        if pj.resultado_real is None:
            continue
        chave = frozenset(pj.picks.items())
        if chave in vistos:
            continue
        vistos.add(chave)
        jogos.append(pj)

    if not jogos:
        print("Nenhum jogo encerrado com placar real no cache ainda.")
        sys.exit(0)

    estrategias = {
        "eu (real)": lambda pj: pj.picks.get(meu_nome),
        "espelhar": lambda pj: _modal({n: p for n, p in pj.picks.items() if n in membros}),
        "diferenciar": lambda pj: _diferenciar({n: p for n, p in pj.picks.items() if n in membros}),
        "favorito 1x0": lambda pj: _favorito_1x0(pj.picks),
    }

    # Pontos reais dos adversários (fixos) sobre os mesmos jogos.
    adversarios = [m for m in membros if m != meu_nome]
    pts_adv = dict.fromkeys(adversarios, 0)
    for pj in jogos:
        real = pj.resultado_real
        assert real is not None
        for m in adversarios:
            pick = pj.picks.get(m)
            if pick is not None:
                pts_adv[m] += calcular_pontos(pick[0], pick[1], real[0], real[1])[0]

    print(f"\n{'=' * 64}")
    print(f"  Replay sobre {len(jogos)} jogo(s) encerrado(s) | você: {meu_nome}")
    print("  (adversários fixos nos picks reais; placar real de cada jogo)")
    print(f"{'-' * 64}")
    print(f"  {'Estratégia':16}{'meus pts':>9}   posição na subliga")

    linhas = []
    for nome, escolher in estrategias.items():
        meus = 0
        faltou = 0
        for pj in jogos:
            real = pj.resultado_real
            assert real is not None
            pick = escolher(pj)
            if pick is None:
                faltou += 1
                continue
            meus += calcular_pontos(pick[0], pick[1], real[0], real[1])[0]
        # posição = 1 + nº de adversários com estritamente mais pontos
        acima = sum(1 for p in pts_adv.values() if p > meus)
        empate = sum(1 for p in pts_adv.values() if p == meus)
        pos = acima + 1
        pos_txt = f"{pos}º de {len(membros)}" + (f" (+{empate} empat.)" if empate else "")
        nota = f"  [{faltou} s/ pick]" if faltou else ""
        linhas.append((nome, meus, pos, pos_txt + nota))

    for nome, meus, _pos, pos_txt in linhas:
        print(f"  {nome:16}{meus:>9}   {pos_txt}")
    print(f"{'-' * 64}")

    real = next(linha for linha in linhas if linha[0] == "eu (real)")
    melhor = min(linhas, key=lambda x: (x[2], -x[1]))
    if melhor[0] == "eu (real)":
        print("  -> Seus palpites reais foram a melhor estratégia neste recorte.")
    else:
        print(f"  -> '{melhor[0]}' teria te posto em {melhor[3].split()[0]} "
              f"(real: {real[3].split()[0]}, {real[1]} pts).")
    print("  (um campeonato só, retrospectivo, não diz qual estratégia ganha sempre.)")
    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    main()
