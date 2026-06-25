"""
Análise model-free da multidão real da subliga.

Usa os picks reais raspados (palpites_por_jogo) e o placar real de cada jogo (de
quem cravou). Não usa o modelo de placar, então independe do teste do modelo. Mede:
- aglomeração: quanto a multidão concentra no placar modal (pré-condição para
  diferenciar valer a pena);
- estrutura dos pontos: as posições vêm de cravadas compartilhadas (não separam)
  ou de cravadas "donas" (poucos acertaram, então saltam)?
- onde você ganhou ou perdeu terreno vs. o líder.

Uso: uv run python -m pesquisa.estudos.analise_multidao
"""
from __future__ import annotations

from collections import Counter

from bolao.config import carregar_config
from bolao.fontes.site import carregar_dados_bolao
from bolao.nomes import casar_identidade


def _fmt(p: tuple[int, int]) -> str:
    return f"{p[0]}x{p[1]}"


def main() -> None:
    config = carregar_config()
    membros = set(config.bolao_membros)
    if not membros:
        print("Defina BOLAO_MEMBROS no .env para o recorte da subliga.")
        return

    dados = carregar_dados_bolao()
    if dados is None:
        print("Sem dados no cache, rode o scrape primeiro.")
        return

    eu = casar_identidade([n for n, *_ in dados.classificacao],
                          config.bolao_email, config.bolao_nome)

    # Dedup: o scraper às vezes repete o último card; chave = (id, picks).
    vistos: set = set()
    jogos = []
    for pj in dados.palpites_por_jogo:
        chave = (pj.jogo_id, frozenset(pj.picks.items()))
        if chave in vistos:
            continue
        vistos.add(chave)
        jogos.append(pj)

    pontos_total: Counter = Counter()
    print(f"{'='*78}")
    print("  Subliga: jogo a jogo (multidão geral + recorte dos membros)")
    print(f"{'='*78}\n")

    for pj in jogos:
        real = pj.resultado_real
        # recorte da subliga
        picks20 = {n: p for n, p in pj.picks.items() if n in membros}
        pts20 = {n: pj.pontos.get(n, 0) for n in picks20}

        # Aglomeração na multidão geral
        cnt_geral = Counter(pj.picks.values())
        modal_g, n_modal_g = cnt_geral.most_common(1)[0]
        share_g = n_modal_g / sum(cnt_geral.values())

        print(f"### {pj.jogo_id}")
        print(f"    resultado real: {_fmt(real) if real else '??? (ninguém cravou)'}")
        print(f"    multidão ({sum(cnt_geral.values())}): modal {_fmt(modal_g)} = {share_g:.0%}", end="")
        if real:
            cravaram_g = sum(1 for p in pj.picks.values() if p == real)
            print(f"  | cravaram o real: {cravaram_g}/{sum(cnt_geral.values())} ({cravaram_g/sum(cnt_geral.values()):.0%})")
        else:
            print()

        if picks20:
            cravaram20 = [n for n, p in picks20.items() if real and p == real]
            print(f"    subliga ({len(picks20)}/{len(membros)}): " +
                  ", ".join(f"{n.split()[0]} {_fmt(p)}{'*' if real and p==real else ''}"
                            for n, p in sorted(picks20.items())))
            if real:
                print(f"    cravaram na subliga: {len(cravaram20)} "
                      f"({', '.join(n.split()[0] for n in cravaram20) or '-'})")
            for n, p in pts20.items():
                pontos_total[n] += p
        print()

    # resumo: pontos acumulados no recorte da subliga (so jogos raspados)
    print(f"{'-'*78}")
    print("  Pontos no recorte (8 jogos raspados, só membros com pick em cada jogo):")
    for n, p in pontos_total.most_common():
        marca = "  <- você" if n == eu else ""
        print(f"    {n:<22} {p:>3} pts{marca}")
    print(f"{'-'*78}\n")

    # cravada compartilhada vs. cravada 'dona'
    print("  Onde as cravadas aconteceram (multidão geral):")
    crav_modal = 0   # cravadas em que o real == placar modal (compartilhada)
    crav_off = 0     # cravadas em que o real != modal (poucos donos, separa)
    for pj in jogos:
        real = pj.resultado_real
        if not real:
            continue
        cnt = Counter(pj.picks.values())
        modal = cnt.most_common(1)[0][0]
        n_real = cnt.get(real, 0)
        tag = "MODAL (compartilhada)" if real == modal else f"off-modal (só {n_real} donos)"
        print(f"    {pj.jogo_id:<34} real {_fmt(real):<5} -> {tag}")
        if real == modal:
            crav_modal += 1
        else:
            crav_off += 1
    print(f"\n    Resultados que bateram no modal da multidão: {crav_modal}")
    print(f"    Resultados 'surpresa' (fora do modal):       {crav_off}")
    print(f"{'-'*78}\n")


if __name__ == "__main__":
    main()
