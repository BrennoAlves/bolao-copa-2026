"""
Casa o delta de Elo (computado aqui) com o forecast do 538 jogo a jogo, mede o RPS
de cada opção e da mistura, e compara as diferenças com bootstrap pareado (mesmo
jogo, opções diferentes). Sem vazamento: tudo que entra é anterior ao apito.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import numpy as np

from bolao.modelo import predizer
from pesquisa.experimento.dados import Partida
from pesquisa.experimento.elo import ParametrosEmpate, SnapshotJogo, prob_1x2_elo
from pesquisa.experimento.mercado import JogoMercado

Prob1X2 = tuple[float, float, float]


# Apelidos que diferem entre 538 e martj42 (canonicalizados após normalizar).
_ALIASES = {
    "usa": "united states",
    "korea republic": "south korea",
    "ir iran": "iran",
}


def normalizar_nome(nome: str) -> str:
    """Casefold + remoção de acentos + apelidos, para casar nomes entre fontes."""
    sem_acento = "".join(
        c for c in unicodedata.normalize("NFKD", nome) if not unicodedata.combining(c)
    )
    base = sem_acento.casefold().strip()
    return _ALIASES.get(base, base)


def _orientar_elo(
    partida: Partida, dr: float, params: ParametrosEmpate, time1: str
) -> Prob1X2:
    """Probabilidades de Elo reordenadas para (time1, empate, time2), a ordem do 538.

    O delta de Elo vem no referencial casa/fora do histórico; aqui remapeio para a
    ordem em que o 538 listou os times.
    """
    pc, pe, pf = prob_1x2_elo(dr, params)
    if normalizar_nome(partida.time_casa) == normalizar_nome(time1):
        return pc, pe, pf
    # o 538 inverteu a ordem em relação à fonte histórica
    return pf, pe, pc


def blend(a: Prob1X2, b: Prob1X2, peso_b: float) -> Prob1X2:
    """Mistura linear renormalizada: peso_b=0 dá só a; peso_b=1 dá só b."""
    p = [(1 - peso_b) * x + peso_b * y for x, y in zip(a, b, strict=True)]
    s = sum(p)
    return p[0] / s, p[1] / s, p[2] / s


@dataclass(frozen=True)
class JogoCasado:
    """Um jogo da Copa 2022 com as probabilidades de mercado e de Elo alinhadas."""

    jogo: JogoMercado
    probs_mercado: Prob1X2
    probs_elo: Prob1X2


def casar(
    jogos_mercado: list[JogoMercado],
    snapshots: list[SnapshotJogo],
    params: ParametrosEmpate,
) -> tuple[list[JogoCasado], list[JogoMercado]]:
    """
    Para cada jogo do 538, encontra o snapshot de Elo correspondente (mesma data,
    mesmo par de times) e alinha as orientações. Devolve (casados, não-casados).
    """
    indice: dict[tuple, SnapshotJogo] = {}
    for s in snapshots:
        chave = (s.partida.data, frozenset({
            normalizar_nome(s.partida.time_casa),
            normalizar_nome(s.partida.time_fora),
        }))
        indice[chave] = s

    casados: list[JogoCasado] = []
    faltantes: list[JogoMercado] = []
    for j in jogos_mercado:
        chave = (j.data, frozenset({normalizar_nome(j.time1), normalizar_nome(j.time2)}))
        snap = indice.get(chave)
        if snap is None:
            faltantes.append(j)
            continue
        casados.append(
            JogoCasado(
                jogo=j,
                probs_mercado=(j.prob1, j.prob_empate, j.prob2),
                probs_elo=_orientar_elo(snap.partida, snap.dr_efetivo, params, j.time1),
            )
        )
    return casados, faltantes


def placar_previsto(probs: Prob1X2, lambda_total: float) -> tuple[int, int]:
    """
    Placar que o modelo apostaria dado o 1X2 e o lambda de gols.

    Chama o mesmo `predizer` da produção, então o que sai daqui é o que o bolão
    pontuaria. O lambda_total é igual para todas as opções; só o 1X2 muda.
    """
    pred = predizer(
        time_casa="t1",
        time_fora="t2",
        prob_casa=probs[0],
        prob_empate=probs[1],
        prob_fora=probs[2],
        lambda_total=lambda_total,
        top_n=1,
    )
    aposta = pred.melhor_placar
    return aposta.gols_casa, aposta.gols_fora


def pontos_bolao(previsto: tuple[int, int], real: tuple[int, int]) -> int:
    """Pontuação do bolão: 3 (placar exato), 1 (resultado certo) ou 0."""
    if previsto == real:
        return 3
    res_prev = (previsto[0] > previsto[1]) - (previsto[0] < previsto[1])
    res_real = (real[0] > real[1]) - (real[0] < real[1])
    return 1 if res_prev == res_real else 0


def bootstrap_diff(
    rps_a: list[float],
    rps_b: list[float],
    n_amostras: int = 10_000,
    semente: int = 42,
) -> tuple[float, float, float, float]:
    """
    Bootstrap pareado da diferença média (a menos b) de RPS por jogo.

    Positivo significa b melhor (RPS menor). Retorna (média, ic_baixo, ic_alto,
    p_b_melhor), onde p_b_melhor é a fração de reamostragens em que b supera a.
    """
    diffs = np.array(rps_a) - np.array(rps_b)
    rng = np.random.default_rng(semente)
    n = len(diffs)
    medias = np.array(
        [diffs[rng.integers(0, n, n)].mean() for _ in range(n_amostras)]
    )
    return (
        float(diffs.mean()),
        float(np.percentile(medias, 2.5)),
        float(np.percentile(medias, 97.5)),
        float((medias > 0).mean()),
    )
