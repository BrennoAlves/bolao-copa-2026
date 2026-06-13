"""Modelo de placares com Poisson independente e correção Dixon-Coles.

Os lambdas saem de um ajuste inverso que reproduz as probabilidades 1X2 do
mercado. A aposta sugerida maximiza E[pontos] = 2*P(placar) + P(resultado).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
from loguru import logger
from scipy.optimize import minimize
from scipy.stats import poisson

# Placar máximo considerado (5x5 cobre >99% dos jogos reais)
_MAX_GOLS = 5

# corrige a superestimação de 0x0 e 1x1 do modelo independente
_RHO = -0.13

# prorrogação = 30 min com taxa de gols proporcional à regulamentar
_FRACAO_PRORROGACAO = 30 / 90

# os 32-avos começam na tarde de 28/06/2026; meio-dia UTC separa com folga
# os últimos jogos de grupo, que caem na madrugada UTC de 28/06
_INICIO_MATA_MATA_UTC = datetime(2026, 6, 28, 12, 0, tzinfo=UTC)


def eh_mata_mata(kickoff_utc: datetime) -> bool:
    """Jogos a partir dos 32-avos contam o placar até o fim da prorrogação."""
    return kickoff_utc >= _INICIO_MATA_MATA_UTC


@dataclass
class PlacarPredito:
    """Um placar específico com sua probabilidade estimada."""

    gols_casa: int
    gols_fora: int
    probabilidade: float
    pontos_esperados: float = 0.0  # E[pontos] = 2*P(placar) + P(resultado)

    @property
    def label(self) -> str:
        return f"{self.gols_casa} x {self.gols_fora}"

    @property
    def resultado(self) -> str:
        if self.gols_casa > self.gols_fora:
            return "casa"
        if self.gols_casa < self.gols_fora:
            return "fora"
        return "empate"


@dataclass
class Predicao:
    """Resultado completo da predição de um jogo."""

    time_casa: str
    time_fora: str

    # Probabilidades de resultado (pós-combinação com Polymarket)
    prob_casa: float
    prob_empate: float
    prob_fora: float

    # Lambdas estimados (gols esperados por time)
    lambda_casa: float
    lambda_fora: float

    # Top placares ordenados por probabilidade
    placares: list[PlacarPredito] = field(default_factory=list)

    # Placar escolhido para a aposta (máximo E[pontos] sobre toda a grade)
    aposta: PlacarPredito | None = None

    # placar max-EV puro, preenchido só quando a Teoria dos Jogos desviou da
    # sugestão EV. None = TJ não ativada, ou TJ e EV concordam.
    aposta_ev: PlacarPredito | None = None

    @property
    def melhor_placar(self) -> PlacarPredito:
        # palpite da aposta: maximiza E[pontos], não só P(placar)
        return self.aposta or self.placares[0]

    @property
    def confianca(self) -> str:
        """Nível de confiança baseado na probabilidade do placar apostado."""
        p = self.melhor_placar.probabilidade
        if p >= 0.15:
            return "Alta"
        if p >= 0.10:
            return "Média"
        return "Baixa"


def _pontos_bolao(g_casa: int, g_fora: int, r_casa: int, r_fora: int) -> int:
    """Calcula os pontos no sistema do bolão: 3 cravada, 1 resultado."""
    if g_casa == r_casa and g_fora == r_fora:
        return 3
    if (g_casa > g_fora and r_casa > r_fora) or \
       (g_casa < g_fora and r_casa < r_fora) or \
       (g_casa == g_fora and r_casa == r_fora):
        return 1
    return 0


def predizer(
    time_casa: str,
    time_fora: str,
    prob_casa: float,
    prob_empate: float,
    prob_fora: float,
    lambda_total: float,
    top_n: int = 10,
    mata_mata: bool = False,
    palpites_oponentes: list[tuple[int, int]] | None = None,
) -> Predicao:
    """
    Calcula a distribuição de placares (Poisson independente + Dixon-Coles) e
    devolve a Predicao com ranking de placares, a aposta de maior E[pontos] e as
    probabilidades de resultado. Com `mata_mata`, converte a matriz de 90 min na
    do fim da prorrogação. Com `palpites_oponentes`, aplica a Teoria dos Jogos
    para maximizar o ganho relativo.
    """
    lambda_casa, lambda_fora = _estimar_lambdas(
        prob_casa, prob_empate, prob_fora, lambda_total
    )

    logger.debug(
        "Poisson | lambda_casa={lc:.3f} lambda_fora={lf:.3f} | mata_mata={mm} | {c} x {f}",
        lc=lambda_casa,
        lf=lambda_fora,
        mm=mata_mata,
        c=time_casa,
        f=time_fora,
    )

    matriz = _montar_matriz(lambda_casa, lambda_fora)
    if mata_mata:
        matriz = _aplicar_prorrogacao(matriz, lambda_casa, lambda_fora)

    p_casa_modelo, p_empate_modelo, p_fora_modelo = _probs_resultado(matriz)
    prob_por_resultado = {
        "casa": p_casa_modelo,
        "empate": p_empate_modelo,
        "fora": p_fora_modelo,
    }

    placares = _matriz_para_placares(matriz)

    # baseline max-EV calculado sempre, referência para detectar divergência com TJ
    aposta_ev_pick = max(
        placares,
        key=lambda p: 2 * p.probabilidade + prob_por_resultado[p.resultado],
    )

    if palpites_oponentes:
        logger.info("Aplicando Teoria dos Jogos usando {n} palpites de oponentes.", n=len(palpites_oponentes))
        # média de pontos dos oponentes por resultado possível: depende só de
        # (r_casa, r_fora), não do placar apostado, então calcula uma vez.
        media_ops_por_resultado: dict[tuple[int, int], float] = {}
        for r_casa in range(_MAX_GOLS + 1):
            for r_fora in range(_MAX_GOLS + 1):
                if float(matriz[r_casa, r_fora]) < 1e-6:
                    continue
                pontos_ops = [_pontos_bolao(op_c, op_f, r_casa, r_fora) for op_c, op_f in palpites_oponentes]
                media_ops_por_resultado[(r_casa, r_fora)] = sum(pontos_ops) / len(pontos_ops)

        for p in placares:
            p.pontos_esperados = sum(
                float(matriz[r_casa, r_fora]) * (_pontos_bolao(p.gols_casa, p.gols_fora, r_casa, r_fora) - media)
                for (r_casa, r_fora), media in media_ops_por_resultado.items()
            )
    else:
        for p in placares:
            p.pontos_esperados = 2 * p.probabilidade + prob_por_resultado[p.resultado]

    # aposta = máximo E[pontos relativos/absolutos] sobre toda a grade
    aposta = max(placares, key=lambda p: p.pontos_esperados)

    # aposta_ev só é relevante quando TJ está ativa e mudou a sugestão
    aposta_ev = (
        aposta_ev_pick
        if palpites_oponentes and aposta_ev_pick.label != aposta.label
        else None
    )

    placares.sort(key=lambda p: p.probabilidade, reverse=True)

    predicao = Predicao(
        time_casa=time_casa,
        time_fora=time_fora,
        prob_casa=prob_casa,
        prob_empate=prob_empate,
        prob_fora=prob_fora,
        lambda_casa=lambda_casa,
        lambda_fora=lambda_fora,
        placares=placares[:top_n],
        aposta=aposta,
        aposta_ev=aposta_ev,
    )

    logger.info(
        "Predição: {c} x {f} | aposta={p} ({pc:.1%}, E_val={e:.2f}) | confiança={conf}",
        c=time_casa,
        f=time_fora,
        p=aposta.label,
        pc=aposta.probabilidade,
        e=aposta.pontos_esperados,
        conf=predicao.confianca,
    )

    return predicao


def _montar_matriz(lambda_casa: float, lambda_fora: float) -> np.ndarray:
    """Matriz normalizada de probabilidades de placar com correção Dixon-Coles."""
    gols = np.arange(_MAX_GOLS + 1)
    p_casa = poisson.pmf(gols, lambda_casa)
    p_fora = poisson.pmf(gols, lambda_fora)
    matriz = np.outer(p_casa, p_fora)
    return _correcao_dixon_coles(matriz, lambda_casa, lambda_fora)


def _aplicar_prorrogacao(
    matriz: np.ndarray,
    lambda_casa: float,
    lambda_fora: float,
) -> np.ndarray:
    """
    Converte a matriz de 90 minutos na do placar ao fim dos 120 min.

    Cada empate kxk em 90' é redistribuído pela convolução com os gols da
    prorrogação (Poisson com lambda proporcional aos 30 min extras). Empates
    seguem possíveis: quem decide nos pênaltis termina empatado no placar, que é
    o que o bolão pontua.
    """
    lambda_extra_casa = lambda_casa * _FRACAO_PRORROGACAO
    lambda_extra_fora = lambda_fora * _FRACAO_PRORROGACAO

    gols = np.arange(_MAX_GOLS + 1)
    extra = np.outer(
        poisson.pmf(gols, lambda_extra_casa),
        poisson.pmf(gols, lambda_extra_fora),
    )
    extra /= extra.sum()  # devolve a cauda truncada (ínfima com lambda/3) à grade

    final = matriz.copy()
    for k in range(_MAX_GOLS + 1):
        p_empate_k = matriz[k, k]
        if p_empate_k <= 0:
            continue
        final[k, k] = 0.0
        for i in range(_MAX_GOLS + 1):
            for j in range(_MAX_GOLS + 1):
                # placares além da grade são comprimidos na borda (massa ~1e-8)
                final[min(k + i, _MAX_GOLS), min(k + j, _MAX_GOLS)] += p_empate_k * extra[i, j]

    final /= final.sum()
    return final


def _probs_resultado(matriz: np.ndarray) -> tuple[float, float, float]:
    """Probabilidades de (casa, empate, fora) implícitas na matriz de placares."""
    p_casa = float(np.tril(matriz, -1).sum())   # gols_casa > gols_fora
    p_empate = float(np.trace(matriz))
    p_fora = float(np.triu(matriz, 1).sum())    # gols_fora > gols_casa
    return p_casa, p_empate, p_fora


def _estimar_lambdas(
    prob_casa: float,
    prob_empate: float,
    prob_fora: float,
    lambda_total: float,
) -> tuple[float, float]:
    """Ajuste inverso: acha (lambda_casa, lambda_fora) que reproduzem o 1X2 do
    mercado, com âncora suave em lambda_total."""
    soma = prob_casa + prob_fora
    if soma == 0:
        chute = np.array([lambda_total / 2, lambda_total / 2])
    else:
        chute = np.array([
            max(lambda_total * prob_casa / soma, 0.1),
            max(lambda_total * prob_fora / soma, 0.1),
        ])

    ancora = max(lambda_total, 0.5)

    def _erro(params: np.ndarray) -> float:
        lc, lf = float(params[0]), float(params[1])
        pc, pe, pf = _probs_resultado(_montar_matriz(lc, lf))
        return (
            (pc - prob_casa) ** 2
            + (pe - prob_empate) ** 2
            + (pf - prob_fora) ** 2
            + 0.05 * ((lc + lf - lambda_total) / ancora) ** 2
        )

    res = minimize(
        _erro,
        chute,
        method="L-BFGS-B",
        bounds=[(0.05, 4.5), (0.05, 4.5)],
    )

    lc, lf = res.x if res.success else chute
    return float(max(lc, 0.05)), float(max(lf, 0.05))


def _correcao_dixon_coles(
    matriz: np.ndarray,
    lambda_casa: float,
    lambda_fora: float,
) -> np.ndarray:
    """Correção de Dixon-Coles para 0x0, 1x0, 0x1, 1x1, mais comuns do que o
    modelo independente prevê."""
    def tau(x: int, y: int) -> float:
        if x == 0 and y == 0:
            return 1 - lambda_casa * lambda_fora * _RHO
        if x == 1 and y == 0:
            return 1 + lambda_fora * _RHO
        if x == 0 and y == 1:
            return 1 + lambda_casa * _RHO
        if x == 1 and y == 1:
            return 1 - _RHO
        return 1.0

    corrigida = matriz.copy()
    for i in range(min(2, _MAX_GOLS + 1)):
        for j in range(min(2, _MAX_GOLS + 1)):
            corrigida[i, j] *= tau(i, j)

    corrigida /= corrigida.sum()
    return corrigida


def _matriz_para_placares(matriz: np.ndarray) -> list[PlacarPredito]:
    """Converte a matriz de probabilidades em lista de PlacarPredito."""
    placares = []
    for i in range(_MAX_GOLS + 1):
        for j in range(_MAX_GOLS + 1):
            placares.append(PlacarPredito(
                gols_casa=i,
                gols_fora=j,
                probabilidade=float(matriz[i, j]),
            ))
    return placares
