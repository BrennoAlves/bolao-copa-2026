"""
Elo de seleções, computado aqui a partir do histórico em ordem cronológica.

Computo em vez de baixar pronto porque o Elo publicado nos sites é o rating de
hoje, que já embute o resultado dos jogos que quero prever. Rodando a fórmula em
ordem e fotografando o rating de cada seleção antes do jogo-alvo, nada do futuro
vaza. Uso o World Football Elo (K varia com a importância do jogo, multiplicador
de saldo de gols). A conversão de delta de Elo para 1X2 usa um modelo de empate
calibrado só em jogos anteriores ao torneio que está sendo avaliado.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from loguru import logger
from scipy.optimize import curve_fit

from pesquisa.experimento.dados import COPA_DO_MUNDO, Partida

# Vantagem de mando de campo, em pontos de Elo (zero em campo neutro).
_VANTAGEM_MANDO = 100.0

# Rating inicial de toda seleção. Computar desde 1872 dilui o chute inicial
# muito antes de 2022, então o valor exato é irrelevante para a avaliação.
_RATING_INICIAL = 1500.0


def _fator_k(torneio: str) -> float:
    """
    Importância do jogo: magnitude do ajuste de rating.

    Aproxima as faixas do World Football Elo. Não precisa ser exata: erros de
    classificação se diluem no histórico longo e afetam todas as opções igual.
    """
    t = torneio.lower()
    if "world cup" in t and "qual" not in t:
        return 60.0  # fase final de Copa
    if any(
        marca in t
        for marca in ("qualif", "nations league", "euro", "copa am", "cup of nations",
                      "asian cup", "gold cup", "confederations")
    ):
        return 40.0  # eliminatórias e torneios continentais
    if "friendly" in t:
        return 20.0  # amistoso
    return 30.0  # demais competições


def _mult_saldo_gols(saldo: int) -> float:
    """Goleadas carregam mais informação: multiplicador cresce com o saldo."""
    s = abs(saldo)
    if s <= 1:
        return 1.0
    if s == 2:
        return 1.5
    if s == 3:
        return 1.75
    return 1.75 + (s - 3) / 8.0


def _esperado(dr: float) -> float:
    return float(1.0 / (1.0 + 10.0 ** (-dr / 400.0)))


class MotorElo:
    """Ratings atualizados jogo a jogo em ordem cronológica (único estado mutável aqui)."""

    def __init__(self, inicial: float = _RATING_INICIAL) -> None:
        self._ratings: dict[str, float] = {}
        self._inicial = inicial

    def rating(self, time: str) -> float:
        """Rating atual da seleção (default para quem ainda não jogou)."""
        return self._ratings.get(time, self._inicial)

    def dr_efetivo(self, partida: Partida) -> float:
        """delta de Elo casa menos fora, já com a vantagem de mando (zero se neutro)."""
        vantagem = 0.0 if partida.neutro else _VANTAGEM_MANDO
        return self.rating(partida.time_casa) - self.rating(partida.time_fora) + vantagem

    def processar(self, partida: Partida) -> None:
        """Atualiza os ratings dos dois times com o resultado de um jogo disputado."""
        if partida.gols_casa is None or partida.gols_fora is None:
            return
        w_casa = {"C": 1.0, "E": 0.5, "F": 0.0}[partida.resultado]
        k = _fator_k(partida.torneio) * _mult_saldo_gols(partida.gols_casa - partida.gols_fora)
        delta = k * (w_casa - _esperado(self.dr_efetivo(partida)))
        self._ratings[partida.time_casa] = self.rating(partida.time_casa) + delta
        self._ratings[partida.time_fora] = self.rating(partida.time_fora) - delta


@dataclass(frozen=True)
class ParametrosEmpate:
    """Modelo do empate em função do delta de Elo: pe(dr) = pe_max * exp(-(dr/tau)^2)."""

    pe_max: float  # prob de empate quando os times são equivalentes (dr=0)
    tau: float     # escala: quanto maior, mais devagar o empate decai com abs(dr)


def _curva_empate(dr: np.ndarray, pe_max: float, tau: float) -> np.ndarray:
    return pe_max * np.exp(-((dr / tau) ** 2))


def prob_1x2_elo(dr_efetivo: float, params: ParametrosEmpate) -> tuple[float, float, float]:
    """Converte o delta de Elo em (P(casa), P(empate), P(fora)).

    Mantém P(casa) + 0.5*P(empate) na expectativa de pontos do Elo, modela o empate
    à parte e divide o resto em vitória/derrota. Renormaliza no fim.
    """
    we = _esperado(dr_efetivo)
    pe = params.pe_max * math.exp(-((dr_efetivo / params.tau) ** 2))
    p_casa = max(we - 0.5 * pe, 1e-6)
    p_fora = max((1.0 - we) - 0.5 * pe, 1e-6)
    pe = max(pe, 1e-6)
    total = p_casa + pe + p_fora
    return p_casa / total, pe / total, p_fora / total


@dataclass(frozen=True)
class SnapshotJogo:
    """delta de Elo de um jogo-alvo, fotografado ANTES de o resultado ocorrer."""

    partida: Partida
    dr_efetivo: float


@dataclass(frozen=True)
class PreparoElo:
    """Saída do passo cronológico: dados de calibração + snapshots da Copa-alvo."""

    snapshots: list[SnapshotJogo]
    params_empate: ParametrosEmpate


def preparar_elo(partidas: list[Partida], ano_copa: int = 2022) -> PreparoElo:
    """
    Passa pelo histórico uma vez, em ordem, e devolve sem vazamento:
    - o delta de Elo de cada jogo da Copa-alvo, fotografado antes do jogo;
    - os parâmetros do empate calibrados só com jogos anteriores a essa Copa.

    A separação no tempo é o que impede o Elo e a calibração de enxergar os
    resultados que vão ser avaliados.
    """
    disputadas = [p for p in partidas if p.disputada]
    jogos_alvo = [
        p for p in disputadas if p.torneio == COPA_DO_MUNDO and p.data.year == ano_copa
    ]
    if not jogos_alvo:
        raise ValueError(f"Nenhum jogo disputado da Copa {ano_copa} no dataset")
    corte = min(p.data for p in jogos_alvo)

    motor = MotorElo()
    drs_cal: list[float] = []
    empates_cal: list[float] = []
    snapshots: list[SnapshotJogo] = []
    ids_alvo = {id(p) for p in jogos_alvo}

    for p in disputadas:
        dr = motor.dr_efetivo(p)
        if id(p) in ids_alvo:
            snapshots.append(SnapshotJogo(partida=p, dr_efetivo=dr))
        elif p.data < corte:
            drs_cal.append(dr)
            empates_cal.append(1.0 if p.resultado == "E" else 0.0)
        motor.processar(p)

    params = _calibrar_empate(np.array(drs_cal), np.array(empates_cal))
    logger.info(
        "Elo preparado | {n} jogos da Copa {a} | empate calibrado: pe_max={p:.3f} tau={t:.0f} "
        "(em {c} jogos pré-corte)",
        n=len(snapshots),
        a=ano_copa,
        p=params.pe_max,
        t=params.tau,
        c=len(drs_cal),
    )
    return PreparoElo(snapshots=snapshots, params_empate=params)


def _calibrar_empate(drs: np.ndarray, empates: np.ndarray) -> ParametrosEmpate:
    """
    Ajusta pe_max e tau à frequência empírica de empates vs abs(delta de Elo) por
    mínimos quadrados não-lineares. Fallback para defaults se não convergir.
    """
    try:
        (pe_max, tau), _ = curve_fit(
            _curva_empate,
            drs,
            empates,
            p0=[0.27, 300.0],
            bounds=([0.05, 50.0], [0.6, 2000.0]),
            maxfev=10_000,
        )
        return ParametrosEmpate(pe_max=float(pe_max), tau=float(abs(tau)))
    except (RuntimeError, ValueError) as e:
        logger.warning("Calibração de empate não convergiu ({e}); usando defaults", e=e)
        return ParametrosEmpate(pe_max=0.27, tau=300.0)
