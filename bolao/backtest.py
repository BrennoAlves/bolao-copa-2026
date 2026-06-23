"""Backtesting para medir a qualidade do modelo ao longo do torneio.

Guarda o snapshot da predição no momento da aposta e preenche o placar real
quando o jogo encerra, o que permite comparar variantes do modelo ("só The Odds
API" vs "blend com Polymarket") por RPS, Brier e pontos ganhos.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger

from bolao.cache import cache_chaves, cache_get, cache_set

_CACHE_KEY_PREFIX = "backtest"
_TTL_BACKTEST = 60 * 24 * 3600  # 60 dias, cobre todo o torneio com margem


@dataclass
class RegistroBacktest:
    """Snapshot de predição + resultado de um único jogo."""

    jogo_id: str
    time_casa: str
    time_fora: str
    kickoff_utc: datetime

    # probabilidades brutas da The Odds API (consenso + remoção da margem por potência)
    prob_casa_api: float
    prob_empate_api: float
    prob_fora_api: float

    # probabilidades usadas no modelo (após blend com Polymarket)
    prob_casa_modelo: float
    prob_empate_modelo: float
    prob_fora_modelo: float

    # lambdas de Poisson ajustados às probs do modelo
    lambda_casa: float
    lambda_fora: float

    # placar apostado e métricas da aposta
    gols_apostados_casa: int
    gols_apostados_fora: int
    prob_apostada: float    # P(placar apostado), base do Brier
    pontos_esperados: float  # E[pontos] no momento da aposta

    # top-10 placares como (label, prob) para lookup pós-jogo
    top10_placares: list[tuple[str, float]] = field(default_factory=list)

    # preenchido após o jogo encerrar
    gols_reais_casa: int | None = None
    gols_reais_fora: int | None = None
    pontos_ganhos: int | None = None

    @property
    def completo(self) -> bool:
        """True quando o placar real já foi registrado."""
        return self.gols_reais_casa is not None

    @property
    def label_apostado(self) -> str:
        return f"{self.gols_apostados_casa} x {self.gols_apostados_fora}"

    @property
    def label_real(self) -> str | None:
        if self.gols_reais_casa is None or self.gols_reais_fora is None:
            return None
        return f"{self.gols_reais_casa} x {self.gols_reais_fora}"


def registrar_predicao_backtest(
    jogo_id: str,
    time_casa: str,
    time_fora: str,
    kickoff_utc: datetime,
    prob_casa_api: float,
    prob_empate_api: float,
    prob_fora_api: float,
    prob_casa_modelo: float,
    prob_empate_modelo: float,
    prob_fora_modelo: float,
    lambda_casa: float,
    lambda_fora: float,
    gols_apostados_casa: int,
    gols_apostados_fora: int,
    prob_apostada: float,
    pontos_esperados: float,
    top10_placares: list[tuple[str, float]],
) -> None:
    """Persiste (ou sobrescreve) o snapshot da predição no momento da aposta.

    Chamado tanto na aposta inicial (T-60) quanto nos refinamentos que mudam o
    palpite, o registro sempre reflete a predição mais recente enviada.
    """
    registro = RegistroBacktest(
        jogo_id=jogo_id,
        time_casa=time_casa,
        time_fora=time_fora,
        kickoff_utc=kickoff_utc,
        prob_casa_api=prob_casa_api,
        prob_empate_api=prob_empate_api,
        prob_fora_api=prob_fora_api,
        prob_casa_modelo=prob_casa_modelo,
        prob_empate_modelo=prob_empate_modelo,
        prob_fora_modelo=prob_fora_modelo,
        lambda_casa=lambda_casa,
        lambda_fora=lambda_fora,
        gols_apostados_casa=gols_apostados_casa,
        gols_apostados_fora=gols_apostados_fora,
        prob_apostada=prob_apostada,
        pontos_esperados=pontos_esperados,
        top10_placares=top10_placares,
    )
    cache_set(f"{_CACHE_KEY_PREFIX}:{jogo_id}", registro, ttl=_TTL_BACKTEST)
    logger.info(
        "Backtest gravado | {c} x {f} | aposta={gc}x{gf} | P={p:.1%}",
        c=time_casa,
        f=time_fora,
        gc=gols_apostados_casa,
        gf=gols_apostados_fora,
        p=prob_apostada,
    )


def registrar_resultado_backtest(
    jogo_id: str,
    gols_reais_casa: int,
    gols_reais_fora: int,
    pontos_ganhos: int,
) -> None:
    """Preenche o placar real e os pontos ganhos no registro existente.

    Se o registro não existir (jogo apostado antes do backtest ser implantado),
    loga aviso e retorna sem erro para não afetar o fluxo principal.
    """
    registro: RegistroBacktest | None = cache_get(f"{_CACHE_KEY_PREFIX}:{jogo_id}")
    if registro is None:
        logger.warning(
            "Backtest: registro de {id} não encontrado, resultado não gravado", id=jogo_id
        )
        return

    registro.gols_reais_casa = gols_reais_casa
    registro.gols_reais_fora = gols_reais_fora
    registro.pontos_ganhos = pontos_ganhos
    cache_set(f"{_CACHE_KEY_PREFIX}:{jogo_id}", registro, ttl=_TTL_BACKTEST)
    logger.info(
        "Backtest atualizado | {c} x {f} | real={rc}x{rf} | pts={p}",
        c=registro.time_casa,
        f=registro.time_fora,
        rc=gols_reais_casa,
        rf=gols_reais_fora,
        p=pontos_ganhos,
    )


def listar_registros_backtest() -> list[RegistroBacktest]:
    """Retorna todos os registros de backtest ordenados por kickoff."""
    registros = []
    for chave in cache_chaves(f"{_CACHE_KEY_PREFIX}:"):
        r = cache_get(chave)
        if isinstance(r, RegistroBacktest):
            registros.append(r)
    registros.sort(key=lambda r: r.kickoff_utc)
    return registros


def calcular_rps(
    prob_casa: float,
    prob_empate: float,
    prob_fora: float,
    gols_reais_casa: int,
    gols_reais_fora: int,
) -> float:
    """Ranked Probability Score para 3 categorias ordenadas: [casa, empate, fora].
    Fica em [0, 1], menor é melhor.
    """
    if gols_reais_casa > gols_reais_fora:
        cumul_o = [1.0, 1.0]
    elif gols_reais_casa == gols_reais_fora:
        cumul_o = [0.0, 1.0]
    else:
        cumul_o = [0.0, 0.0]

    cumul_f = [prob_casa, prob_casa + prob_empate]
    return 0.5 * sum((f - o) ** 2 for f, o in zip(cumul_f, cumul_o, strict=False))


def calcular_metricas_backtest(
    registros: list[RegistroBacktest],
) -> dict[str, float | int]:
    """Agrega métricas (pontos, cravadas, RPS, Brier) de todos os registros completos.

    O RPS é calculado em duas variantes: com as probs brutas da API e com as
    probs após o blend com o Polymarket, para comparar as duas fontes.
    """
    completos = [r for r in registros if r.completo]
    n = len(completos)
    if n == 0:
        return {"n_jogos": 0}

    pts_total = sum(r.pontos_ganhos or 0 for r in completos)
    pts_esperados = sum(r.pontos_esperados for r in completos)
    n_cravadas = sum(1 for r in completos if r.pontos_ganhos == 3)
    n_resultados = sum(1 for r in completos if r.pontos_ganhos == 1)
    n_erros = sum(1 for r in completos if r.pontos_ganhos == 0)

    rps_api_sum = 0.0
    rps_mod_sum = 0.0
    brier_sum = 0.0

    for r in completos:
        rc = r.gols_reais_casa
        rf = r.gols_reais_fora
        assert rc is not None and rf is not None  # garantido pelo filtro completos

        rps_api_sum += calcular_rps(
            r.prob_casa_api, r.prob_empate_api, r.prob_fora_api, rc, rf
        )
        rps_mod_sum += calcular_rps(
            r.prob_casa_modelo, r.prob_empate_modelo, r.prob_fora_modelo, rc, rf
        )

        acertou = rc == r.gols_apostados_casa and rf == r.gols_apostados_fora
        brier_sum += (r.prob_apostada - float(acertou)) ** 2

    return {
        "n_jogos": n,
        "pts_total": pts_total,
        "pts_esperados": round(pts_esperados, 2),
        "n_cravadas": n_cravadas,
        "n_resultados": n_resultados,
        "n_erros": n_erros,
        "rps_medio_api": round(rps_api_sum / n, 4),
        "rps_medio_modelo": round(rps_mod_sum / n, 4),
        "brier_medio": round(brier_sum / n, 4),
    }
