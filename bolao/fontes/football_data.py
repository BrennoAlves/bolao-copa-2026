"""Placar via football-data.org quando a The Odds API não devolve o resultado.

Opcional: sem FOOTBALL_DATA_TOKEN nada roda. Casa os times por similaridade.
"""
from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import httpx
from loguru import logger

from bolao.nomes import casar_nome

_BASE = "https://api.football-data.org/v4"
_COMPETICAO = os.getenv("FOOTBALL_DATA_COMP", "WC")  # código da competição no football-data.org


def buscar_placar(
    token: str,
    time_casa: str,
    time_fora: str,
    dias_atras: int = 3,
) -> tuple[int, int] | None:
    """Placar final de um jogo recente do torneio, casando os times por similaridade.

    Retorna (gols_casa, gols_fora) na orientação (time_casa, time_fora), ou None
    se não achar o jogo encerrado na janela. Falhas de rede/HTTP retornam None
    (é fonte secundária, não deve derrubar o fluxo).
    """
    if not token.strip():
        return None
    hoje = datetime.now(UTC).date()
    try:
        resp = httpx.get(
            f"{_BASE}/competitions/{_COMPETICAO}/matches",
            headers={"X-Auth-Token": token},
            params={
                "status": "FINISHED",
                "dateFrom": (hoje - timedelta(days=dias_atras)).isoformat(),
                "dateTo": hoje.isoformat(),
            },
            timeout=10.0,
        )
        resp.raise_for_status()
        jogos = resp.json().get("matches", [])
    except (httpx.HTTPError, ValueError) as e:
        logger.warning("football-data indisponível: {e}", e=e)
        return None

    return _casar_placar(jogos, time_casa, time_fora)


def _casar_placar(
    jogos: list[dict],
    time_casa: str,
    time_fora: str,
) -> tuple[int, int] | None:
    """Encontra o jogo cujos times casam com (time_casa, time_fora) e extrai o placar."""
    nomes_casa = [j.get("homeTeam", {}).get("name", "") for j in jogos]
    alvo_casa = casar_nome(time_casa, nomes_casa)
    if alvo_casa is None:
        logger.debug("football-data: '{c}' não casou com nenhum mandante", c=time_casa)
        return None

    for jogo in jogos:
        if jogo.get("homeTeam", {}).get("name", "") != alvo_casa:
            continue
        if casar_nome(time_fora, [jogo.get("awayTeam", {}).get("name", "")]) is None:
            continue
        full = jogo.get("score", {}).get("fullTime", {})
        gols_casa, gols_fora = full.get("home"), full.get("away")
        if gols_casa is None or gols_fora is None:
            return None
        return int(gols_casa), int(gols_fora)
    return None
