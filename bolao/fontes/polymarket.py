"""Probabilidades do Polymarket. Ele só tem mercado de campeão por seleção
(ex.: "Will Brazil win the World Cup?"), não de resultado por jogo. Usamos as
odds de campeão para estimar a força relativa entre os dois times e ajustar de
leve as probabilidades da The Odds API."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, cast

import httpx
from loguru import logger

from bolao.cache import cache_get, cache_set

_BASE = "https://gamma-api.polymarket.com"
_CACHE_TTL = 600  # 10 minutos

# mercado de campeão do torneio no Polymarket. slug e frase variam por edição,
# então saem do código para o .env (com o default da edição atual).
_EVENTO_SLUG = os.getenv("POLYMARKET_SLUG", "world-cup-winner")
_PERGUNTA_CAMPEAO = os.getenv("POLYMARKET_CAMPEAO", "win the 2026 fifa world cup")


def _parse_json_field(valor: str | list[Any] | None) -> list[Any]:
    """Campos como outcomes/outcomePrices vêm como string JSON do Gamma."""
    if isinstance(valor, str):
        return cast(list[Any], json.loads(valor))
    return valor or []


@dataclass
class ProbPolymarket:
    """Probabilidades de um jogo extraídas do Polymarket via força relativa dos times."""

    prob_casa: float
    prob_empate: float | None
    prob_fora: float
    liquidez_usd: float


# nomes da The Odds API para palavras-chave no Polymarket
_NOME_PARA_POLY: dict[str, str] = {
    "Brazil": "brazil",
    "Argentina": "argentina",
    "France": "france",
    "Spain": "spain",
    "England": "england",
    "Germany": "germany",
    "Portugal": "portugal",
    "Netherlands": "netherlands",
    "Belgium": "belgium",
    "Uruguay": "uruguay",
    "Mexico": "mexico",
    "USA": "united states",
    "Canada": "canada",
    "South Korea": "south korea",
    "Japan": "japan",
    "Australia": "australia",
    "Morocco": "morocco",
    "Senegal": "senegal",
    "Colombia": "colombia",
    "Ecuador": "ecuador",
    "Switzerland": "switzerland",
    "Turkey": "turkey",
    "Sweden": "sweden",
    "Norway": "norway",
    "Austria": "austria",
    "Croatia": "croatia",
    "Czech Republic": "czech republic",
    "Saudi Arabia": "saudi arabia",
    "Iran": "iran",
    "South Africa": "south africa",
    "Ghana": "ghana",
    "Ivory Coast": "ivory coast",
    "Cameroon": "cameroon",
}


def buscar_probabilidades(time_casa: str, time_fora: str) -> ProbPolymarket | None:
    """Estima força relativa entre dois times via odds de campeão do Polymarket.
    Retorna None se nenhum dos times tiver mercado ativo.
    """
    chave_cache = f"polymarket:{time_casa}:{time_fora}".lower()
    dados = cache_get(chave_cache)
    if dados is not None:
        return cast("ProbPolymarket", dados)

    mercados_campeao = _buscar_mercados_campeao()
    resultado = _calcular_forca_relativa(mercados_campeao, time_casa, time_fora)

    if resultado:
        cache_set(chave_cache, resultado, ttl=_CACHE_TTL)

    return resultado


def _buscar_mercados_campeao() -> dict[str, float]:
    """
    Retorna dicionário {nome_time_lower: prob_campeao} para todos os times
    com mercado ativo no evento de campeão do Polymarket.
    """
    chave_cache = "polymarket:campeao"
    cached = cache_get(chave_cache)
    if cached is not None:
        return cast(dict[str, float], cached)

    try:
        # busca direta pelo slug (~200 KB) em vez de listar os 100 eventos de
        # maior volume (~9 MB, estourava o timeout de leitura)
        resp = httpx.get(
            f"{_BASE}/events",
            params={"slug": _EVENTO_SLUG},
            timeout=10.0,
        )
        resp.raise_for_status()
        eventos = resp.json()

        if not eventos:
            logger.debug("Polymarket: evento '{s}' não encontrado", s=_EVENTO_SLUG)
            return {}

        mercados: dict[str, float] = {}
        for m in eventos[0].get("markets", []):
            pergunta = m.get("question", "").lower()

            if _PERGUNTA_CAMPEAO not in pergunta:
                continue

            # O Gamma retorna outcomes/outcomePrices como strings JSON serializadas
            try:
                outcomes = _parse_json_field(m.get("outcomes"))
                prices_raw = _parse_json_field(m.get("outcomePrices"))
                if len(outcomes) < 2 or len(prices_raw) < 2:
                    continue

                prob_yes = float(prices_raw[0])
                team = (
                    pergunta
                    .replace("will ", "")
                    .replace(f" {_PERGUNTA_CAMPEAO}?", "")
                    .strip()
                )
                mercados[team] = prob_yes
            except (ValueError, IndexError, TypeError):
                continue

        logger.debug("Polymarket: {n} times com odds de campeão", n=len(mercados))
        cache_set(chave_cache, mercados, ttl=_CACHE_TTL)
        return mercados

    except Exception as e:
        logger.warning("Polymarket indisponível: {e}", e=e)
        return {}


def _calcular_forca_relativa(
    mercados: dict[str, float],
    time_casa: str,
    time_fora: str,
) -> ProbPolymarket | None:
    """Usa as probabilidades de campeão para estimar força relativa entre dois times.

    Lógica: P(casa vence jogo) proporcional a P(casa campeão) / P(fora campeão).
    É uma aproximação, serve como sinal complementar às odds de jogo.
    """
    if not mercados:
        return None

    def _achar_prob(nome: str) -> float | None:
        chave_poly = _NOME_PARA_POLY.get(nome, nome.lower())
        if chave_poly in mercados:
            return mercados[chave_poly]
        for team, prob in mercados.items():
            if nome.lower() in team or team in nome.lower():
                return prob
        return None

    p_casa = _achar_prob(time_casa)
    p_fora = _achar_prob(time_fora)

    if p_casa is None or p_fora is None:
        logger.debug(
            "Polymarket: sem odds de campeão para '{c}' ou '{f}'",
            c=time_casa,
            f=time_fora,
        )
        return None

    soma = p_casa + p_fora
    if soma == 0:
        return None

    ratio_casa = p_casa / soma
    ratio_fora = p_fora / soma

    # não inventamos prob de empate: prob_empate=None faz o blend ancorar o
    # empate nas odds e usar o Polymarket só na razão casa/fora.
    logger.info(
        "Polymarket ok | {c}({pc:.3f}) x {f}({pf:.3f}) | ratio casa={rc:.1%} fora={rf:.1%}",
        c=time_casa,
        pc=p_casa,
        f=time_fora,
        pf=p_fora,
        rc=ratio_casa,
        rf=ratio_fora,
    )

    return ProbPolymarket(
        prob_casa=ratio_casa,   # razão casa/fora (2 vias, soma 1 com prob_fora)
        prob_empate=None,
        prob_fora=ratio_fora,
        liquidez_usd=100_000.0,  # mercado de campeão tem $2B+ em volume
    )


# peso default do Polymarket no blend: priors de força agregam pouco sobre o mercado.
_PESO_POLY_DEFAULT = 0.25


def _peso_poly() -> float:
    """Peso do Polymarket, sobrescrevível por PESO_POLYMARKET no .env."""
    try:
        return float(os.getenv("PESO_POLYMARKET", str(_PESO_POLY_DEFAULT)))
    except ValueError:
        return _PESO_POLY_DEFAULT


def ajustar_probabilidades(
    prob_casa: float,
    prob_empate: float,
    prob_fora: float,
    poly: ProbPolymarket | None,
    peso_poly: float | None = None,
) -> tuple[float, float, float]:
    """Ajusta o 1X2 das odds com o sinal de força do Polymarket.

    O empate fica ANCORADO nas odds (mercado por jogo); o Polymarket só desloca a
    razão casa/fora, com peso modesto.
    """
    if poly is None or poly.liquidez_usd < 5_000:
        return prob_casa, prob_empate, prob_fora

    peso = _peso_poly() if peso_poly is None else peso_poly
    peso_odds = 1 - peso

    # poly.prob_casa/fora são razões (2 vias); reescala para a massa fora do
    # empate das ODDS, mantendo o empate do mercado intacto.
    escala = 1 - prob_empate
    poly_casa = poly.prob_casa * escala
    poly_fora = poly.prob_fora * escala

    p_casa = peso_odds * prob_casa + peso * poly_casa
    p_fora = peso_odds * prob_fora + peso * poly_fora
    p_empate = prob_empate  # ancorado nas odds

    total = p_casa + p_empate + p_fora
    return p_casa / total, p_empate / total, p_fora / total
