"""Odds em tempo real do torneio via The Odds API (1X2 e Over/Under), com a
margem removida pelo método power. Probabilidades são a mediana entre books."""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx
from loguru import logger
from scipy.optimize import brentq
from scipy.stats import poisson

from bolao.cache import cache_get, cache_set
from bolao.config import ODDS_SPORT as _SPORT

_BASE = "https://api.the-odds-api.com/v4"

# ttl do cache de agendamento: 7 horas, acima do re-scan de 6h. os checkpoints
# de aposta (T-60 a T-5) usam ignorar_cache=True e sempre buscam odds frescas;
# este ttl afeta só _agendar_jogos, que só precisa descobrir novos jogos.
_CACHE_TTL = 7 * 3600

# créditos restantes (header x-requests-remaining). o scheduler lê para alertar
# quando a chave gratuita estiver acabando: você cria outra e troca ODDS_API_KEY.
CHAVE_CREDITOS = "odds_api:creditos_restantes"


@dataclass
class OddsJogo:
    """Odds consolidadas de um jogo para alimentar o modelo de Poisson."""

    id: str
    time_casa: str
    time_fora: str
    kickoff_utc: datetime

    # probabilidades implícitas (sem margem da casa) em [0, 1]
    prob_casa: float
    prob_empate: float
    prob_fora: float

    # média de gols esperados (mercado Over/Under convertido)
    lambda_total: float

    # True quando não havia mercado de totals e lambda caiu no default (2.5).
    # propagado para a notificação avisar que o volume de gols é um chute genérico.
    lambda_estimado: bool = False


def buscar_odds(api_key: str, ignorar_cache: bool = False) -> list[OddsJogo]:
    """Odds de todos os jogos futuros do torneio, com cache em disco.

    ignorar_cache força requisição fresca, usado em T-60min, quando as odds já
    refletem as escalações divulgadas.
    """
    chave_cache = f"odds_api:{_SPORT}"
    dados = None if ignorar_cache else cache_get(chave_cache)

    if dados is None:
        dados = _buscar_da_api(api_key)
        if dados:
            cache_set(chave_cache, dados, ttl=_CACHE_TTL)

    return dados or []


def _buscar_da_api(api_key: str) -> list[OddsJogo]:
    params = {
        "apiKey": api_key,
        "regions": "eu",           # mercado mais líquido
        "markets": "h2h,totals",   # resultado + over/under
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }

    try:
        resp = httpx.get(
            f"{_BASE}/sports/{_SPORT}/odds",
            params=params,
            timeout=15.0,
        )
        resp.raise_for_status()

        _registrar_creditos(resp.headers.get("x-requests-remaining"))

        dados = resp.json()
        logger.info(
            "The Odds API: {n} jogos recebidos | créditos restantes: {c}",
            n=len(dados),
            c=resp.headers.get("x-requests-remaining", "?"),
        )
        return _parsear(dados)

    except httpx.HTTPStatusError as e:
        logger.error("Erro HTTP na The Odds API: {s} ({b})", s=e.response.status_code, b=e.response.text)
        return []
    except httpx.RequestError as e:
        logger.error("Erro de rede na The Odds API: {e}", e=e)
        return []


def _registrar_creditos(restantes_header: str | None) -> None:
    """Persiste os créditos restantes para o scheduler alertar quando acabar."""
    if restantes_header is None:
        return
    try:
        cache_set(CHAVE_CREDITOS, int(float(restantes_header)), ttl=7 * 24 * 3600)
    except ValueError:
        logger.debug("Header x-requests-remaining ilegível: {h}", h=restantes_header)


def _parsear(jogos_raw: list[dict]) -> list[OddsJogo]:
    resultado = []

    for jogo in jogos_raw:
        try:
            kickoff = datetime.fromisoformat(jogo["commence_time"].replace("Z", "+00:00"))

            # ignora jogos que já começaram
            if kickoff <= datetime.now(UTC):
                continue

            h2h_books, totals_books = _extrair_mercados(jogo.get("bookmakers", []))
            if not h2h_books:
                continue

            prob_casa, prob_empate, prob_fora = _consenso_h2h(
                h2h_books, jogo["home_team"], jogo["away_team"]
            )
            lambda_total, lambda_estimado = _consenso_lambda_com_flag(totals_books)

            resultado.append(OddsJogo(
                id=jogo["id"],
                time_casa=jogo["home_team"],
                time_fora=jogo["away_team"],
                kickoff_utc=kickoff,
                prob_casa=prob_casa,
                prob_empate=prob_empate,
                prob_fora=prob_fora,
                lambda_total=lambda_total,
                lambda_estimado=lambda_estimado,
            ))

        except (KeyError, ValueError, TypeError) as e:
            logger.warning("Erro ao parsear jogo {id}: {e}", id=jogo.get("id", "?"), e=e)

    logger.debug("{n} jogos futuros parseados com sucesso", n=len(resultado))
    return resultado


def _extrair_mercados(
    bookmakers: list[dict],
) -> tuple[list[dict[str, float]], list[tuple[float, float, float]]]:
    """
    Extrai os mercados h2h e totals de todos os bookmakers.

    O consenso entre books reduz o ruído de uma linha defasada ou margem atípica
    de um book só.
    """
    h2h_books: list[dict[str, float]] = []
    totals_books: list[tuple[float, float, float]] = []  # (linha, odd_over, odd_under)

    for bm in bookmakers:
        for market in bm.get("markets", []):
            if market["key"] == "h2h":
                h2h_books.append({o["name"]: float(o["price"]) for o in market["outcomes"]})
            elif market["key"] == "totals":
                par = _parear_over_under(market["outcomes"])
                if par is not None:
                    totals_books.append(par)

    return h2h_books, totals_books


def _parear_over_under(outcomes: list[dict]) -> tuple[float, float, float] | None:
    """Pareia Over e Under da mesma linha. Sem o par não dá para tirar a margem."""
    overs = {o.get("point"): o["price"] for o in outcomes if o["name"] == "Over"}
    unders = {o.get("point"): o["price"] for o in outcomes if o["name"] == "Under"}

    for linha, odd_over in overs.items():
        if linha is not None and linha in unders:
            return float(linha), float(odd_over), float(unders[linha])
    return None


def _consenso_h2h(
    h2h_books: list[dict[str, float]],
    time_casa: str,
    time_fora: str,
) -> tuple[float, float, float]:
    """
    Mediana das probabilidades sem margem entre os bookmakers.

    O mapeamento casa/fora é feito pelo nome do time: a ordem dos outcomes na
    resposta da API não é garantida.
    """
    probs_casa: list[float] = []
    probs_empate: list[float] = []
    probs_fora: list[float] = []

    for h2h in h2h_books:
        draw_key = next((k for k in h2h if "draw" in k.lower()), None)
        if draw_key is None or time_casa not in h2h or time_fora not in h2h:
            continue

        p_casa, p_empate, p_fora = _remover_margem_power(
            [h2h[time_casa], h2h[draw_key], h2h[time_fora]]
        )
        probs_casa.append(p_casa)
        probs_empate.append(p_empate)
        probs_fora.append(p_fora)

    if not probs_casa:
        raise ValueError(f"Nenhum bookmaker com H2H completo: {time_casa} x {time_fora}")

    # medianas por desfecho não somam exatamente 1, então renormaliza no final
    p = [
        statistics.median(probs_casa),
        statistics.median(probs_empate),
        statistics.median(probs_fora),
    ]
    total = sum(p)
    return p[0] / total, p[1] / total, p[2] / total


def _remover_margem_power(odds: list[float]) -> list[float]:
    """Tira a margem pelo método power: acha k tal que soma de (1/odd_i)^k = 1."""
    brutas = [1 / o for o in odds]
    soma = sum(brutas)

    # sem margem ou odds degeneradas (<= 1.0): proporcional já basta
    if soma <= 1.0 or any(p >= 1.0 for p in brutas):
        return [p / soma for p in brutas]

    def _excesso(k: float) -> float:
        return sum(math.pow(p, k) for p in brutas) - 1.0

    try:
        k = float(brentq(_excesso, 1.0, 10.0))
    except ValueError:
        # margem tão grande que nem k=10 resolve: fallback proporcional
        return [p / soma for p in brutas]

    ajustadas = [math.pow(p, k) for p in brutas]
    total = sum(ajustadas)
    return [p / total for p in ajustadas]


def _consenso_lambda_com_flag(
    totals_books: list[tuple[float, float, float]],
) -> tuple[float, bool]:
    """Mediana do lambda implícito no Over/Under; flag indica se caiu no default.

    Retorna (lambda, estimado): estimado=True quando nenhum par Over/Under válido
    existia e usamos a média histórica (2.5), sinal de que o volume de gols é um
    chute genérico, não derivado do mercado.
    """
    lambdas = []
    for linha, odd_over, odd_under in totals_books:
        lam = _total_para_lambda(linha, odd_over, odd_under)
        if lam is not None:
            lambdas.append(lam)
    if lambdas:
        return statistics.median(lambdas), False
    return 2.5, True  # média histórica de gols por jogo em torneios de seleções


def _total_para_lambda(linha: float, odd_over: float, odd_under: float) -> float | None:
    """Converte o par Over/Under em lambda de gols totais.

    Normaliza o par antes (1/odd_over cru deixa a margem embutida e infla lambda);
    depois resolve P(G > linha) = prob_over para G Poisson.
    """
    p_over_bruta = 1 / odd_over
    p_under_bruta = 1 / odd_under
    prob_over = p_over_bruta / (p_over_bruta + p_under_bruta)
    prob_over = min(max(prob_over, 0.02), 0.98)

    # para linha L, "over" exige G >= floor(L)+1 (linhas .5 não têm push)
    k = math.floor(linha)

    def _f(lam: float) -> float:
        return 1 - float(poisson.cdf(k, lam)) - prob_over

    try:
        return float(brentq(_f, 0.05, 12.0))
    except ValueError:
        logger.warning("lambda não convergiu para linha {l} (over={o})", l=linha, o=odd_over)
        return None


def buscar_jogo_por_times(
    api_key: str,
    time_casa: str,
    time_fora: str,
) -> OddsJogo | None:
    """Odds de um jogo específico pelo nome dos times (parcial, case-insensitive)."""
    jogos = buscar_odds(api_key)
    casa_lower = time_casa.lower()
    fora_lower = time_fora.lower()

    for jogo in jogos:
        if (
            casa_lower in jogo.time_casa.lower()
            and fora_lower in jogo.time_fora.lower()
        ):
            return jogo

    logger.warning(
        "Jogo '{c}' x '{f}' não encontrado na The Odds API",
        c=time_casa,
        f=time_fora,
    )
    return None
