"""Monta o mesmo `DadosBolao` que o scraping do Playwright, mas via HTTP, sem
abrir o navegador. É o caminho de baixo custo de CPU para os jobs de leitura.

Fontes: standings via /api/leaderboard, picks via /api/game-guesses (ambos
Next.js, auth por cookie) e jogos/times/placar via PostgREST (supabase.py). Usa
as mesmas dataclasses e chave de cache do raspar_bolao, então os consumidores
(perfis, notificações, estratégia) não notam diferença.
"""
from __future__ import annotations

from datetime import UTC, datetime

import httpx
from loguru import logger

from bolao.cache import cache_set
from bolao.config import URL_BASE as _BASE
from bolao.config import Config
from bolao.fontes.sessao_http import carregar_cookies_sessao
from bolao.fontes.site import (
    _CHAVE_CACHE,
    _TTL_CACHE,
    DadosBolao,
    PalpiteJogo,
)
from bolao.fontes.supabase import buscar_jogos_supabase, obter_headers_supabase
from bolao.modelo import _pontos_bolao
from bolao.nomes import casar_identidade

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


def buscar_leaderboard(
    cliente: httpx.Client, subleague: str, meu_user_id: str = ""
) -> tuple[list[tuple[str, int, int, int]], str | None]:
    """Classificação da subliga via /api/leaderboard.

    Mapeia para o formato do scraping (nome, AP, AV, pts), com AP=exactScoreHits
    (cravadas) e AV=winnerHits (acertos de resultado). A API já vem ordenada por
    pontos e preservamos a ordem.

    Retorna `(classificacao, meu_nome)`, com meu_nome sendo o nome da linha cujo
    `profile.id` casa `meu_user_id` (None se não setado ou não achado). Como o id
    é estável, a identificação auto-segue trocas de nome.
    """
    r = cliente.get(f"{_BASE}/api/leaderboard", params={"subleagueId": subleague})
    r.raise_for_status()
    dados = r.json().get("data", [])
    classificacao: list[tuple[str, int, int, int]] = []
    meu_nome: str | None = None
    for d in dados:
        prof = d.get("profile") or {}
        nome = prof.get("name") or ""
        if not nome:
            continue
        if meu_user_id and prof.get("id") == meu_user_id:
            meu_nome = nome
        classificacao.append(
            (
                nome,
                int(d.get("exactScoreHits", 0)),
                int(d.get("winnerHits", 0)),
                int(d.get("totalPoints", 0)),
            )
        )
    return classificacao, meu_nome


def buscar_picks_jogo(
    cliente: httpx.Client,
    game_id: str,
    placar_real: tuple[int, int],
    nomes_validos: set[str] | None = None,
) -> tuple[dict[str, tuple[int, int]], dict[str, int]]:
    """Picks de um jogo via /api/game-guesses: (picks, pontos) por nome.

    O endpoint devolve TODOS os palpites (todas as ligas) sob a chave `profiles`
    e SEM os pontos, então filtramos por `nomes_validos` (membros da subliga, se
    fornecido) e calculamos os pontos do `placar_real` (3 cravada / 1 resultado /
    0). Ignora linhas mascaradas (is_masked) e palpites sem placar.
    """
    r = cliente.get(f"{_BASE}/api/game-guesses", params={"gameId": game_id})
    r.raise_for_status()
    dados = r.json().get("data", [])
    r_casa, r_fora = placar_real

    picks: dict[str, tuple[int, int]] = {}
    pontos: dict[str, int] = {}
    for d in dados:
        if d.get("is_masked"):
            continue
        nome = (d.get("profiles") or {}).get("name") or ""
        gc = d.get("home_guess")
        gf = d.get("away_guess")
        if not nome or gc is None or gf is None:
            continue
        if nomes_validos is not None and nome not in nomes_validos:
            continue
        try:
            gc_i, gf_i = int(gc), int(gf)
        except (ValueError, TypeError):
            continue
        picks[nome] = (gc_i, gf_i)
        pontos[nome] = _pontos_bolao(gc_i, gf_i, r_casa, r_fora)
    return picks, pontos


def _posicao_de(
    classificacao: list[tuple[str, int, int, int]], meu_nome: str
) -> tuple[int, int]:
    """Posição (1-based) e pontos da linha `meu_nome`; (1, 0) se não encontrar."""
    for i, (nome, _ap, _av, pts) in enumerate(classificacao):
        if nome == meu_nome:
            return i + 1, pts
    return 1, 0


def raspar_bolao_http(config: Config) -> DadosBolao | None:
    """Constrói um DadosBolao equivalente ao do scraping, via HTTP.

    Retorna None (para o chamador cair no fallback Playwright) se a sessão não
    estiver salva, os endpoints recusarem (cookie expirado/401) ou faltarem
    credenciais do PostgREST.
    """
    cookies = carregar_cookies_sessao(config.bolao_subleague)
    if cookies is None:
        logger.info("Sessão HTTP indisponível, caindo para o scraping Playwright")
        return None

    # headers do PostgREST (apikey + Bearer) p/ a lista de jogos/times/placar
    headers_supa = obter_headers_supabase(config)
    if not headers_supa:
        logger.info("Headers Supabase indisponíveis, caindo para o scraping Playwright")
        return None

    try:
        # follow_redirects=True é OBRIGATÓRIO: o www responde 307 para o ápice. Sem
        # seguir o redirect, todo GET vira HTTPStatusError e derruba este caminho HTTP,
        # forçando o fallback de scraping de DOM (Chromium), que satura a CPU do host
        # de 1 core. (httpx não segue redirects por padrão.)
        with httpx.Client(
            cookies=cookies, headers={"user-agent": _UA}, timeout=15.0,
            follow_redirects=True,
        ) as cliente:
            classificacao, meu_nome_id = buscar_leaderboard(
                cliente, config.bolao_subleague, config.bolao_user_id
            )
            if not classificacao:
                logger.warning("Leaderboard HTTP vazio, fallback para scraping")
                return None
            nomes_subliga = {nome for nome, *_ in classificacao}

            # Jogos encerrados: já começaram e têm placar real publicado
            jogos = buscar_jogos_supabase(headers_supa)
            agora = datetime.now(UTC)
            palpites: list[PalpiteJogo] = []
            for j in jogos:
                if j.get("home_score") is None or j.get("away_score") is None:
                    continue  # sem placar = não encerrado
                try:
                    kickoff = datetime.fromisoformat(j["game_time"])
                except (KeyError, ValueError):
                    continue
                if kickoff > agora:
                    continue

                placar = (int(j["home_score"]), int(j["away_score"]))
                picks, pontos = buscar_picks_jogo(cliente, j["id"], placar, nomes_subliga)
                if not picks:
                    continue
                casa = str(j.get("home_team", ""))
                fora = str(j.get("away_team", ""))
                palpites.append(
                    PalpiteJogo(
                        jogo_id=f"{casa} x {fora}",
                        time_casa=casa,
                        time_fora=fora,
                        picks=picks,
                        pontos=pontos,
                    )
                )
    except httpx.HTTPStatusError as e:
        # 401/403 = cookie ou token expirado, fallback (que revalida a sessão)
        logger.warning("HTTP {s} ao montar DadosBolao via API, fallback", s=e.response.status_code)
        return None
    except httpx.HTTPError as e:
        logger.warning("Erro de rede ao montar DadosBolao via API ({e}), fallback", e=e)
        return None

    # identidade na ordem: user_id (auto-segue troca de nome), depois BOLAO_NOME,
    # depois prefixo do e-mail.
    meu_nome = meu_nome_id or casar_identidade(
        [n for n, *_ in classificacao], config.bolao_email, config.bolao_nome
    ) or ""
    minha_posicao, meus_pontos = _posicao_de(classificacao, meu_nome)
    dados = DadosBolao(
        scraped_at=datetime.now(),
        subleague_nome=config.bolao_subleague_nome,
        minha_posicao=minha_posicao,
        meus_pontos=meus_pontos,
        classificacao=classificacao,
        palpites_por_jogo=palpites,
        # /api/leaderboard?subleagueId= já devolve exatamente a subliga
        ja_filtrado_subleague=True,
        meu_nome=meu_nome,
    )
    cache_set(_CHAVE_CACHE, dados, ttl=_TTL_CACHE)
    logger.info(
        "Dados do bolão via HTTP | {n} participantes | {j} jogos com picks",
        n=len(classificacao),
        j=len(palpites),
    )
    return dados
