"""Integração com a API REST do Supabase do site do bolão.

Lê os palpites dos adversários (tabela `guesses`) para alimentar a Teoria dos
Jogos. A RLS do site só libera palpites alheios DEPOIS do kickoff, então só
extraímos de jogos já iniciados (ler antes volta vazio e dispara alerta de
violação de política). O sinal pré-kickoff vem de `estimar_perfis`, sobre os
palpites de jogos encerrados.
"""
from __future__ import annotations

import contextlib
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
from loguru import logger
from playwright.sync_api import Request, sync_playwright

from bolao.config import SUPABASE_REST_URL as _SUPA_URL
from bolao.config import URL_BASE as _SITE_URL
from bolao.config import Config
from bolao.fontes.sessao_http import carregar_access_token, carregar_cookies_sessao
from bolao.navegador import abrir_browser_logado, caminho_sessao, fechar_overlays, secao_browser
from bolao.nomes import nome_presente

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

# Headers do Supabase retidos globalmente para reuso entre jobs sem abrir o browser
_headers_supa: dict[str, str] = {}
_headers_supa_ts: float = 0.0          # timestamp (monotonic) da última captura
_HEADERS_TTL: float = 50 * 60          # renova após 50 min (JWT expira em ~60 min)

# Cache da lista de jogos (estática durante o torneio)
_jogos_cache: list[dict] = []
_jogos_cache_ts: float = 0.0
_JOGOS_TTL: float = 30 * 60            # renova a cada 30 min

# cache dos membros da subliga (user_id -> nome): restringe a Teoria dos Jogos
# aos adversários reais da liga (não a todas as ligas do site)
_membros_cache: dict[str, str] = {}
_membros_cache_ts: float = 0.0


def _caminho_apikey(subleague: str) -> Path:
    """Arquivo onde persistimos a apikey (anon key) pública, estática por projeto."""
    return caminho_sessao(subleague).parent / "supabase-apikey.txt"


def _ler_apikey_persistida(subleague: str) -> str | None:
    """Lê a apikey salva de uma captura anterior; None se nunca foi capturada."""
    caminho = _caminho_apikey(subleague)
    try:
        with open(caminho, encoding="utf-8") as f:
            valor = f.read().strip()
            return valor or None
    except OSError:
        return None


def _persistir_apikey(subleague: str, apikey: str | None) -> None:
    """Salva a apikey pública pra permitir montar headers do disco sem browser."""
    if not apikey:
        return
    caminho = _caminho_apikey(subleague)
    with contextlib.suppress(OSError):
        caminho.parent.mkdir(parents=True, exist_ok=True)
        with open(caminho, "w", encoding="utf-8") as f:
            f.write(apikey)


def _headers_do_disco(config: Config) -> dict[str, str] | None:
    """Monta {apikey, authorization} sem abrir o browser: apikey persistida + JWT
    (Bearer) lido do cookie de sessão salvo. None se faltar apikey ou o JWT
    estiver expirado/ausente; aí o chamador faz o sniff via Playwright.
    """
    apikey = _ler_apikey_persistida(config.bolao_subleague)
    if not apikey:
        return None
    token = carregar_access_token(config.bolao_subleague)
    if not token:
        return None
    return {"apikey": apikey, "authorization": f"Bearer {token}"}


def obter_headers_supabase(config: Config, forcar_renovacao: bool = False) -> dict[str, str]:
    """Headers de auth do Supabase (apikey + Bearer JWT) para o PostgREST.

    Ordem de preferência, do mais barato ao mais caro: cache em memória (dentro
    do TTL), montagem a partir do storage_state em disco (sem browser), e sniff
    via Playwright (fallback), que também persiste a apikey pública.
    """
    global _headers_supa, _headers_supa_ts
    jwt_expirado = (time.monotonic() - _headers_supa_ts) > _HEADERS_TTL
    if _headers_supa and not forcar_renovacao and not jwt_expirado:
        return _headers_supa

    # tenta montar do disco antes de gastar um Chromium
    if not forcar_renovacao:
        do_disco = _headers_do_disco(config)
        if do_disco:
            _headers_supa = do_disco
            _headers_supa_ts = time.monotonic()
            logger.info("Headers do Supabase montados do storage_state (sem browser).")
            return _headers_supa

    if jwt_expirado and _headers_supa:
        logger.info("JWT do Supabase expirado, renovando headers...")

    logger.info("Obtendo headers de auth do Supabase via Playwright...")
    novos_headers: dict[str, str] = {}

    # secao_browser serializa com apostas/scrapes: este sniff sobe um Chromium e
    # não pode rodar em paralelo no host de 1 core. TimeoutError (sem vaga) propaga
    # e é tolerado pelos chamadores (caem para scraping / palpites vazios).
    with secao_browser(), sync_playwright() as pw:
        # Usa os utils padrão para reaproveitar cookies do bolão
        try:
            page = abrir_browser_logado(
                pw, config.bolao_email, config.bolao_password, config.bolao_subleague
            )
        except Exception as e:
            logger.error("Erro ao abrir browser logado no Supabase fetcher: {e}", e=e)
            return {}

        browser = page.context.browser

        def _on_request(req: Request) -> None:
            # authorization é o JWT de sessão (só aparece em requests autenticados).
            # apikey é pública (anon key), mas também precisamos dela nos headers da API.
            # sai assim que tiver o authorization (prova que o login funcionou).
            if "authorization" in novos_headers:
                return
            if "/rest/v1/" in req.url and req.method == "GET":
                h = {k.lower(): v for k, v in req.headers.items()}
                for k in ("apikey", "authorization", "x-client-info", "accept-profile"):
                    if k in h:
                        novos_headers[k] = h[k]

        page.on("request", _on_request)

        try:
            fechar_overlays(page)
            # Recarrega para garantir que novos requests Supabase disparem após o listener estar ativo
            with contextlib.suppress(Exception):
                page.reload(wait_until="networkidle", timeout=15_000)

            if "authorization" not in novos_headers:
                logger.warning("Não foi possível capturar o token de sessão do Supabase.")
                return {}

            _headers_supa = novos_headers
            _headers_supa_ts = time.monotonic()
            # persiste a apikey pública: nas próximas vezes montamos do disco sem browser
            _persistir_apikey(config.bolao_subleague, novos_headers.get("apikey"))
            logger.success("Headers do Supabase capturados com sucesso.")
            return _headers_supa

        except Exception as e:
            logger.error("Erro durante a captura de headers do Supabase: {e}", e=e)
            return {}
        finally:
            if browser:
                browser.close()


def buscar_jogos_supabase(headers: dict[str, str]) -> list[dict]:
    """Lista os jogos da base do Supabase e mapeia os nomes dos times.

    Retorna jogos com 'id', 'game_time', 'home_team', 'away_team' e o placar real
    ('home_score'/'away_score', None se ainda não finalizado). Cacheado por
    _JOGOS_TTL segundos: a lista é estática durante o torneio.
    """
    global _jogos_cache, _jogos_cache_ts
    if _jogos_cache and (time.monotonic() - _jogos_cache_ts) < _JOGOS_TTL:
        return _jogos_cache

    if not headers:
        return []
    url_games = (
        f"{_SUPA_URL}/games?select=id,game_time,home_team_id,away_team_id,home_score,away_score"
    )
    url_teams = f"{_SUPA_URL}/teams?select=id,name"
    try:
        r_games = httpx.get(url_games, headers=headers, timeout=10.0)
        r_games.raise_for_status()
        r_teams = httpx.get(url_teams, headers=headers, timeout=10.0)
        r_teams.raise_for_status()

        teams_map = {t["id"]: t["name"] for t in r_teams.json()}

        jogos = []
        for g in r_games.json():
            g['home_team'] = teams_map.get(g.get("home_team_id"), "")
            g['away_team'] = teams_map.get(g.get("away_team_id"), "")
            jogos.append(g)

        _jogos_cache = jogos
        _jogos_cache_ts = time.monotonic()
        return jogos
    except Exception as e:
        logger.error("Erro ao buscar jogos/times do Supabase: {e}", e=e)
        return []


def buscar_membros_subliga(config: Config) -> dict[str, str]:
    """Mapa user_id -> nome dos membros da subliga alvo, via /api/leaderboard.

    Restringe a Teoria dos Jogos aos adversários da liga em que o usuário compete:
    o endpoint de palpites devolve TODAS as ligas do site. Cacheado por _JOGOS_TTL
    (a composição da liga é estável). {} se sem sessão.
    """
    global _membros_cache, _membros_cache_ts
    if _membros_cache and (time.monotonic() - _membros_cache_ts) < _JOGOS_TTL:
        return _membros_cache

    cookies = carregar_cookies_sessao(config.bolao_subleague)
    if not cookies:
        return {}
    url = f"{_SITE_URL}/api/leaderboard?subleagueId={config.bolao_subleague}"
    try:
        # follow_redirects=True: www para o ápice responde 307 (ver site_http). sem
        # isso a resolução de membros falha calada e a Teoria dos Jogos cai no perfil
        # de "todos os palpiteiros do site" em vez dos adversários reais da subliga.
        r = httpx.get(
            url, cookies=cookies, headers={"user-agent": _UA}, timeout=10.0,
            follow_redirects=True,
        )
        r.raise_for_status()
        membros: dict[str, str] = {}
        for d in r.json().get("data", []):
            prof = d.get("profile") or {}
            uid = prof.get("id")
            if uid:
                membros[uid] = prof.get("name", "")
        if membros:
            _membros_cache = membros
            _membros_cache_ts = time.monotonic()
        return membros
    except Exception as e:
        logger.warning("Falha ao buscar membros da subliga: {e}", e=e)
        return {}


def buscar_palpites_oponentes(
    game_id: str,
    headers: dict[str, str],
    ids_oponentes: set[str] | None = None,
) -> list[tuple[int, int]]:
    """Palpites dos oponentes para o game_id, como (gols_casa, gols_fora).

    `ids_oponentes` filtra pelos adversários da subliga (sem o próprio usuário).
    None = todos os usuários do site, só quando não dá pra resolver os membros.
    """
    if not headers:
        return []
    url = f"{_SUPA_URL}/guesses?game_id=eq.{game_id}&select=user_id,home_guess,away_guess"
    try:
        r = httpx.get(url, headers=headers, timeout=10.0)
        r.raise_for_status()
        data = r.json()

        palpites = []
        for row in data:
            if ids_oponentes is not None and row.get("user_id") not in ids_oponentes:
                continue
            gc = row.get("home_guess")
            gf = row.get("away_guess")
            if gc is None or gf is None:
                continue
            try:
                palpites.append((int(gc), int(gf)))
            except (ValueError, TypeError):
                continue

        return palpites
    except Exception as e:
        logger.error("Erro ao buscar palpites do Supabase para {id}: {e}", id=game_id, e=e)
        return []


def extrair_palpites_jogo(config: Config, time_casa: str, time_fora: str) -> list[tuple[int, int]]:
    """Palpites dos oponentes de um jogo pelos nomes dos times (resolve o game_id)."""
    headers = obter_headers_supabase(config)
    if not headers:
        return []
        
    jogos = buscar_jogos_supabase(headers)

    # Encontrar o jogo por similaridade ou exatidão usando a função existente nome_presente
    jogo = None
    for j in jogos:
        j_casa = str(j.get("home_team", ""))
        j_fora = str(j.get("away_team", ""))

        if nome_presente(time_casa, j_casa) and nome_presente(time_fora, j_fora):
            jogo = j
            break

    if jogo is None:
        logger.warning(
            "Jogo {c} x {f} não encontrado no Supabase para extração de palpites",
            c=time_casa, f=time_fora
        )
        return []

    # a RLS só libera palpites alheios DEPOIS do kickoff. antes disso a leitura
    # volta vazia E dispara alerta de violação de política no Supabase, então
    # nem tentamos: a Teoria dos Jogos pré-kickoff usa só perfis de jogos
    # encerrados (estimar_perfis), não o palpite do jogo que vai começar.
    try:
        kickoff = datetime.fromisoformat(jogo["game_time"])
    except (KeyError, ValueError):
        kickoff = None
    if kickoff is not None and kickoff > datetime.now(UTC):
        logger.info(
            "Palpites de {c} x {f} ainda mascarados (pré-kickoff), pulando leitura",
            c=time_casa, f=time_fora,
        )
        return []

    game_id = jogo["id"]

    # Restringe aos adversários da subliga (excluindo o próprio usuário). Sem os
    # membros (ex.: sessão indisponível), cai em None = todos os palpiteiros do site.
    membros = buscar_membros_subliga(config)
    ids_oponentes: set[str] | None = None
    if membros:
        prefixo = config.bolao_email.lower()
        meu_id = next(
            (uid for uid, nome in membros.items()
             if nome and prefixo.startswith(nome.lower().split()[0].lower())),
            None,
        )
        ids_oponentes = {uid for uid in membros if uid != meu_id}

    logger.info("Extraindo dados de palpites do jogo {c} x {f} (ID: {id})", c=time_casa, f=time_fora, id=game_id)
    palpites = buscar_palpites_oponentes(game_id, headers, ids_oponentes=ids_oponentes)
    logger.info(
        "{n} palpites de oponentes extraídos ({m} membros da subliga).",
        n=len(palpites), m=len(ids_oponentes) if ids_oponentes is not None else -1,
    )

    return palpites
