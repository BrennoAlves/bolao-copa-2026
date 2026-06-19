"""Reaproveita a sessão logada do Playwright (salva em disco por browser.py) para
autenticar requisições HTTP diretas, sem abrir o navegador.

Evita raspar o DOM com Chromium (o maior pico de CPU do servidor). A sessão
persistida traz o cookie `sb-<ref>-auth-token`, que autentica os endpoints
Next.js `/api/*` e carrega o access_token (JWT) usado como Bearer no PostgREST do
Supabase. Os jobs de aposta (apostador.py) mantêm o storage_state fresco.
"""
from __future__ import annotations

import json
import urllib.parse
from base64 import urlsafe_b64decode
from time import time
from typing import Any

from loguru import logger

from bolao.navegador import caminho_sessao

# Nome do cookie de sessão do Supabase no domínio do site.
# O sufixo é o "ref" do projeto Supabase (mesma origem usada em supabase.py).
_PREFIXO_COOKIE_SUPA = "sb-"
_SUFIXO_COOKIE_SUPA = "-auth-token"


def _ler_storage_state(subleague: str) -> dict[str, Any] | None:
    """Lê o JSON de storage_state salvo pelo Playwright; None se ausente/ilegível."""
    caminho = caminho_sessao(subleague)
    if not caminho.exists():
        logger.debug("Sessão HTTP indisponível: {c} não existe", c=caminho)
        return None
    try:
        with open(caminho, encoding="utf-8") as f:
            state: dict[str, Any] = json.load(f)
            return state
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Falha ao ler storage_state {c}: {e}", c=caminho, e=e)
        return None


def _cookie_supabase(state: dict[str, Any]) -> dict[str, Any] | None:
    """Acha o cookie sb-*-auth-token do storage_state. None se não houver."""
    cookies: list[dict[str, Any]] = state.get("cookies", [])
    for ck in cookies:
        nome = ck.get("name", "")
        if nome.startswith(_PREFIXO_COOKIE_SUPA) and nome.endswith(_SUFIXO_COOKIE_SUPA):
            return ck
    return None


def carregar_cookies_sessao(subleague: str) -> dict[str, str] | None:
    """Cookies da sessão logada para usar nos endpoints Next.js `/api/*`.

    Retorna {nome_cookie: valor} pronto para `httpx.Client(cookies=...)`, ou None
    se a sessão não estiver salva (o chamador deve cair no fallback Playwright).
    """
    state = _ler_storage_state(subleague)
    if state is None:
        return None
    ck = _cookie_supabase(state)
    if ck is None:
        logger.debug("Cookie de sessão Supabase ausente no storage_state")
        return None
    return {ck["name"]: ck["value"]}


def _decodificar_exp(jwt: str) -> int | None:
    """Extrai o campo `exp` (epoch) do payload do JWT, sem validar assinatura."""
    try:
        payload_b64 = jwt.split(".")[1]
        # base64url sem padding: completa para múltiplo de 4
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = json.loads(urlsafe_b64decode(payload_b64))
        exp = payload.get("exp")
        return int(exp) if exp is not None else None
    except (IndexError, ValueError, json.JSONDecodeError):
        return None


def carregar_access_token(subleague: str, margem_s: int = 60) -> str | None:
    """Access token (JWT) do Supabase a partir do cookie de sessão.

    O valor do cookie é um array JSON URL-encoded: [access_token, refresh_token, ...].
    Retorna None se ausente ou se já expirou (com `margem_s` de folga); aí o
    chamador deve renovar via browser (obter_headers_supabase).
    """
    state = _ler_storage_state(subleague)
    if state is None:
        return None
    ck = _cookie_supabase(state)
    if ck is None:
        return None
    try:
        arr = json.loads(urllib.parse.unquote(ck["value"]))
    except (json.JSONDecodeError, KeyError):
        logger.debug("Valor do cookie Supabase em formato inesperado")
        return None
    if not isinstance(arr, list) or not arr or not isinstance(arr[0], str):
        return None
    token = arr[0]

    # Token expirado é inútil para o PostgREST (que não faz refresh como o Next.js)
    exp = _decodificar_exp(token)
    if exp is not None and exp <= time() + margem_s:
        logger.debug("Access token do disco expirado (exp={e}), precisa renovar", e=exp)
        return None
    return token
