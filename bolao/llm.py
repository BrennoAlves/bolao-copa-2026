"""Fallback LLM quando a heurística de matching de nomes ou de overlays falha.

Sem ANTHROPIC_API_KEY (ou sem o pacote `anthropic`) todas as funções retornam
None. Orçamento diário de chamadas para não gastar em loop.
"""
from __future__ import annotations

import os
import re
from datetime import UTC, datetime
from typing import Any

from loguru import logger

from bolao.cache import cache_get, cache_set

_MODELO = "claude-haiku-4-5-20251001"
_MAX_CHAMADAS_DIA = 20
_MAX_TOKENS = 16


def _orcamento_ok() -> bool:
    """True se ainda há orçamento de chamadas hoje (guarda contra gasto em loop)."""
    return _chamadas_hoje() < _MAX_CHAMADAS_DIA


def _chamadas_hoje() -> int:
    chave = f"llm:chamadas:{datetime.now(UTC):%Y-%m-%d}"
    return int(cache_get(chave) or 0)


def _registrar_chamada() -> None:
    chave = f"llm:chamadas:{datetime.now(UTC):%Y-%m-%d}"
    cache_set(chave, _chamadas_hoje() + 1, ttl=2 * 24 * 3600)


def _cliente() -> Any | None:
    """Cliente Anthropic, ou None se sem chave ou sem o pacote (LLM desativado)."""
    if not os.getenv("ANTHROPIC_API_KEY", "").strip():
        return None
    try:
        import anthropic
    except ImportError:
        logger.warning("ANTHROPIC_API_KEY definido mas pacote 'anthropic' não instalado")
        return None
    return anthropic.Anthropic()


def _parse_indice(texto: str, n: int) -> int | None:
    """Extrai o primeiro inteiro do texto e valida que é um índice em [0, n)."""
    m = re.search(r"-?\d+", texto)
    if m is None:
        return None
    idx = int(m.group())
    return idx if 0 <= idx < n else None


def _perguntar_indice(prompt: str, n_opcoes: int, rotulo: str) -> int | None:
    """Pergunta ao LLM e devolve um índice validado, ou None em qualquer falha."""
    cliente = _cliente()
    if cliente is None or not _orcamento_ok():
        return None
    try:
        resp = cliente.messages.create(
            model=_MODELO,
            max_tokens=_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        _registrar_chamada()
        idx = _parse_indice(resp.content[0].text, n_opcoes)
        logger.info("LLM ({r}) respondeu índice {i}", r=rotulo, i=idx)
        return idx
    except Exception as e:
        logger.warning("LLM ({r}) falhou: {e}", r=rotulo, e=e)
        return None


def escolher_card(jogo: str, candidatos: list[str]) -> int | None:
    """Dado o jogo-alvo e os textos dos cards visíveis, devolve o índice do card
    correspondente. Só usado quando o matching por similaridade não resolveu.
    """
    if not candidatos:
        return None
    opcoes = "\n".join(f"{i}: {c[:120]}" for i, c in enumerate(candidatos))
    prompt = (
        f"Qual card corresponde ao jogo '{jogo}'?\n{opcoes}\n"
        "Responda APENAS o número do card, ou -1 se nenhum corresponder."
    )
    return _perguntar_indice(prompt, len(candidatos), "escolher_card")
