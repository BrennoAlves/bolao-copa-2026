"""Cache em disco com TTL usando diskcache, para evitar chamadas repetidas às
APIs externas entre as tentativas de retry. Fica em ~/.cache/bolao/.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import diskcache

# Diretório base de tudo que persiste em disco (cache, sessão, diagnósticos, log).
CACHE_DIR = Path.home() / ".cache" / "bolao"
_cache = diskcache.Cache(str(CACHE_DIR))


def cache_get(chave: str) -> Any | None:
    return _cache.get(chave)


def cache_set(chave: str, valor: Any, ttl: int = 600) -> None:
    """Armazena um valor no cache com TTL em segundos."""
    _cache.set(chave, valor, expire=ttl)


def cache_chaves(prefixo: str = "") -> list[str]:
    """Lista as chaves vivas (não expiradas) que começam com `prefixo`."""
    return [
        str(chave)
        for chave in _cache.iterkeys()
        if str(chave).startswith(prefixo)
    ]
