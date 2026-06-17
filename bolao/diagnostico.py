"""Observabilidade do Playwright: captura evidência quando a automação falha.

Embrulha uma sessão do navegador e, em qualquer exceção, salva um pacote de
diagnóstico (trace.zip, screenshot e HTML) num diretório datado, loga o caminho
em ERROR e re-levanta. Em sucesso o trace é descartado, a menos que
BOLAO_TRACE=sempre, para não encher o disco do servidor.
"""
from __future__ import annotations

import contextlib
import os
import shutil
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from playwright.sync_api import Page

from bolao.cache import CACHE_DIR

_DIR_DIAGNOSTICOS = CACHE_DIR / "diagnosticos"


def _trace_sempre() -> bool:
    """True se BOLAO_TRACE=sempre: guarda o trace mesmo em sucesso (debug)."""
    return os.getenv("BOLAO_TRACE", "").strip().lower() == "sempre"


@contextlib.contextmanager
def sessao_diagnostico(page: Page, rotulo: str) -> Iterator[None]:
    """Liga o tracing do contexto e captura evidência se algo falhar no bloco.

    O rotulo vira prefixo do diretório de diagnóstico (ex: "aposta-brasil-arg").
    """
    contexto = page.context
    with contextlib.suppress(Exception):
        contexto.tracing.start(screenshots=True, snapshots=True)

    sucesso = False
    try:
        yield
        sucesso = True
    except Exception:
        _salvar_evidencia(page, rotulo)
        raise
    finally:
        _encerrar_tracing(contexto, rotulo, salvar=(_trace_sempre() and sucesso))


def _salvar_evidencia(page: Page, rotulo: str) -> None:
    """Salva screenshot + HTML da página no diretório do diagnóstico atual."""
    destino = diretorio_para(rotulo)
    with contextlib.suppress(Exception):
        page.screenshot(path=str(destino / "tela.png"), full_page=True)
    with contextlib.suppress(Exception):
        (destino / "pagina.html").write_text(page.content(), encoding="utf-8")
    with contextlib.suppress(Exception):
        page.context.tracing.stop(path=str(destino / "trace.zip"))
    logger.error(
        "Diagnóstico salvo em {d}, baixe e rode `playwright show-trace {d}/trace.zip`",
        d=destino,
    )


def _encerrar_tracing(contexto: object, rotulo: str, salvar: bool) -> None:
    """Para o tracing; em sucesso normal apenas descarta (stop sem path)."""
    tracing = getattr(contexto, "tracing", None)
    if tracing is None:
        return
    with contextlib.suppress(Exception):
        if salvar:
            tracing.stop(path=str(diretorio_para(rotulo) / "trace.zip"))
        else:
            tracing.stop()


def diretorio_para(rotulo: str) -> Path:
    """Diretório datado para um diagnóstico/ensaio; cria se preciso."""
    seguro = "".join(c if c.isalnum() or c in "-_" else "-" for c in rotulo)[:60]
    carimbo = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    destino = _DIR_DIAGNOSTICOS / f"{seguro}-{carimbo}"
    destino.mkdir(parents=True, exist_ok=True)
    return destino


def limpar_diagnosticos_antigos(max_dias: int = 7) -> None:
    """Remove diagnósticos com mais de max_dias, chamado no startup do daemon."""
    if not _DIR_DIAGNOSTICOS.exists():
        return
    limite = time.time() - max_dias * 86_400
    for sub in _DIR_DIAGNOSTICOS.iterdir():
        with contextlib.suppress(OSError):
            if sub.is_dir() and sub.stat().st_mtime < limite:
                shutil.rmtree(sub, ignore_errors=True)
