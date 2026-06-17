"""Utilitários de navegador Playwright compartilhados entre apostador.py e os scrapers.

Concentra login, anti-detecção, persistência de sessão e tratamento de overlays. A
anti-detecção (UA realista, viewport comum, locale/tz pt-BR, patch de
navigator.webdriver) evita que o site rejeite o headless em silêncio. A sessão salva
é reaproveitada e só reloga se expirou, para não bater no rate-limit. Timeout no login
vira ErroLogin; overlay que não fecha dentro do teto vira ErroOverlay.
"""
from __future__ import annotations

import contextlib
import os
import threading
import time
from collections.abc import Iterator
from pathlib import Path

from loguru import logger
from playwright.sync_api import Browser, BrowserContext, Locator, Page, Playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeout

from bolao.cache import CACHE_DIR
from bolao.config import URL_BASE as _URL_BASE
from bolao.erros import ErroLogin, ErroOverlay

_TIMEOUT = 15_000  # ms

# ua de chrome estável em linux, sem o "headlesschrome" que entrega o bot
_UA_REALISTA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

# diretório das sessões salvas (uma por subliga), chmod 600 porque contém cookies
_DIR_SESSAO = CACHE_DIR

# rótulos dos botões que fecham modais de boas-vindas/avisos, em ordem de preferência.
# não usar Escape com o modal de palpite aberto, fecharia o palpite
_ROTULOS_DISPENSAR = (
    "ENTENDIDO", "IGNORAR", "OCULTAR", "OK", "CONFIRMAR", "FECHAR",
    "Entendido", "Ignorar", "Ocultar", "Ok", "Confirmar", "Fechar",
    "entendido", "ignorar", "ocultar", "ok", "confirmar", "fechar",
)

# seletores genéricos de botão de fechar, independentes do texto
_SELETORES_FECHAR = (
    'button[aria-label*="lose" i]',     # "Close", "Fechar"
    'button[aria-label*="echar" i]',
    'button[aria-label*="ispensar" i]',
    'button:has-text("×")',
    'button:has-text("✕")',
    'button:has-text("✖")',
    "[data-dismiss]",
    "[data-radix-collection-item][tabindex]",
)

# teto de tempo para limpar overlays
_TETO_OVERLAY_S = 12.0

# serialização global de chromium. no host de produção (1 core + HT, ~2GB), dois
# browsers simultâneos saturam a CPU, porque jobs de aposta e de scrape (que caem
# no Playwright quando o caminho HTTP falha) rodam no mesmo ThreadPoolExecutor.
# este semáforo garante no máximo uma sessão Playwright por vez entre os jobs.
_LIMITE_BROWSER = threading.BoundedSemaphore(1)
# espera por uma vaga. o teto só estoura se um detentor travar; com os timeouts do
# Playwright e o teto do scrape, o detentor sempre libera em poucos minutos.
_TIMEOUT_VAGA_BROWSER_S = 900.0


@contextlib.contextmanager
def secao_browser() -> Iterator[None]:
    """
    Serializa o uso de Chromium entre jobs concorrentes do agendador.

    Bloqueia até uma vaga ficar livre; levanta TimeoutError se nenhum detentor
    liberar dentro do teto (sinal de sessão Playwright travada). Os chamadores
    (apostar/raspar_bolao/obter_headers_supabase) tratam esse timeout e seguem.
    """
    if not _LIMITE_BROWSER.acquire(timeout=_TIMEOUT_VAGA_BROWSER_S):
        raise TimeoutError(
            f"sem vaga de browser após {_TIMEOUT_VAGA_BROWSER_S:.0f}s, "
            "provável sessão Playwright presa"
        )
    try:
        yield
    finally:
        _LIMITE_BROWSER.release()


def abrir_browser(pw: Playwright) -> Browser:
    """Abre Chromium headless com flags de servidor Linux + anti-automação."""
    return pw.chromium.launch(
        headless=True,
        args=[
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--disable-blink-features=AutomationControlled",
        ],
    )


def criar_contexto(browser: Browser, storage_state: str | None = None) -> BrowserContext:
    """Cria um contexto com cara de navegador real e, se houver, sessão restaurada.

    O init script esconde navigator.webdriver, principal sinal usado para detectar
    automação; UA, viewport e locale completam o disfarce básico.
    """
    contexto = browser.new_context(
        user_agent=_UA_REALISTA,
        viewport={"width": 1366, "height": 768},
        locale="pt-BR",
        timezone_id="America/Sao_Paulo",
        storage_state=storage_state,
    )
    contexto.add_init_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return contexto


def caminho_sessao(subleague: str) -> Path:
    """Arquivo de storage_state para uma subliga (cookies + localStorage da sessão)."""
    return _DIR_SESSAO / f"sessao-{subleague}.json"


# alias interno usado pelas funções privadas deste módulo
_caminho_sessao = caminho_sessao


def _sessao_salva(subleague: str) -> str | None:
    """Caminho da sessão salva, se existir; senão None (faz login do zero)."""
    caminho = _caminho_sessao(subleague)
    return str(caminho) if caminho.exists() else None


def _salvar_sessao(contexto: BrowserContext, subleague: str) -> None:
    """Persiste cookies/estado e restringe a permissão (contém credenciais de sessão)."""
    caminho = _caminho_sessao(subleague)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    contexto.storage_state(path=str(caminho))
    with contextlib.suppress(OSError):
        os.chmod(caminho, 0o600)


def sessao_valida(page: Page, subleague: str) -> bool:
    """True se a sessão restaurada já está logada (não cai em /login)."""
    try:
        page.goto(f"{_URL_BASE}/?subleague={subleague}", wait_until="domcontentloaded")
        return "/login" not in page.url
    except PlaywrightTimeout:
        return False


def fazer_login(page: Page, email: str, password: str, subleague: str) -> None:
    """Faz login e termina na página da subliga, classificando falhas como ErroLogin.

    Em falha de credencial, 2FA ou bloqueio de IP a mensagem diz onde a página parou.
    """
    url_login = f"{_URL_BASE}/login?subleague={subleague}"
    logger.debug("Acessando login: {u}", u=url_login)

    page.goto(url_login, wait_until="domcontentloaded")
    page.fill('input[type="email"], input[name="email"]', email)
    page.fill('input[type="password"], input[name="password"]', password)
    page.click('button[type="submit"]')

    try:
        # login pode ser lento em picos, daí o timeout estendido (45s)
        page.wait_for_url(lambda url: "/login" not in url, timeout=45_000)
    except PlaywrightTimeout as e:
        pista = _pista_erro_login(page)
        raise ErroLogin(
            f"login não concluído (página ainda em {page.url}). {pista}"
        ) from e

    logger.debug("Login realizado com sucesso")
    page.goto(f"{_URL_BASE}/?subleague={subleague}", wait_until="domcontentloaded")


def _pista_erro_login(page: Page) -> str:
    """Tenta extrair uma mensagem de erro de credencial visível para diagnóstico."""
    try:
        alerta = page.locator('[role="alert"], .error, .toast').first
        if alerta.count() and alerta.is_visible():
            return f"Mensagem na tela: '{alerta.inner_text().strip()[:120]}'"
    except Exception:
        pass
    return "Sem mensagem visível (possível 2FA, captcha ou bloqueio de IP)."


def abrir_pagina_logada(
    browser: Browser,
    email: str,
    password: str,
    subleague: str,
) -> Page:
    """Cria contexto, reaproveita sessão salva quando válida e devolve a Page pronta.

    Reusar a sessão evita relogar a cada checkpoint (T-60/45/30/15/5), reduzindo o
    risco de rate-limit ou captcha por login repetido.
    """
    contexto = criar_contexto(browser, storage_state=_sessao_salva(subleague))
    page = contexto.new_page()
    page.set_default_timeout(_TIMEOUT)

    if _sessao_salva(subleague) and sessao_valida(page, subleague):
        logger.debug("Sessão reaproveitada, sem novo login")
        # não regravamos o storage_state aqui: navegar (mesmo com networkidle) não
        # faz o cliente Supabase renovar o JWT no cookie. o Bearer fresco só existe
        # transitoriamente no header de uma chamada /rest/v1, que é o que o sniff de
        # obter_headers_supabase captura. regravar só persistiria o token expirado.
        return page

    fazer_login(page, email, password, subleague)
    _salvar_sessao(contexto, subleague)
    return page


def abrir_browser_logado(
    pw: Playwright,
    email: str,
    password: str,
    subleague: str,
) -> Page:
    """Abre browser e contexto logado e devolve a Page (browser em page.context.browser)."""
    browser = abrir_browser(pw)
    return abrir_pagina_logada(browser, email, password, subleague)


def _modal_palpite_aberto(page: Page) -> bool:
    """True se o modal de palpite está aberto (aí não dá para usar Escape/backdrop)."""
    try:
        modal = page.get_by_role("dialog").filter(has_text="Salvar Palpite")
        return modal.count() > 0 and modal.first.is_visible()
    except Exception:
        return False


def fechar_overlays(page: Page) -> None:
    """Fecha modais de boas-vindas/avisos cujos backdrops interceptam cliques.

    Cascata: rótulos conhecidos, depois botões genéricos de fechar, e por fim
    Escape (só se o modal de palpite não estiver aberto). Limitada por um teto de
    tempo; se o overlay não some, levanta ErroOverlay.
    """
    backdrop = '[class*="backdrop-blur"]'
    inicio = time.monotonic()
    limpos = 0

    while time.monotonic() - inicio < _TETO_OVERLAY_S:
        overlays = page.locator(backdrop)
        if not _algum_visivel(overlays):
            limpos += 1
            if limpos >= 2:
                return
            page.wait_for_timeout(300)
            continue

        limpos = 0
        if _tentar_fechar_overlay(page):
            page.wait_for_timeout(400)
            continue

        # nada clicável: Escape é seguro só se o modal de palpite não estiver aberto
        if not _modal_palpite_aberto(page):
            with contextlib.suppress(Exception):
                page.keyboard.press("Escape")
        page.wait_for_timeout(600)

    if _algum_visivel(page.locator(backdrop)):
        raise ErroOverlay(
            "overlay persistente bloqueou a página após "
            f"{_TETO_OVERLAY_S:.0f}s, provável popup novo sem rótulo conhecido"
        )


def _tentar_fechar_overlay(page: Page) -> bool:
    """Tenta fechar um overlay por rótulo conhecido ou seletor genérico; True se clicou."""
    for rotulo in _ROTULOS_DISPENSAR:
        try:
            botao = page.get_by_text(rotulo, exact=True)
            if botao.count() and botao.first.is_visible():
                botao.first.click()
                logger.debug("Overlay dispensado via rótulo '{r}'", r=rotulo)
                return True
        except Exception:
            continue

    for sel in _SELETORES_FECHAR:
        try:
            el = page.locator(sel).first
            if el.count() and el.is_visible():
                el.click()
                logger.debug("Overlay dispensado via seletor '{s}'", s=sel)
                return True
        except Exception:
            continue
    return False


def _algum_visivel(locator: Locator) -> bool:
    """True se ao menos um elemento do locator estiver visível."""
    for el in locator.all():
        try:
            if el.is_visible():
                return True
        except Exception:
            continue
    return False


def clicar_dispensando(page: Page, locator: Locator, tentativas: int = 4) -> None:
    """Clica num elemento, dispensando overlays que surjam e interceptem o clique.

    Cada falha por interceptação fecha os overlays e tenta de novo.
    """
    ultimo_erro: Exception | None = None
    for _ in range(tentativas):
        try:
            locator.click(timeout=5_000)
            return
        except Exception as e:
            msg = str(e).lower()
            if "intercept" in msg or "timeout" in msg or "not visible" in msg:
                ultimo_erro = e
                fechar_overlays(page)
            else:
                raise
    if ultimo_erro:
        raise ultimo_erro
