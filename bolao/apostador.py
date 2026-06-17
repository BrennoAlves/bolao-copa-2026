"""Automação do site do bolão com Playwright.

Fluxo: abre sessão logada (reaproveitada quando válida), fecha overlays, localiza
o card do jogo por similaridade de nomes, abre o modal, ajusta o placar, salva e
confirma que o site persistiu. Toda falha é classificada (bolao.erros) para o
agendador decidir entre retry e ação manual, e a sessão é rastreada
(bolao.diagnostico) para deixar evidência quando algo quebra. somente_ensaio=True
executa tudo menos salvar.
"""
from __future__ import annotations

import contextlib
import os
import re
from dataclasses import dataclass

from loguru import logger
from playwright.sync_api import Locator, Page, sync_playwright

from bolao.diagnostico import diretorio_para, sessao_diagnostico
from bolao.erros import ErroBolao, ErroCardNaoEncontrado, ErroModalPalpite, ErroVerificacaoIncerta
from bolao.modelo import PlacarPredito, Predicao
from bolao.navegador import (
    _TIMEOUT,
    abrir_browser,
    abrir_pagina_logada,
    clicar_dispensando,
    fechar_overlays,
    secao_browser,
)
from bolao.nomes import nome_presente, traduzir_pt


def _normalizar_texto(texto: str) -> str:
    """Colapsa espaços/newlines em espaço único, para logging e guarda de tamanho."""
    return re.sub(r"\s+", " ", texto).strip()


@dataclass
class ResultadoAposta:
    """Resultado da tentativa de aposta. `tipo_erro` classifica a falha."""

    sucesso: bool
    time_casa: str
    time_fora: str
    gols_casa: int
    gols_fora: int
    mensagem: str
    tipo_erro: str | None = None  # nome da exceção (ErroLogin, ErroOverlay, ...)


def apostar(
    email: str,
    password: str,
    subleague: str,
    predicao: Predicao,
    somente_ensaio: bool = False,
) -> ResultadoAposta:
    """Executa a aposta no site do bolão (ou ensaia, sem salvar, se somente_ensaio).

    Devolve ResultadoAposta com sucesso=True se o palpite foi salvo (ou ensaiado),
    ou sucesso=False com a descrição e o tipo do erro.
    """
    placar = predicao.melhor_placar
    logger.info(
        "Iniciando {modo} | {c} x {f} | palpite: {p}",
        modo="ENSAIO" if somente_ensaio else "aposta",
        c=predicao.time_casa,
        f=predicao.time_fora,
        p=placar.label,
    )
    rotulo = f"aposta-{traduzir_pt(predicao.time_casa)}-{traduzir_pt(predicao.time_fora)}"

    # secao_browser serializa o chromium (no máximo 1 por vez no host de 1 core).
    # se nenhuma vaga abrir no teto, aborta a aposta (o checkpoint seguinte refaz).
    try:
        with secao_browser(), sync_playwright() as pw:
            browser = abrir_browser(pw)
            try:
                page = abrir_pagina_logada(browser, email, password, subleague)
                with sessao_diagnostico(page, rotulo):
                    _apostar_no_jogo(
                        page,
                        predicao.time_casa,
                        predicao.time_fora,
                        placar.gols_casa,
                        placar.gols_fora,
                        somente_ensaio=somente_ensaio,
                    )

                logger.success(
                    "{acao}: {c} {gc} x {gf} {f}",
                    acao="Ensaio concluído" if somente_ensaio else "Palpite salvo",
                    c=predicao.time_casa,
                    gc=placar.gols_casa,
                    gf=placar.gols_fora,
                    f=predicao.time_fora,
                )
                return _resultado(predicao, placar, True, "Palpite registrado com sucesso")

            except ErroBolao as e:
                logger.error("{t}: {e}", t=type(e).__name__, e=e)
                return _resultado(predicao, placar, False, str(e), tipo_erro=type(e).__name__)
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                logger.error("Erro inesperado na aposta: {m}", m=msg)
                return _resultado(predicao, placar, False, msg, tipo_erro=type(e).__name__)
            finally:
                browser.close()
    except TimeoutError as e:
        logger.error("Aposta abortada (sem vaga de browser): {e}", e=e)
        return _resultado(predicao, placar, False, str(e), tipo_erro="TimeoutError")


def _resultado(
    predicao: Predicao,
    placar: PlacarPredito,
    sucesso: bool,
    mensagem: str,
    tipo_erro: str | None = None,
) -> ResultadoAposta:
    """Monta o ResultadoAposta a partir da predição, evitando repetir os campos."""
    return ResultadoAposta(
        sucesso=sucesso,
        time_casa=predicao.time_casa,
        time_fora=predicao.time_fora,
        gols_casa=placar.gols_casa,
        gols_fora=placar.gols_fora,
        mensagem=mensagem,
        tipo_erro=tipo_erro,
    )


def _apostar_no_jogo(
    page: Page,
    time_casa: str,
    time_fora: str,
    gols_casa: int,
    gols_fora: int,
    somente_ensaio: bool = False,
) -> None:
    """Encontra o jogo na página principal e submete (ou ensaia) o palpite."""
    fechar_overlays(page)

    # aguarda os cards: "PLACAR" aparece neles, com fallback no botão de palpite
    try:
        page.wait_for_selector("text=PLACAR", timeout=25_000)
    except Exception:
        logger.debug("'PLACAR' não encontrado, aguardando botão de palpite diretamente")
        page.wait_for_selector('[title="Palpitar"], :text("Editar palpite")', timeout=_TIMEOUT)

    # o site agrupa os jogos por dia e só renderiza o dia corrente por padrão. o
    # card de um jogo de outro dia (todo jogo de 00:00 pertence ao dia seguinte)
    # nunca entra no DOM. aplica o filtro "Todos" antes de procurar, senão o jogo
    # de meia-noite falha eternamente com "card não localizado".
    _mostrar_todos_os_jogos(page)

    # o modal de aviso/boas-vindas ("ÚLTIMA CHANCE", Radix animado) surge de forma
    # assíncrona, depois que os cards renderizam, então a 1ª passada de overlays
    # (logo após o login) corre cedo demais. dispensa e busca em algumas tentativas,
    # tolerando o modal que aparece tarde.
    if not _localizar_card_com_retry(page, time_casa, time_fora):
        raise ErroCardNaoEncontrado(
            f"'{time_casa}' x '{time_fora}' não está aberto para palpite "
            f"(encerrado, prazo expirado ou card não localizado)."
        )

    _abrir_modal_palpite(page, time_casa, time_fora)

    dialog = page.get_by_role("dialog").filter(has_text="Salvar Palpite")
    dialog.wait_for(state="visible", timeout=_TIMEOUT)

    with contextlib.suppress(Exception):
        logger.debug("Modal aberto | conteúdo: {t}", t=_normalizar_texto(dialog.inner_text()))

    n_inputs = dialog.locator("input").count()
    if n_inputs == 0:
        modal_texto = _normalizar_texto(dialog.inner_text()) if dialog.count() else "(sem dialog)"
        raise ErroModalPalpite(
            f"modal de palpite sem inputs (estrutura inesperada). Conteúdo: {modal_texto[:200]}"
        )
    logger.debug("Modal de palpite aberto | {n} inputs encontrados", n=n_inputs)

    _ajustar_gols(dialog, 0, gols_casa)   # input esquerdo = time casa
    _ajustar_gols(dialog, 1, gols_fora)   # input direito = time fora

    if somente_ensaio:
        destino = diretorio_para(f"ensaio-{traduzir_pt(time_casa)}-{traduzir_pt(time_fora)}")
        with contextlib.suppress(Exception):
            page.screenshot(path=str(destino / "modal-preenchido.png"))
        logger.info(
            "ENSAIO: modal preenchido {gc}x{gf} (NÃO salvo). Screenshot em {d}",
            gc=gols_casa, gf=gols_fora, d=destino,
        )
        page.keyboard.press("Escape")
        return

    clicar_dispensando(page, dialog.get_by_text("Salvar Palpite", exact=True))

    sim_salvar = page.get_by_role("button", name="Sim, Salvar")
    sim_salvar.wait_for(state="visible", timeout=_TIMEOUT)
    logger.debug("Confirmação apareceu, clicando 'Sim, Salvar'")
    clicar_dispensando(page, sim_salvar)
    sim_salvar.wait_for(state="hidden", timeout=_TIMEOUT)
    logger.debug("Palpite confirmado e salvo")

    _verificar_aposta_salva(page, time_casa, time_fora, gols_casa, gols_fora)


def _verificar_aposta_salva(
    page: Page,
    time_casa: str,
    time_fora: str,
    gols_casa: int,
    gols_fora: int,
) -> None:
    """Confirma que o site persistiu os valores reabrindo o modal e relendo os inputs.

    Se não conseguir reabrir o modal, levanta ErroVerificacaoIncerta em vez de
    assumir sucesso, para não reportar "apostei" sem ter apostado.
    """
    page.wait_for_timeout(800)  # margem para o react sincronizar o estado pós-save

    # salvar pode resetar a lista para a view padrão (dia corrente); reaplica
    # "Todos" para o card voltar ao DOM antes de reabrir o modal de conferência
    _mostrar_todos_os_jogos(page)

    try:
        _abrir_modal_palpite(page, time_casa, time_fora, preferir_editar=True)
        dialog_check = page.get_by_role("dialog").filter(has_text="Salvar Palpite")
        dialog_check.wait_for(state="visible", timeout=5_000)

        atual_casa = _ler_valor(dialog_check.locator("input").nth(0))
        atual_fora = _ler_valor(dialog_check.locator("input").nth(1))

        page.keyboard.press("Escape")
        dialog_check.wait_for(state="hidden", timeout=5_000)

        if atual_casa != gols_casa or atual_fora != gols_fora:
            raise ErroVerificacaoIncerta(
                f"site mostra {atual_casa}x{atual_fora} mas esperado "
                f"{gols_casa}x{gols_fora}, palpite não foi atualizado"
            )

        logger.info("Verificação ok: palpite {gc}x{gf} confirmado", gc=gols_casa, gf=gols_fora)

    except ErroVerificacaoIncerta:
        raise
    except Exception as e:
        raise ErroVerificacaoIncerta(
            f"não deu para confirmar o palpite ({type(e).__name__}: {e}). "
            "Confira manualmente no site, pode ter sido salvo ou não."
        ) from e


def _mostrar_todos_os_jogos(page: Page) -> None:
    """Aciona o filtro "Todos" para trazer todos os jogos (de qualquer dia) ao DOM.

    O site agrupa por dia e renderiza só o dia corrente por padrão, então o card de
    um jogo que começa em outro dia (caso de todo jogo de 00:00, que pertence ao dia
    seguinte) não entrava na página e a aposta falhava com "card não localizado".
    "Todos" achata a lista, servindo tanto para a aposta inicial (botão "Palpitar")
    quanto para a re-aposta do refinamento (botão "Editar palpite").

    Se o botão sumir (layout mudou), loga e segue: o retry de localização ainda roda
    na view padrão.
    """
    for botao in page.get_by_role("button").all():
        try:
            if not botao.is_visible():
                continue
            texto = (botao.inner_text() or "").strip()
        except Exception:
            continue
        # inner_text() vem em maiúsculo ("TODOS (72)") por causa do text-transform
        # do CSS, daí a comparação case-insensitive (o HTML-fonte guarda "Todos",
        # mas é o texto renderizado que chega aqui). tolera emoji e contagem; não
        # casa com o rodapé "Todos os direitos" (sem "(").
        if "todos (" in texto.lower():
            try:
                clicar_dispensando(page, botao)
                page.wait_for_timeout(800)  # deixa a lista re-renderizar com todos os dias
                logger.debug("Filtro 'Todos' aplicado, jogos de todos os dias no DOM")
            except Exception as e:
                logger.debug("Falha ao aplicar filtro 'Todos' ({e}), seguindo na view padrão", e=e)
            return
    logger.debug("Botão 'Todos' não encontrado, seguindo na view padrão")


def _localizar_card_com_retry(
    page: Page, time_casa: str, time_fora: str, tentativas: int = 4
) -> bool:
    """Dispensa overlays e procura o card, tolerando o modal que aparece atrasado.

    A cada tentativa fecha overlays e procura o card; se um modal animado surgiu
    no meio, a próxima volta o pega.
    """
    for tentativa in range(tentativas):
        fechar_overlays(page)
        if _jogo_aberto_para_aposta(page, time_casa, time_fora):
            return True
        logger.debug("Card não encontrado (tentativa {t}), aguardando overlay tardio", t=tentativa + 1)
        page.wait_for_timeout(700)
    return False


def _palpites_visiveis_em(node: Locator) -> int:
    """Conta botões de palpite visíveis sob o nó.

    O site renderiza 2 botões 'Palpitar' por card (duplicata responsiva
    desktop/mobile), mas só 1 fica visível no breakpoint atual. Contar apenas os
    visíveis distingue 'um card' (1 visível) de 'wrapper de vários jogos' (2+).
    Cuidado com dois extremos: contar o total rejeitava todo card (a 2ª cópia
    oculta inflava a contagem), e aceitar qualquer botão pegava o wrapper do
    vizinho e apostava no jogo errado.
    """
    alvos = (
        node.get_by_title("Palpitar").all()
        + node.get_by_text("Editar palpite", exact=True).all()
    )
    visiveis = 0
    for alvo in alvos:
        with contextlib.suppress(Exception):
            if alvo.is_visible():
                visiveis += 1
    return visiveis


def _jogo_aberto_para_aposta(page: Page, time_casa: str, time_fora: str) -> bool:
    """True se o card do jogo (com botão Palpitar/Editar) está visível na página."""

    botoes = (
        page.get_by_title("Palpitar").all()
        + page.get_by_text("Editar palpite", exact=True).all()
    )
    for botao in botoes:
        with contextlib.suppress(Exception):
            if not botao.is_visible():
                continue
        node = botao
        for _ in range(10):
            node = node.locator("xpath=..")
            try:
                texto = node.inner_text()
            except Exception:
                break
            if len(texto) > 600:
                break
            if nome_presente(time_casa, texto) and nome_presente(time_fora, texto):
                # card único = exatamente 1 botão de palpite visível. 2+ visíveis
                # significa que o ancestral engloba o card vizinho (nomes podem ter
                # vindo de outro jogo), então rejeita para não apostar no jogo errado.
                if _palpites_visiveis_em(node) == 1:
                    return True
                break
    return False


def _abrir_modal_palpite(
    page: Page,
    time_casa: str,
    time_fora: str,
    preferir_editar: bool = False,
) -> None:
    """Clica no botão de palpite da linha do confronto, casando os nomes por
    similaridade (tolera acento, grafia e país novo). Sobe a árvore do botão até
    achar o card com ambos os times, sem ultrapassar o card de um único jogo.
    """

    def _tentar_botao(botao: Locator) -> bool:
        node = botao
        for _ in range(10):
            node = node.locator("xpath=..")
            try:
                texto = node.inner_text()
            except Exception:
                return False
            if len(texto) > 600:
                return False
            if nome_presente(time_casa, texto) and nome_presente(time_fora, texto):
                # crítico: só clica se este ancestral for um card (exatamente 1 botão
                # de palpite visível). com 2+, o ancestral engloba cards vizinhos e os
                # nomes podem ter vindo de outro jogo, e clicar aqui abriria o modal
                # errado. contar só os visíveis ignora a 2ª cópia responsiva oculta
                # do próprio card.
                if _palpites_visiveis_em(node) != 1:
                    return False  # ancestral multi-card: botão é de outro jogo
                clicar_dispensando(page, botao)
                return True
        return False

    def _tentar_estrategia(titulo: str, by_text: bool) -> bool:
        locators = (
            page.get_by_text(titulo, exact=True).all()
            if by_text
            else page.get_by_title(titulo).all()
        )
        for botao in locators:
            with contextlib.suppress(Exception):
                if not botao.is_visible():
                    continue
            if _tentar_botao(botao):
                logger.debug("'{b}' encontrado para {c} x {f}", b=titulo, c=time_casa, f=time_fora)
                return True
        return False

    estrategias = (
        [("Editar palpite", True), ("Palpitar", False)]
        if preferir_editar
        else [("Palpitar", False), ("Editar palpite", True)]
    )
    for titulo, by_text in estrategias:
        if _tentar_estrategia(titulo, by_text):
            return

    # último recurso (opcional): pede ao LLM para desambiguar o card. só dispara
    # com ANTHROPIC_API_KEY configurado; qualquer falha cai no raise abaixo.
    if os.getenv("ANTHROPIC_API_KEY", "").strip() and _tentar_card_via_llm(page, time_casa, time_fora):
        return

    raise ErroCardNaoEncontrado(
        f"'{time_casa}' x '{time_fora}' não encontrado na página "
        "(jogo já começou, prazo expirou ou nomes não casaram com nenhum card)."
    )


def _tentar_card_via_llm(page: Page, time_casa: str, time_fora: str) -> bool:
    """Coleta os textos dos cards com botão de palpite e pede ao LLM o índice do
    jogo-alvo. Qualquer exceção retorna False (cai no erro normal).
    """
    from bolao.llm import escolher_card

    try:
        botoes = (
            page.get_by_title("Palpitar").all()
            + page.get_by_text("Editar palpite", exact=True).all()
        )
        pares: list[tuple[Locator, str]] = []
        for botao in botoes:
            node = botao
            for _ in range(6):
                node = node.locator("xpath=..")
                with contextlib.suppress(Exception):
                    texto = _normalizar_texto(node.inner_text())
                    if 20 <= len(texto) <= 600:
                        pares.append((botao, texto))
                        break
        if not pares:
            return False
        idx = escolher_card(f"{time_casa} x {time_fora}", [t for _, t in pares])
        if idx is None:
            return False
        logger.warning("Card desambiguado via LLM para {c} x {f}", c=time_casa, f=time_fora)
        clicar_dispensando(page, pares[idx][0])
        return True
    except Exception as e:
        logger.debug("Fallback LLM de card não resolveu: {e}", e=e)
        return False


def _ajustar_gols(dialog: Locator, indice_campo: int, gols_alvo: int) -> None:
    """Ajusta o nº de gols de um time clicando + / - até gols_alvo, partindo de
    qualquer valor atual. Tenta force=True se o React não atualizar.
    """
    inp = dialog.locator("input").nth(indice_campo)
    mais = inp.locator("xpath=following-sibling::button[1]")
    menos = inp.locator("xpath=preceding-sibling::button[1]")
    pagina = inp.page

    def _clicar_ate(botao: Locator, condicao: str) -> None:
        atual = _ler_valor(inp)
        while (atual < gols_alvo) if condicao == "subir" else (atual > gols_alvo):
            botao.click()
            pagina.wait_for_timeout(200)
            novo = _ler_valor(inp)
            if novo == atual:
                botao.click(force=True)  # react não atualizou, força o clique
                pagina.wait_for_timeout(300)
                novo = _ler_valor(inp)
            if novo == atual:
                raise ErroModalPalpite(
                    f"campo {indice_campo}: botão não atualizou o valor "
                    f"(preso em {atual}, alvo={gols_alvo})"
                )
            atual = novo

    _clicar_ate(mais, "subir")
    _clicar_ate(menos, "descer")

    final = _ler_valor(inp)
    if final != gols_alvo:
        raise ErroModalPalpite(
            f"falha ao ajustar campo {indice_campo}: esperado {gols_alvo}, input mostra {final}"
        )
    logger.debug("Campo {i}: {g} gols confirmados", i=indice_campo, g=gols_alvo)


def _ler_valor(inp: Locator) -> int:
    """Lê o valor inteiro de um input de gols (vazio = 0)."""
    try:
        texto = (inp.input_value() or "").strip()
        return int(texto) if texto else 0
    except (ValueError, TypeError):
        return 0
