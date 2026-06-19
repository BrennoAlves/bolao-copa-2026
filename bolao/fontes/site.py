"""Scraper do site do bolão: raspa standings e picks dos participantes da subliga.

Picks só ficam visíveis após o kickoff. Os placares reais chegam horas depois do
encerramento, por isso o scrape de standings roda T+3h após o kickoff, não logo
após o resultado. Dados ficam em diskcache (TTL=48h) para poupar raspagens.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger
from playwright.sync_api import Locator, Page, sync_playwright

from bolao.cache import cache_get, cache_set
from bolao.config import Config
from bolao.diagnostico import sessao_diagnostico
from bolao.navegador import (
    abrir_browser_logado,
    clicar_dispensando,
    fechar_overlays,
    secao_browser,
)
from bolao.nomes import casar_identidade

_CHAVE_CACHE = "bolao:dados_scraped"
_TTL_CACHE = 48 * 3600  # 48 horas

# Teto de tempo para o scrape de picks de TODOS os jogos encerrados. A lista cresce
# com o torneio e cada modal renderiza centenas de <li> em software (SwiftShader),
# pregando o core a 100%. Sem teto, uma passada chegou a ~28 min em produção. Este
# limite garante que o fallback de DOM nunca prenda a CPU indefinidamente.
_TETO_SCRAPE_PICKS_S = 180.0

# regex de cada listitem de pick. inner_text() do <li> separa os campos por \n:
#   "2\nx\n0\n+3\nFulano"      gols_casa=2, gols_fora=0, pontos=3
#   "0\nx\n1\n0\nBeltrano"     gols_casa=0, gols_fora=1, pontos=0
#   "E\n1\nx\n1\n0\nCiclano"   prefixo "E" indica empate; gols_casa=1, gols_fora=1
# o grupo de pontos (+3/+1/0) é capturado: quem recebeu +3 tem o placar real.
_RE_PICK = re.compile(r"^(?:E\n)?(\d+)\nx\n(\d+)\n([+\-]?\d+)\n(.+)$")


@dataclass
class PalpiteJogo:
    """Picks de todos os participantes num único jogo."""

    jogo_id: str    # "time_casa x time_fora" em português
    time_casa: str
    time_fora: str
    picks: dict[str, tuple[int, int]] = field(default_factory=dict)
    # nome_participante -> (gols_casa, gols_fora)
    pontos: dict[str, int] = field(default_factory=dict)
    # nome_participante -> pontos ganhos no jogo (3 cravada / 1 resultado / 0).
    # o placar de quem fez 3 é o resultado real do jogo (ver resultado_real).

    @property
    def resultado_real(self) -> tuple[int, int] | None:
        """Placar real do jogo: o pick de quem cravou (+3). None se ninguém cravou."""
        # getattr: objetos picklados num cache antigo, antes do campo `pontos`,
        # não têm o atributo. nesse caso devolve None.
        for nome, pts in (getattr(self, "pontos", None) or {}).items():
            if pts == 3:
                return self.picks[nome]
        return None


@dataclass
class DadosBolao:
    """Snapshot das standings e picks de uma subliga, num instante."""

    scraped_at: datetime
    subleague_nome: str
    minha_posicao: int          # minha posição na classificação (1-based)
    meus_pontos: int
    classificacao: list[tuple[str, int, int, int]]  # (nome, AP, AV, pts)
    palpites_por_jogo: list[PalpiteJogo]
    # True quando a classificação já é exatamente a subliga (HTTP via subleagueId);
    # nesse caso o filtro BOLAO_MEMBROS é dispensado, a fonte já é autoritativa.
    ja_filtrado_subleague: bool = False
    # Meu nome resolvido na classificação (por user_id no HTTP, ou nome/e-mail).
    # Consumido pela estratégia/ranking para auto-seguir trocas de nome. "" = não resolvido.
    meu_nome: str = ""


def raspar_bolao(config: Config) -> DadosBolao | None:
    """Abre o browser, faz login e raspa standings + picks da subliga alvo.

    Persiste o resultado em diskcache para evitar raspagens repetidas quando
    vários jobs buscam os mesmos dados em curto intervalo. None se falhar.
    """
    logger.info(
        "Iniciando scrape do bolão | subliga={s}",
        s=config.bolao_subleague_nome,
    )

    try:
        # secao_browser garante que este scrape (fallback do caminho HTTP) nunca
        # rode em paralelo com uma aposta ou outro scrape: dois Chromium no host
        # de 1 core saturavam a CPU.
        with secao_browser(), sync_playwright() as pw:
            page = abrir_browser_logado(
                pw,
                config.bolao_email,
                config.bolao_password,
                config.bolao_subleague,
            )
            browser = page.context.browser
            try:
                with sessao_diagnostico(page, "scrape-bolao"):
                    fechar_overlays(page)
                    classificacao = _raspar_standings(page, config.bolao_email)
                    palpites = _raspar_picks_encerrados(page, config.bolao_subleague_nome)
            finally:
                if browser:
                    browser.close()

    except Exception as e:
        logger.error("Erro ao raspar bolão: {e}", e=e)
        return None

    minha_posicao = 1
    meus_pontos = 0
    # Filtra à subliga para que a posição seja relativa a ela, não à liga inteira
    bolao_membros = set(config.bolao_membros) if config.bolao_membros else set()
    class_alvo = [
        (nome, ap, av, pts) for nome, ap, av, pts in classificacao
        if not bolao_membros or nome in bolao_membros
    ]
    nomes_alvo = [n for n, *_ in class_alvo]
    eu = casar_identidade(nomes_alvo, config.bolao_email, config.bolao_nome)
    if eu is not None:
        i = nomes_alvo.index(eu)
        minha_posicao, meus_pontos = i + 1, class_alvo[i][3]
    else:
        # Não está no recorte da subliga (grafia divergente em BOLAO_MEMBROS?): cai na
        # liga completa para não retornar posição/pontos zerados em silêncio.
        nomes_full = [n for n, *_ in classificacao]
        eu = casar_identidade(nomes_full, config.bolao_email, config.bolao_nome)
        if eu is not None:
            i = nomes_full.index(eu)
            minha_posicao, meus_pontos = i + 1, classificacao[i][3]
            logger.warning(
                "Usuário não encontrado no recorte da subliga (posição relativa à liga: {p}). "
                "Verifique BOLAO_MEMBROS no .env.",
                p=minha_posicao,
            )

    dados = DadosBolao(
        scraped_at=datetime.now(),
        subleague_nome=config.bolao_subleague_nome,
        minha_posicao=minha_posicao,
        meus_pontos=meus_pontos,
        classificacao=class_alvo,
        palpites_por_jogo=palpites,
        meu_nome=eu or "",
    )

    cache_set(_CHAVE_CACHE, dados, ttl=_TTL_CACHE)
    logger.info(
        "Scrape concluído | {n} participantes | {j} jogos com picks",
        n=len(classificacao),
        j=len(palpites),
    )
    return dados


def carregar_dados_bolao() -> DadosBolao | None:
    """Retorna os dados do bolão do cache, ou None se ausentes/expirados."""
    return cache_get(_CHAVE_CACHE)


def obter_dados_bolao(config: Config) -> DadosBolao | None:
    """Obtém os dados do bolão preferindo HTTP (sem browser); cai no scraping
    Playwright se o HTTP estiver desligado ou falhar.

    Ponto único chamado pelos jobs de leitura: concentra a política de fallback
    para que standings/picks sempre tenham uma fonte.
    """
    if config.usar_http_scrape:
        # import tardio: site_http importa este módulo (evita ciclo no load).
        from bolao.fontes.site_http import raspar_bolao_http

        try:
            dados = raspar_bolao_http(config)
            if dados is not None:
                return dados
            logger.info("HTTP não retornou dados, usando scraping Playwright")
        except Exception as e:
            logger.warning("Falha no caminho HTTP ({e}), usando scraping Playwright", e=e)

    return raspar_bolao(config)


def _raspar_standings(
    page: Page,
    bolao_email: str,
) -> list[tuple[str, int, int, int]]:
    """Lê a tabela de classificação da página atual.

    Cada linha tem 5 colunas [rank, nome, AP, AV, pts]; o rank é descartado. O
    overlay de boas-vindas aparece async após o login e pode cobrir a tabela,
    então fecha overlays e re-verifica até a tabela ter linhas.
    """
    # O overlay de boas-vindas aparece de forma assíncrona após o login e impede
    # que a tabela de classificação seja renderizada. Tenta fechar overlays e
    # aguarda qualquer tabela com linhas (não assume posição fixa, o site às
    # vezes renderiza 1 tabela, às vezes 2, dependendo do estado da página).
    _SCRIPT_AGUARDA_QUALQUER_TABELA = (
        "() => [...document.querySelectorAll('table')]"
        "      .some(t => t.querySelectorAll('tbody tr').length > 0)"
    )
    for tentativa in range(4):
        fechar_overlays(page)
        try:
            page.wait_for_function(_SCRIPT_AGUARDA_QUALQUER_TABELA, timeout=10_000)
            break
        except Exception:
            logger.debug(
                "Standings ainda não carregaram (tentativa {t})",
                t=tentativa + 1,
            )
            page.wait_for_timeout(1_500)
    else:
        logger.warning("Tabela de standings não carregou após 4 tentativas")
        return []

    # Encontra a tabela de classificação por estrutura: 5 colunas (rank, nome, AP, AV, pts).
    # Não usa posição fixa (nth(1)) pois o número de tabelas varia por estado da página.
    tabela_classificacao = None
    n_tabelas = page.locator("table").count()
    logger.debug("Tabelas encontradas na página: {n}", n=n_tabelas)
    for idx in range(n_tabelas):
        tabela = page.locator("table").nth(idx)
        primeira = tabela.locator("tbody tr").first
        try:
            n_cols = primeira.locator("td").count()
        except Exception:
            continue
        if n_cols >= 5:
            tabela_classificacao = tabela
            logger.debug("Tabela de standings em nth({i}) | {c} colunas", i=idx, c=n_cols)
            break

    if tabela_classificacao is None:
        logger.warning("Nenhuma tabela com 5+ colunas encontrada para standings")
        return []

    linhas = tabela_classificacao.locator("tbody tr").all()

    classificacao: list[tuple[str, int, int, int]] = []
    for linha in linhas:
        try:
            celulas = linha.locator("td").all()
            # A tabela tem 5 colunas: [rank, nome, AP, AV, pts]
            # A 1ª coluna é o número de posição (ex: "1\n1º"); nome está na 2ª.
            if len(celulas) < 5:
                continue
            # o site adiciona ícones na primeira linha do nome.
            # pega a última linha não-vazia como nome limpo.
            linhas_nome = [ln.strip() for ln in celulas[1].inner_text().split("\n") if ln.strip()]
            nome = linhas_nome[-1] if linhas_nome else ""
            ap = _int_seguro(celulas[2].inner_text())
            av = _int_seguro(celulas[3].inner_text())
            pts = _int_seguro(celulas[4].inner_text())
            if nome:
                classificacao.append((nome, ap, av, pts))
        except Exception as e:
            logger.debug("Erro ao parsear linha da tabela: {e}", e=e)
            continue

    logger.debug("Standings: {n} participantes lidos", n=len(classificacao))
    return classificacao


def _raspar_picks_encerrados(
    page: Page,
    subleague_nome: str,
) -> list[PalpiteJogo]:
    """Navega para a aba de jogos encerrados e extrai os picks por jogo.

    Para cada jogo, clica em "Mostrar todos os palpites", filtra pela tab da
    subliga alvo e parseia os listitems.
    """
    palpites: list[PalpiteJogo] = []

    # Tenta clicar na aba de encerrados (overlay pode ter reaberto desde o login)
    try:
        fechar_overlays(page)
        btn_encerrados = page.get_by_text("Encerrados", exact=False)
        if not btn_encerrados.count():
            logger.info("Aba Encerrados não encontrada na página")
            return palpites
        clicar_dispensando(page, btn_encerrados.first)
        page.wait_for_timeout(1_500)
    except Exception as e:
        logger.warning("Não encontrou aba Encerrados: {e}", e=e)
        return palpites

    # Pega todos os botões "Mostrar todos os palpites"
    try:
        page.wait_for_selector("text=Mostrar", timeout=5_000)
    except Exception:
        logger.info("Nenhum jogo encerrado com palpites disponíveis")
        return palpites

    _TEXTO_MOSTRAR = "Mostrar todos os palpites"
    n_jogos = page.get_by_text(_TEXTO_MOSTRAR, exact=False).count()
    logger.debug("{n} jogos encerrados encontrados", n=n_jogos)

    # guarda contra modal que não atualiza entre cards: se um modal não recarrega,
    # dois jogos diferentes saem com o MESMO conjunto de picks. dedup por conteúdo
    # descarta o duplicado.
    picks_vistos: set[frozenset[tuple[str, tuple[int, int]]]] = set()

    inicio = time.monotonic()
    for i in range(n_jogos):
        if time.monotonic() - inicio > _TETO_SCRAPE_PICKS_S:
            logger.warning(
                "Scrape de picks excedeu {t:.0f}s, interrompendo em {i}/{n} jogos "
                "(picks parciais ainda servem aos perfis)",
                t=_TETO_SCRAPE_PICKS_S, i=i, n=n_jogos,
            )
            break
        try:
            # Sempre clica no primeiro disponível: após abrir/fechar um modal,
            # esse botão pode trocar de texto ("Ocultar"), fazendo nth(i) apontar
            # para o item errado. Usar .first garante o próximo não-processado.
            botao = page.get_by_text(_TEXTO_MOSTRAR, exact=False).first
            if not botao.count():
                logger.debug("Sem mais jogos com 'Mostrar todos os palpites'")
                break

            # Identifica o jogo pelo contexto do botão (sobe até o card)
            jogo_id, time_casa, time_fora = _extrair_nomes_jogo(botao, i)

            clicar_dispensando(page, botao)
            page.wait_for_timeout(1_000)

            # Clica na tab da subliga alvo para filtrar picks
            tab = page.get_by_text(subleague_nome, exact=False)
            if tab.count():
                tab.first.click()
                page.wait_for_timeout(800)

            # Extrai os listitems de picks
            try:
                page.wait_for_selector("ul li", timeout=3_000)
            except Exception:
                logger.debug("Nenhum listitem para jogo {j}", j=jogo_id)
                _fechar_modal_picks(page)
                continue

            items = page.locator("ul li").all()
            logger.debug(
                "Jogo {j}: {n} listitems encontrados",
                j=jogo_id,
                n=len(items),
            )
            picks: dict[str, tuple[int, int]] = {}
            pontos: dict[str, int] = {}
            for item in items:
                texto = item.inner_text().strip()
                m = _RE_PICK.match(texto)
                if m:
                    gc, gf = int(m.group(1)), int(m.group(2))
                    nome = m.group(4).strip()
                    picks[nome] = (gc, gf)
                    pontos[nome] = int(m.group(3))

            assinatura = frozenset(picks.items())
            if picks and assinatura not in picks_vistos:
                picks_vistos.add(assinatura)
                palpites.append(PalpiteJogo(
                    jogo_id=jogo_id,
                    time_casa=time_casa,
                    time_fora=time_fora,
                    picks=picks,
                    pontos=pontos,
                ))
                logger.debug(
                    "Picks | {j}: {n} participantes",
                    j=jogo_id,
                    n=len(picks),
                )
            elif picks:
                logger.debug(
                    "Jogo {j} ignorado: picks idênticos a um já capturado (modal não atualizou)",
                    j=jogo_id,
                )

            _fechar_modal_picks(page)

        except Exception as e:
            logger.warning("Erro ao raspar picks do jogo {i}: {e}", i=i, e=e)
            _fechar_modal_picks(page)

    return palpites


# Linhas de status que aparecem nos cards mas não são nomes de times
_STATUS_IGNORAR = frozenset({
    "encerrado", "encerrada", "em andamento", "finalizado", "não começou",
    "mostrar todos os palpites", "ocultar todos os palpites",
})


def _primeira_linha_texto(texto: str) -> str:
    """Retorna a primeira linha que parece ser um nome de time.

    Filtra em cascata: linhas vazias, só dígitos/horários/datas (ex: "2", "+3",
    "16:00"), linhas que começam com não-letra (emoji de status como "CRAVOU!"),
    e rótulos de status conhecidos ("Encerrado", etc.).
    """
    for linha in texto.strip().split("\n"):
        linha = linha.strip()
        if not linha:
            continue
        if re.match(r"^[+\-]?[\d:./]+$", linha):
            continue
        if not re.match(r"^[a-zA-ZÀ-ÿ]", linha):
            continue  # começa com emoji, símbolo ou dígito coberto pela regra anterior
        if linha.lower() in _STATUS_IGNORAR:
            continue
        return linha[:30]
    return ""


def _extrair_nomes_jogo(
    botao: Locator,
    fallback_idx: int,
) -> tuple[str, str, str]:
    """Sobe a árvore do DOM do botão para achar os nomes dos times no card.

    Retorna (jogo_id, time_casa, time_fora); usa o índice como ID se falhar. O
    inner_text() do card pode incluir o placar (ex: "2\\nx\\n0"), então cada
    candidato é filtrado para ignorar linhas puramente numéricas.
    """
    try:
        node = botao
        for _ in range(8):
            node = node.locator("xpath=..")
            texto = node.inner_text()
            partes = re.split(r"\s+[xX×]\s+", texto.strip())
            if len(partes) >= 2:
                casa = _primeira_linha_texto(partes[0])
                fora = _primeira_linha_texto(partes[1])
                if casa and fora and casa != fora:
                    return f"{casa} x {fora}", casa, fora
    except Exception:
        pass
    idx = str(fallback_idx)
    return f"jogo_{idx}", f"casa_{idx}", f"fora_{idx}"


def _fechar_modal_picks(page: Page) -> None:
    """Fecha o modal de picks clicando fora ou em botão de fechar."""
    try:
        fechar = page.get_by_role("button", name="Fechar")
        if fechar.count() and fechar.first.is_visible():
            fechar.first.click()
            page.wait_for_timeout(500)
            return
    except Exception:
        pass
    try:
        # Clica fora do modal
        page.keyboard.press("Escape")
        page.wait_for_timeout(300)
    except Exception:
        pass


def _int_seguro(texto: str) -> int:
    """Converte texto para int, ignorando sinais e espaços. Retorna 0 em falha."""
    try:
        return int(re.sub(r"[^\d]", "", texto) or "0")
    except ValueError:
        return 0
