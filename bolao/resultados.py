"""Rastreamento de apostas e verificação de resultados pós-jogo.

Persiste o palpite, consulta o placar na The Odds API, compara (+3/+1/0) e
notifica o resultado via cc.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import httpx
from loguru import logger

from bolao.cache import cache_chaves, cache_get, cache_set
from bolao.config import ODDS_SPORT as _SPORT
from bolao.config import SUBLEAGUE_NOME
from bolao.fontes.football_data import buscar_placar as buscar_placar_football_data
from bolao.nomes import nome_presente
from bolao.notificador import _enviar_cc

if TYPE_CHECKING:
    from bolao.fontes.site import DadosBolao

_BASE = "https://api.the-odds-api.com/v4"
_CACHE_KEY_PREFIX = "aposta"
_TTL_APOSTA = 10 * 24 * 3600  # 10 dias, cobre re-agendamentos e prorrogações

# marca jogos cujo resultado já foi notificado, para um restart do daemon não
# reenviar a mesma mensagem
_CHAVE_NOTIFICADO = "resultado:notificado"


def resultado_ja_notificado(jogo_id: str) -> bool:
    """True se o resultado deste jogo já foi enviado (sobrevive a restart)."""
    return cache_get(f"{_CHAVE_NOTIFICADO}:{jogo_id}") is not None


def marcar_resultado_notificado(jogo_id: str) -> None:
    cache_set(f"{_CHAVE_NOTIFICADO}:{jogo_id}", True, ttl=_TTL_APOSTA)


@dataclass
class ApostaRegistrada:
    """Dados persistidos de uma aposta para check de resultado posterior."""

    jogo_id: str
    time_casa: str
    time_fora: str
    kickoff_utc: datetime
    kickoff_brt_str: str
    gols_apostados_casa: int
    gols_apostados_fora: int


def registrar_aposta(
    jogo_id: str,
    time_casa: str,
    time_fora: str,
    kickoff_utc: datetime,
    kickoff_brt_str: str,
    gols_casa: int,
    gols_fora: int,
) -> None:
    """Persiste o palpite em diskcache para check de resultado posterior."""
    aposta = ApostaRegistrada(
        jogo_id=jogo_id,
        time_casa=time_casa,
        time_fora=time_fora,
        kickoff_utc=kickoff_utc,
        kickoff_brt_str=kickoff_brt_str,
        gols_apostados_casa=gols_casa,
        gols_apostados_fora=gols_fora,
    )
    cache_set(f"{_CACHE_KEY_PREFIX}:{jogo_id}", aposta, ttl=_TTL_APOSTA)
    logger.info(
        "Aposta registrada | {c} x {f} | palpite: {gc} x {gf}",
        c=time_casa,
        f=time_fora,
        gc=gols_casa,
        gf=gols_fora,
    )


def listar_apostas_pendentes() -> list[ApostaRegistrada]:
    """
    Retorna todas as apostas persistidas em diskcache.
    Usado no startup para reagendar checks de resultado perdidos num restart.
    """
    apostas = []
    for chave in cache_chaves(f"{_CACHE_KEY_PREFIX}:"):
        aposta = cache_get(chave)
        if isinstance(aposta, ApostaRegistrada):
            apostas.append(aposta)
    return apostas


def buscar_placar_real(
    api_key: str,
    jogo_id: str,
    time_casa: str | None = None,
    time_fora: str | None = None,
    football_data_token: str | None = None,
) -> tuple[int, int] | None:
    """Placar final de um jogo, com redundância de fontes.

    Tenta a The Odds API primeiro; se não devolver (jogo fora da janela, score
    incompleto) e houver token + nomes dos times, cai para o football-data.org.
    A 2ª fonte é opcional: sem `football_data_token`, busca só na primeira.
    """
    placar = _placar_odds_api(api_key, jogo_id)
    if placar is not None:
        return placar
    if football_data_token and time_casa and time_fora:
        fd = buscar_placar_football_data(football_data_token, time_casa, time_fora)
        if fd is not None:
            logger.info("Placar obtido via football-data (fallback): {p}", p=fd)
            return fd
    return None


def _placar_odds_api(api_key: str, jogo_id: str) -> tuple[int, int] | None:
    """
    Consulta o endpoint de scores da The Odds API.
    Retorna (gols_casa, gols_fora) ou None se o jogo ainda não terminou
    ou não foi encontrado na janela de daysFrom=2.
    """
    try:
        resp = httpx.get(
            f"{_BASE}/sports/{_SPORT}/scores",
            params={"apiKey": api_key, "daysFrom": 2},
            timeout=10.0,
        )
        resp.raise_for_status()

        for jogo in resp.json():
            if jogo.get("id") != jogo_id:
                continue

            if not jogo.get("completed", False):
                logger.debug("Jogo {id} ainda em andamento", id=jogo_id)
                return None

            scores = jogo.get("scores") or []
            placar = {s["name"]: s["score"] for s in scores}
            home = jogo.get("home_team", "")
            away = jogo.get("away_team", "")

            if home in placar and away in placar:
                return int(placar[home]), int(placar[away])

            logger.warning("Scores incompletos para jogo {id}: {s}", id=jogo_id, s=scores)
            return None

        logger.debug("Jogo {id} não encontrado no endpoint de scores", id=jogo_id)
        return None

    except httpx.HTTPStatusError as e:
        logger.warning(
            "Erro HTTP ao buscar placar: {s} {b}",
            s=e.response.status_code,
            b=e.response.text[:200],
        )
        return None
    except (httpx.RequestError, ValueError, KeyError, TypeError) as e:
        logger.warning("Erro ao buscar placar real: {e}", e=e)
        return None


def calcular_pontos(
    apostado_casa: int,
    apostado_fora: int,
    real_casa: int,
    real_fora: int,
) -> tuple[int, str]:
    """
    Calcula pontos ganhos no sistema do bolão.
    Retorna (pontos, tipo) onde tipo é "cravada", "resultado" ou "erro".
    """
    if apostado_casa == real_casa and apostado_fora == real_fora:
        return 3, "cravada"

    def _resultado(c: int, f: int) -> str:
        if c > f:
            return "casa"
        if c < f:
            return "fora"
        return "empate"

    if _resultado(apostado_casa, apostado_fora) == _resultado(real_casa, real_fora):
        return 1, "resultado"

    return 0, "erro"


@dataclass
class RankingAoVivo:
    """Posição na subliga calculada em tempo real dos picks públicos + resultados."""

    minha_posicao: int       # 1-based
    total_membros: int
    meus_pontos: int
    lider_nome: str
    lider_pontos: int
    empatados_no_topo: int   # quantos dividem o 1º lugar


def calcular_ranking_ao_vivo(
    dados: DadosBolao,
    meu_nome: str,
    jogo_atual_casa: str | None = None,
    jogo_atual_fora: str | None = None,
    resultado_atual: tuple[int, int] | None = None,
    membros_filtro: set[str] | None = None,
) -> RankingAoVivo | None:
    """
    Classificação da subliga recalculada na hora, sem depender do site (que só
    atualiza ~2x/dia). Soma os pontos de cada membro a partir dos picks públicos
    raspados e do placar real de cada jogo (o pick que cravou define o placar;
    para o jogo recém-encerrado usa-se `resultado_atual`, que o site pode ainda
    não ter processado).

    Os membros vêm da classificação raspada; `membros_filtro` (BOLAO_MEMBROS)
    restringe aos membros da subliga quando o scraper raspa a liga inteira.
    Retorna None se faltar dado essencial.
    """
    # Nome resolvido no fetch (por user_id) auto-segue troca de nome; fallback no arg.
    meu_nome = dados.meu_nome or meu_nome

    membros = [nome for nome, _ap, _av, _pts in dados.classificacao]
    if membros_filtro:
        membros = [n for n in membros if n in membros_filtro]
    if not membros:
        return None

    meu = next((m for m in membros if m.lower() == meu_nome.lower()), None)
    if meu is None:
        # heurística do scraper: e-mail começa com o primeiro nome do jogador
        meu = next(
            (m for m in membros if meu_nome.lower().startswith(m.split()[0].lower())),
            None,
        )
    if meu is None:
        return None

    pontos = dict.fromkeys(membros, 0)
    for pj in dados.palpites_por_jogo:
        placar = pj.resultado_real
        if (
            placar is None
            and resultado_atual is not None
            and jogo_atual_casa is not None
            and jogo_atual_fora is not None
            and nome_presente(jogo_atual_casa, pj.time_casa)
            and nome_presente(jogo_atual_fora, pj.time_fora)
        ):
            placar = resultado_atual
        if placar is None:
            continue  # jogo sem placar conhecido ainda, não conta para ninguém
        for membro in membros:
            pick = pj.picks.get(membro)
            if pick is not None:
                pts, _ = calcular_pontos(pick[0], pick[1], placar[0], placar[1])
                pontos[membro] += pts

    ordenados = sorted(membros, key=lambda m: pontos[m], reverse=True)
    lider = ordenados[0]
    lider_pts = pontos[lider]
    meus_pts = pontos[meu]
    # Posição = 1 + nº de membros com estritamente mais pontos (empates dividem)
    posicao = 1 + sum(1 for m in membros if pontos[m] > meus_pts)
    empatados_topo = sum(1 for m in membros if pontos[m] == lider_pts)

    return RankingAoVivo(
        minha_posicao=posicao,
        total_membros=len(membros),
        meus_pontos=meus_pts,
        lider_nome=lider,
        lider_pontos=lider_pts,
        empatados_no_topo=empatados_topo,
    )


def _linha_ranking(r: RankingAoVivo) -> str:
    """Linha de classificação ao vivo da subliga para anexar às mensagens."""
    if r.minha_posicao == 1 and r.empatados_no_topo == 1:
        return f"🏆 {SUBLEAGUE_NOME} ao vivo: 1º de {r.total_membros} ({r.meus_pontos} pts), liderando!"
    if r.minha_posicao == 1:
        return (
            f"🏆 {SUBLEAGUE_NOME} ao vivo: 1º de {r.total_membros} ({r.meus_pontos} pts), "
            f"empatado com +{r.empatados_no_topo - 1}"
        )
    gap = r.lider_pontos - r.meus_pontos
    return (
        f"📊 {SUBLEAGUE_NOME} ao vivo: {r.minha_posicao}º de {r.total_membros} "
        f"({r.meus_pontos} pts), líder {r.lider_nome} {r.lider_pontos} (-{gap})"
    )


def notificar_resultado(
    cc_api_url: str,
    cc_token: str,
    time_casa: str,
    time_fora: str,
    kickoff_brt_str: str,
    apostado_casa: int,
    apostado_fora: int,
    real_casa: int,
    real_fora: int,
    pontos_acumulados: int | None = None,
    ranking: RankingAoVivo | None = None,
) -> None:
    """Monta e envia notificação de resultado após o jogo encerrar."""
    pontos, tipo = calcular_pontos(apostado_casa, apostado_fora, real_casa, real_fora)
    palpite = f"{apostado_casa} x {apostado_fora}"

    if tipo == "cravada":
        cabecalho = "Cravei. +3 pts no bolão."
        detalhe = f"🎯 Apostei: {palpite} (exato)"
    elif tipo == "resultado":
        cabecalho = "Acertei o resultado. +1 pt no bolão."
        detalhe = f"Apostei: {palpite} (placar errado, mas lado certo)"
    else:
        cabecalho = "Errei. 0 pts no bolão."
        detalhe = f"Apostei: {palpite}"

    mensagem = (
        f"{cabecalho}\n\n"
        f"⚽ {time_casa} {real_casa} x {real_fora} {time_fora}\n"
        f"{detalhe}"
    )
    if pontos_acumulados is not None:
        mensagem += f"\n\n📊 Total das apostas automáticas: {pontos_acumulados} pts"
    if ranking is not None:
        mensagem += f"\n{_linha_ranking(ranking)}"

    logger.info(
        "Resultado | {c} x {f} | real={rc}x{rf} | apostado={ac}x{af} | {t} ({p} pts)",
        c=time_casa,
        f=time_fora,
        rc=real_casa,
        rf=real_fora,
        ac=apostado_casa,
        af=apostado_fora,
        t=tipo,
        p=pontos,
    )

    _enviar_cc(cc_api_url, cc_token, mensagem, urgente=False)


def notificar_resultado_sem_aposta(
    cc_api_url: str,
    cc_token: str,
    time_casa: str,
    time_fora: str,
    real_casa: int,
    real_fora: int,
) -> None:
    """Resultado de jogo cuja aposta automática falhou definitivamente.

    Não há palpite no cache para calcular pontos, então informa só o placar e
    lembra que uma eventual aposta manual deve ser conferida no site.
    """
    mensagem = (
        f"Jogo encerrado.\n\n"
        f"⚽ {time_casa} {real_casa} x {real_fora} {time_fora}\n\n"
        f"Não tenho aposta automática registrada para este jogo. "
        f"Se você apostou manualmente, confira seus pontos no site."
    )

    logger.info(
        "Resultado sem aposta | {c} x {f} | real={rc}x{rf}",
        c=time_casa,
        f=time_fora,
        rc=real_casa,
        rf=real_fora,
    )

    _enviar_cc(cc_api_url, cc_token, mensagem, urgente=False)


@dataclass
class ItemResumo:
    """Uma linha do resumo de resultados (um jogo já encerrado)."""

    time_casa: str
    time_fora: str
    real_casa: int
    real_fora: int
    apostado_casa: int
    apostado_fora: int
    pontos: int


def notificar_resumo_resultados(
    cc_api_url: str,
    cc_token: str,
    itens: list[ItemResumo],
    pontos_acumulados: int | None = None,
    ranking: RankingAoVivo | None = None,
) -> None:
    """Uma única mensagem-tabela com vários resultados de uma vez (recuperação
    pós-restart, em vez de uma mensagem por jogo)."""
    if not itens:
        return

    linhas = []
    for it in itens:
        emoji = "🎯" if it.pontos == 3 else ("✅" if it.pontos == 1 else "❌")
        linhas.append(
            f"{emoji} {it.time_casa} {it.real_casa}x{it.real_fora} {it.time_fora}"
            f" | apostei {it.apostado_casa}x{it.apostado_fora} (+{it.pontos})"
        )

    ganhos = sum(it.pontos for it in itens)
    cabecalho = f"📋 Resumo: {len(itens)} jogo(s) encerrado(s) | +{ganhos} pts"
    mensagem = cabecalho + "\n\n" + "\n".join(linhas)
    if pontos_acumulados is not None:
        mensagem += f"\n\n📊 Total das apostas automáticas: {pontos_acumulados} pts"
    if ranking is not None:
        mensagem += f"\n{_linha_ranking(ranking)}"

    logger.info("Resumo de {n} resultado(s) | +{g} pts", n=len(itens), g=ganhos)
    _enviar_cc(cc_api_url, cc_token, mensagem, urgente=False)
