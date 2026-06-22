"""Alertas Telegram via daemon cc, com fallback em arquivo se o cc estiver fora."""
from __future__ import annotations

import json
from datetime import datetime

import httpx
from loguru import logger
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from bolao.cache import CACHE_DIR
from bolao.config import PROJECT_TAG, SUBLEAGUE_NOME, URL_BASE
from bolao.modelo import Predicao

# arquivo de fallback se cc estiver fora
_FALLBACK_LOG = CACHE_DIR / "falhas.log"

# mantém só as últimas N linhas do fallback, pra não crescer sem limite
_FALLBACK_MAX_LINHAS = 200

_TIMEOUT_CC = 15.0


def notificar_sucesso(
    cc_api_url: str,
    cc_token: str,
    time_casa: str,
    time_fora: str,
    kickoff_brt: datetime,
    predicao: Predicao,
    estrategia: str = "max-EV",
    lambda_estimado: bool = False,
) -> None:
    """Confirmação da aposta com a análise completa (probabilidades, top 5, estratégia).

    Com lambda_estimado=True, avisa que o volume de gols caiu no default
    histórico por falta de mercado de totals.
    """
    horario = kickoff_brt.strftime("%d/%m às %H:%M")
    aposta = predicao.melhor_placar

    top5 = predicao.placares[:5]
    linhas_placares = "\n".join(
        f"  {i + 1}. {p.label}  {p.probabilidade:.1%}  E={p.pontos_esperados:.2f} pts"
        for i, p in enumerate(top5)
    )
    aviso_lambda = (
        "\n⚠ lambda_total estimado (sem mercado de totals), volume de gols é genérico\n"
        if lambda_estimado
        else ""
    )
    aviso_tj = ""
    if predicao.aposta_ev is not None:
        ev = predicao.aposta_ev
        aviso_tj = (
            f"\n💡 Teoria dos Jogos ativa\n"
            f"   Max-EV seria {ev.label} ({ev.probabilidade:.1%}), "
            f"mas {aposta.label} maximiza a vantagem relativa sobre os adversários.\n"
        )

    mensagem = (
        f"Feito. Palpite do bolão registrado.\n\n"
        f"⚽ {time_casa} x {time_fora}, {horario} BRT\n"
        f"🎯 {aposta.label}  {aposta.probabilidade:.1%}  E={aposta.pontos_esperados:.2f} pts  | {predicao.confianca}\n"
        f"   Estratégia: {estrategia}\n"
        f"{aviso_tj}"
        f"{aviso_lambda}\n"
        f"Probabilidades\n"
        f"  {time_casa}: {predicao.prob_casa:.1%}\n"
        f"  Empate:      {predicao.prob_empate:.1%}\n"
        f"  {time_fora}: {predicao.prob_fora:.1%}\n"
        f"  gols esperados: {predicao.lambda_casa:.1f} x {predicao.lambda_fora:.1f}\n\n"
        f"Top 5 placares\n{linhas_placares}"
    )

    _enviar_cc(cc_api_url, cc_token, mensagem, urgente=False)


def notificar_novos_jogos(
    cc_api_url: str,
    cc_token: str,
    jogos: list[tuple[str, str, str]],
) -> None:
    """Avisa quando novos jogos foram detectados no re-scan periódico."""
    n = len(jogos)
    linhas = "\n".join(f"  ⚽ {c} x {f}, {k}" for c, f, k in jogos)

    mensagem = f"Novos jogos entraram na agenda do bolão.\n\n{n} jogo(s) detectado(s):\n{linhas}"

    _enviar_cc(cc_api_url, cc_token, mensagem, urgente=False)


def notificar_falha(
    cc_api_url: str,
    cc_token: str,
    time_casa: str,
    time_fora: str,
    kickoff_brt: datetime,
    predicao: Predicao | None,
    motivo: str,
) -> None:
    """Falha definitiva de aposta. Urgente: o usuário precisa apostar à mão."""
    horario = kickoff_brt.strftime("%d/%m às %H:%M")
    url_bolao = URL_BASE

    if predicao:
        aposta = predicao.melhor_placar
        # mostra E[pontos] em cada linha, senão o palpite parece contradizer o
        # ranking de probabilidades (a aposta maximiza pontos esperados no
        # sistema 3/1, não a chance de cravar)
        linhas_placares = "\n".join(
            f"  {i + 1}. {p.label}  ({p.probabilidade:.1%} | E={p.pontos_esperados:.2f} pts)"
            for i, p in enumerate(predicao.placares[:5])
        )
        detalhes = (
            f"\n\nProbabilidades\n"
            f"  {time_casa}: {predicao.prob_casa:.1%}\n"
            f"  Empate:      {predicao.prob_empate:.1%}\n"
            f"  {time_fora}: {predicao.prob_fora:.1%}\n\n"
            f"Top 5:\n{linhas_placares}\n\n"
            f"Eu apostaria: {aposta.label} "
            f"({aposta.probabilidade:.1%} | E={aposta.pontos_esperados:.2f} pts)\n"
            f"O palpite maximiza pontos esperados (3 cravada / 1 resultado), "
            f"não o placar mais provável."
        )
    else:
        detalhes = "\n\nNão foi possível calcular nem o palpite."

    mensagem = (
        f"Não consegui apostar no bolão. Desta vez você precisa agir.\n\n"
        f"⚽ {time_casa} x {time_fora}\n"
        f"🕐 {horario} BRT\n"
        f"Motivo: {motivo}"
        f"{detalhes}\n\n"
        f"👉 {url_bolao}"
    )

    _enviar_cc(cc_api_url, cc_token, mensagem)


def notificar_palpite_atualizado(
    cc_api_url: str,
    cc_token: str,
    time_casa: str,
    time_fora: str,
    gols_casa_anterior: int,
    gols_fora_anterior: int,
    predicao: Predicao,
    min_antes_kickoff: int,
) -> None:
    """Avisa que o palpite foi alterado com sucesso num checkpoint de refinamento."""
    aposta = predicao.melhor_placar
    nota_tj = ""
    if predicao.aposta_ev is not None:
        nota_tj = f"\n💡 TJ diverge do max-EV ({predicao.aposta_ev.label}), pick diferenciado mantido."
    mensagem = (
        f"O cenário mudou. Atualizei o palpite no bolão.\n\n"
        f"⚽ {time_casa} x {time_fora}  |  T-{min_antes_kickoff}min\n\n"
        f"{gols_casa_anterior} x {gols_fora_anterior} -> {aposta.label}"
        f"  ({aposta.probabilidade:.1%}, E={aposta.pontos_esperados:.2f} pts, {predicao.confianca})"
        f"{nota_tj}"
    )
    _enviar_cc(cc_api_url, cc_token, mensagem, urgente=False)


def notificar_acao_manual_necessaria(
    cc_api_url: str,
    cc_token: str,
    time_casa: str,
    time_fora: str,
    gols_casa: int,
    gols_fora: int,
    min_antes_kickoff: int,
    motivo: str,
) -> None:
    """Urgente: falhou ao registrar/alterar palpite no refinamento. Pede ação manual."""
    url_bolao = URL_BASE
    mensagem = (
        f"Precisei de você agora, bolão. Não consegui alterar o palpite.\n\n"
        f"⚽ {time_casa} x {time_fora}  |  T-{min_antes_kickoff}min\n"
        f"✏️ Registre: {gols_casa} x {gols_fora}\n\n"
        f"Motivo: {motivo}\n"
        f"👉 {url_bolao}"
    )
    _enviar_cc(cc_api_url, cc_token, mensagem, urgente=True)


def notificar_recuperacao_janela(
    cc_api_url: str,
    cc_token: str,
    time_casa: str,
    time_fora: str,
    checkpoint_min: int,
) -> None:
    """Avisa que o T-60 foi perdido por restart tardio, mas já há nova tentativa agendada."""
    mensagem = (
        f"Acordei tarde para o bolão. Perdi a janela T-60.\n\n"
        f"⚽ {time_casa} x {time_fora}\n"
        f"Vou tentar de novo em T-{checkpoint_min}min."
    )
    _enviar_cc(cc_api_url, cc_token, mensagem, urgente=False)


def notificar_creditos_baixos(
    cc_api_url: str,
    cc_token: str,
    restantes: int,
    limite: int,
) -> None:
    """Urgente: créditos da The Odds API acabando. Sem eles as apostas param."""
    mensagem = (
        f"Estou ficando cega, bolão. Créditos da The Odds API quase no fim.\n\n"
        f"📉 Restantes: {restantes} (limite: {limite})\n\n"
        f"Crie uma chave nova em https://the-odds-api.com,\n"
        f"troque ODDS_API_KEY no .env e reinicie.\n"
        f"Sem odds, as apostas param."
    )
    _enviar_cc(cc_api_url, cc_token, mensagem, urgente=True)


def notificar_aposta_ousada(
    cc_api_url: str,
    cc_token: str,
    time_casa: str,
    time_fora: str,
    pick_ousado: str,
    pick_provavel: str,
    estrategia: str,
    prob_campeao_ousado: float,
    prob_campeao_provavel: float,
    minha_posicao: int,
    lider: str,
    pts_diferenca: float,
) -> None:
    """FYI: a teoria dos jogos recomenda sair do placar mais provável. Não exige
    ação; a aposta automática segue o max-EV fora da reta final.
    """
    ganho = (prob_campeao_ousado - prob_campeao_provavel) * 100
    url_bolao = URL_BASE

    mensagem = (
        f"Teoria dos jogos recomenda uma aposta diferente.\n\n"
        f"⚽ {time_casa} x {time_fora}\n\n"
        f"🎯 Pick ousado: {pick_ousado}\n"
        f"   (em vez de {pick_provavel}, que é o mais provável)\n\n"
        f"Estratégia: {estrategia}\n"
        f"P(campeão) com ousado: {prob_campeao_ousado:.1%}"
        f" vs {prob_campeao_provavel:.1%} com o modal"
        f" (+{ganho:.1f} pp)\n\n"
        f"Você está em {minha_posicao}º ({pts_diferenca:+.0f} pts de {lider})\n\n"
        f"Se quiser apostar ousado, entre no site antes do kickoff:\n"
        f"👉 {url_bolao}"
    )

    _enviar_cc(cc_api_url, cc_token, mensagem, urgente=False)


def notificar_ranking_atualizado(
    cc_api_url: str,
    cc_token: str,
    classificacao: list[tuple[str, int, int, int]],
    minha_posicao: int,
    proximo_jogo: str | None,
    estrategia_recomendada: str | None,
) -> None:
    """Envia o ranking atualizado da subliga depois que o site processa o resultado."""
    linhas = []
    for i, (nome, _ap, _av, pts) in enumerate(classificacao[:5], 1):
        marcador = "->" if i == minha_posicao else "  "
        linhas.append(f"{marcador} {i}. {nome}: {pts} pts")

    corpo = "\n".join(linhas)

    extras = ""
    if proximo_jogo and estrategia_recomendada:
        extras = (
            f"\n\nPróximo jogo: {proximo_jogo}\n"
            f"Estratégia recomendada: {estrategia_recomendada}"
        )

    mensagem = f"Classificação {SUBLEAGUE_NOME} atualizada.\n\n{corpo}{extras}"

    _enviar_cc(cc_api_url, cc_token, mensagem, urgente=False)


def notificar_daemon_no_ar(
    cc_api_url: str,
    cc_token: str,
    n_jogos: int,
    proxima_aposta: str | None,
    recuperados: int,
    creditos: int | None,
) -> None:
    """Par da mensagem "daemon caiu" do systemd: confirma que o serviço voltou
    e está operante, sem o usuário precisar abrir terminal.
    """
    linhas = [f"Voltei. Daemon do bolão no ar.\n\n📅 {n_jogos} jogo(s) na agenda"]
    if proxima_aposta:
        linhas.append(f"⏰ Próxima aposta: {proxima_aposta}")
    if recuperados:
        linhas.append(f"♻️ {recuperados} job(s) recuperado(s) do cache")
    if creditos is not None:
        linhas.append(f"🪙 Créditos The Odds API: {creditos}")
    _enviar_cc(cc_api_url, cc_token, "\n".join(linhas), urgente=False)


def notificar_reinicio_em_loop(
    cc_api_url: str,
    cc_token: str,
    reinicios: int,
    janela_min: int,
) -> None:
    """Urgente: o daemon está caindo e subindo repetidamente. O restart
    automático do systemd não está resolvendo, precisa de intervenção.
    """
    mensagem = (
        f"Estou reiniciando em loop: {reinicios} vezes em {janela_min} min.\n\n"
        f"O restart automático não está resolvendo, algo quebrou de verdade.\n"
        f"As apostas podem parar até alguém olhar o servidor."
    )
    _enviar_cc(cc_api_url, cc_token, mensagem, urgente=True)


def notificar_resumo_diario(
    cc_api_url: str,
    cc_token: str,
    jogos_hoje: list[tuple[str, str, str, str]],
    minha_posicao: int | None,
    meus_pontos: int | None,
    creditos: int | None,
) -> None:
    """Bom-dia diário com a agenda. Também é o heartbeat do sistema: se esta
    mensagem não chegar num dia, algo morreu, mesmo sem o daemon ter caído.
    """
    if jogos_hoje:
        linhas = "\n".join(
            f"  ⚽ {c} x {f}, {k} (aposto às {a})"
            for c, f, k, a in jogos_hoje
        )
        corpo = f"Hoje tem bolão. {len(jogos_hoje)} jogo(s):\n{linhas}"
    else:
        corpo = "Sem jogos hoje. Sigo de plantão."

    extras = []
    if minha_posicao is not None and meus_pontos is not None:
        extras.append(f"📊 Você: {minha_posicao}º no {SUBLEAGUE_NOME} com {meus_pontos} pts")
    if creditos is not None:
        extras.append(f"🪙 Créditos The Odds API: {creditos}")

    mensagem = corpo + ("\n\n" + "\n".join(extras) if extras else "")
    _enviar_cc(cc_api_url, cc_token, mensagem, urgente=False)


def notificar_scrape_falhou(
    cc_api_url: str,
    cc_token: str,
    tipo: str,
    motivo: str,
) -> None:
    """O scrape do site falhou (login quebrou, layout mudou). Sem ele o ranking
    não chega e os perfis dos adversários envelhecem.
    """
    mensagem = (
        f"Não consegui raspar {tipo} do site.\n\n"
        f"Motivo: {motivo}\n\n"
        f"O ranking e os perfis dos adversários podem ficar desatualizados. "
        f"Se repetir, o site deve ter mudado."
    )
    _enviar_cc(cc_api_url, cc_token, mensagem, urgente=False)


def notificar_placar_nao_confirmado(
    cc_api_url: str,
    cc_token: str,
    time_casa: str,
    time_fora: str,
) -> None:
    """A The Odds API não devolveu o placar dentro da janela (~3h pós-jogo).
    Sem isso o resultado nunca chegaria e o usuário ficaria esperando.
    """
    mensagem = (
        f"Não consegui confirmar o placar de {time_casa} x {time_fora}.\n\n"
        f"A API de resultados não retornou o jogo. "
        f"Confira no site quantos pontos fez:\n"
        f"👉 {URL_BASE}"
    )
    _enviar_cc(cc_api_url, cc_token, mensagem, urgente=False)


def _deve_retry(exc: BaseException) -> bool:
    """Retry só vale a pena para erros transitórios: rede ou 5xx (não 4xx)."""
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return False


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=0.5, max=4),
    retry=retry_if_exception(_deve_retry),
    reraise=True,
)
def _post_cc(cc_api_url: str, cc_token: str, payload: dict) -> None:
    """POST ao /notify do cc, com retry exponencial+jitter em falha transitória."""
    resp = httpx.post(
        f"{cc_api_url}/notify",
        headers={"Authorization": f"Bearer {cc_token}"},
        json=payload,
        timeout=_TIMEOUT_CC,
    )
    resp.raise_for_status()


def _enviar_cc(
    cc_api_url: str,
    cc_token: str,
    mensagem: str,
    urgente: bool = True,
) -> None:
    """Entrega a notificação via cc; em falha grava em arquivo de fallback local."""
    payload = {"project": PROJECT_TAG, "message": mensagem, "urgente": urgente}

    try:
        _post_cc(cc_api_url, cc_token, payload)
        logger.info("Notificação enviada via cc | urgente={u}", u=urgente)
        return
    except httpx.HTTPStatusError as e:
        logger.error("cc retornou {s}: {b}", s=e.response.status_code, b=e.response.text[:200])
    except httpx.RequestError as e:
        logger.error("cc indisponível após retries: {e}", e=e)

    _gravar_fallback(mensagem)


def _gravar_fallback(mensagem: str) -> None:
    """Grava a notificação não-entregue em arquivo, mantendo só as últimas N linhas."""
    try:
        _FALLBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
        linha = json.dumps(
            {"ts": datetime.now().isoformat(), "msg": mensagem}, ensure_ascii=False
        )
        anteriores: list[str] = []
        if _FALLBACK_LOG.exists():
            anteriores = _FALLBACK_LOG.read_text(encoding="utf-8").splitlines()
        recentes = [*anteriores, linha][-_FALLBACK_MAX_LINHAS:]
        _FALLBACK_LOG.write_text("\n".join(recentes) + "\n", encoding="utf-8")
        logger.warning("Notificação gravada em fallback local: {p}", p=_FALLBACK_LOG)
    except Exception as e:
        logger.critical("Falhou até o fallback local: {e}", e=e)
