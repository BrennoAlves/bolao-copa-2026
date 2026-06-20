"""Daemon do bolão: agenda um job de aposta por jogo, prediz o placar, aposta no
site, notifica e checa o resultado. Reinicia sozinho via systemd."""
from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from typing import cast
from zoneinfo import ZoneInfo

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

from bolao.apostador import apostar
from bolao.backtest import registrar_predicao_backtest, registrar_resultado_backtest
from bolao.cache import cache_get, cache_set
from bolao.config import Config, carregar_config, validar_config_runtime
from bolao.diagnostico import limpar_diagnosticos_antigos
from bolao.fontes import ajustar_probabilidades, buscar_odds, buscar_probabilidades
from bolao.fontes.odds_api import CHAVE_CREDITOS, OddsJogo
from bolao.fontes.site import DadosBolao, carregar_dados_bolao, obter_dados_bolao
from bolao.modelo import Predicao, eh_mata_mata, predizer
from bolao.notificador import (
    notificar_acao_manual_necessaria,
    notificar_aposta_ousada,
    notificar_creditos_baixos,
    notificar_daemon_no_ar,
    notificar_falha,
    notificar_novos_jogos,
    notificar_palpite_atualizado,
    notificar_placar_nao_confirmado,
    notificar_ranking_atualizado,
    notificar_recuperacao_janela,
    notificar_reinicio_em_loop,
    notificar_resumo_diario,
    notificar_scrape_falhou,
    notificar_sucesso,
)
from bolao.resultados import (
    ApostaRegistrada,
    ItemResumo,
    RankingAoVivo,
    buscar_placar_real,
    calcular_pontos,
    calcular_ranking_ao_vivo,
    listar_apostas_pendentes,
    marcar_resultado_notificado,
    notificar_resultado,
    notificar_resultado_sem_aposta,
    notificar_resumo_resultados,
    registrar_aposta,
    resultado_ja_notificado,
)
from bolao.simulador import (
    Adversario,
    JogoSimulado,
    ResultadoSimulacao,
    estimar_perfis,
    simular,
)

# Buffer após o kickoff para checar o placar (90min jogo + intervalo + acréscimos)
_BUFFER_RESULTADO_MIN = 115


def _registrar_backtest_seguro(
    jogo: OddsJogo,
    prob_casa_modelo: float,
    prob_empate_modelo: float,
    prob_fora_modelo: float,
    predicao: Predicao,
) -> None:
    """Grava (ou sobrescreve) o registro de backtest do jogo. Em try/except para
    nunca bloquear o fluxo principal."""
    try:
        aposta = predicao.melhor_placar
        top10 = [(p.label, p.probabilidade) for p in predicao.placares]
        registrar_predicao_backtest(
            jogo_id=jogo.id,
            time_casa=jogo.time_casa,
            time_fora=jogo.time_fora,
            kickoff_utc=jogo.kickoff_utc,
            prob_casa_api=jogo.prob_casa,
            prob_empate_api=jogo.prob_empate,
            prob_fora_api=jogo.prob_fora,
            prob_casa_modelo=prob_casa_modelo,
            prob_empate_modelo=prob_empate_modelo,
            prob_fora_modelo=prob_fora_modelo,
            lambda_casa=predicao.lambda_casa,
            lambda_fora=predicao.lambda_fora,
            gols_apostados_casa=aposta.gols_casa,
            gols_apostados_fora=aposta.gols_fora,
            prob_apostada=aposta.probabilidade,
            pontos_esperados=aposta.pontos_esperados,
            top10_placares=top10,
        )
    except Exception as e:
        logger.warning("Backtest: falha ao gravar predição: {e}", e=e)

# jogos à frente simulados para decidir o pick do jogo atual. com 1 jogo só não
# há sinal de P(1o) quando se está atrás (não fecha o gap); o ganho da otimizada
# vem de simular uma janela.
_JANELA_OTIMIZACAO = 8


def _membros_filtro(config: Config, dados: DadosBolao) -> set[str] | None:
    """Filtro de membros da subliga para restringir a classificação raspada.

    Quando os dados vêm do HTTP, a classificação já é exatamente a subliga, então
    a lista manual de membros é dispensada. Só o scraping da liga inteira precisa
    do filtro.
    """
    if getattr(dados, "ja_filtrado_subleague", False):
        return None
    return set(config.bolao_membros)


def _meu_nome(config: Config) -> str:
    """Meu identificador para casar na classificação: o nome configurado (casa
    exato, sem confundir apelido/homônimo) ou o prefixo do e-mail. O casamento
    exato vem primeiro, então o nome cheio resolve quem sou eu."""
    return config.bolao_nome or config.bolao_email.split("@")[0]


def _adversarios_da_subliga(
    dados: DadosBolao, meu_nome: str, membros: set[str] | None
) -> tuple[float, list[Adversario]]:
    """Separa minha pontuação dos adversários, com perfis empíricos dos picks
    reais. `membros` restringe à subliga quando o scraper raspa a liga inteira.
    """
    # prioriza o nome resolvido no fetch (por user_id, auto-segue troca de nome);
    # cai no `meu_nome` derivado da config só se o fetch não resolveu.
    meu_nome = dados.meu_nome or meu_nome
    perfis = estimar_perfis(dados, meu_nome)

    # resolve meu nome canônico: o e-mail tem um prefixo (ex. "joao123") e a
    # classificação exibe o nome ("Joao"). sem o heurístico de prefixo eu seria
    # contado como adversário e meus_pontos ficaria 0 (P(1o) viraria lixo).
    nomes = [n for n, *_ in dados.classificacao if not membros or n in membros]
    eu = next((n for n in nomes if n.lower() == meu_nome.lower()), None) or next(
        (n for n in nomes if meu_nome.lower().startswith(n.split()[0].lower())), None
    )

    meus_pontos = 0.0
    adversarios: list[Adversario] = []
    for nome, _ap, _av, pts in dados.classificacao:
        if membros and nome not in membros:
            continue  # fora da subliga, ignora o resto da liga
        if nome == eu:
            meus_pontos = float(pts)
        else:
            adversarios.append(
                Adversario(nome=nome, pontos=float(pts), perfil=perfis.get(nome))
            )
    return meus_pontos, adversarios


def _simular_com_standings(
    dados: DadosBolao,
    predicao: Predicao,
    meu_nome: str,
    membros: set[str] | None = None,
) -> list[ResultadoSimulacao] | None:
    """Monte Carlo de um jogo com os standings reais, usado pela notificação FYI
    de aposta ousada (modo auto). Retorna None se faltar dado."""
    if len(predicao.placares) < 36:
        return None
    meus_pontos, adversarios = _adversarios_da_subliga(dados, meu_nome, membros)
    if not adversarios:
        return None
    try:
        jogo_simulado = JogoSimulado.de_predicao(predicao)
        return simular([jogo_simulado], meus_pontos, adversarios, n_sims=5_000)
    except Exception as e:
        logger.warning("Erro no Monte Carlo: {e}", e=e)
        return None


def _predizer_proximos(config: Config, excluir_id: str, n: int) -> list[JogoSimulado]:
    """Prediz os próximos `n` jogos (exceto `excluir_id`) para a janela de simulação."""
    if n <= 0:
        return []
    try:
        jogos = sorted(buscar_odds(config.odds_api_key), key=lambda j: j.kickoff_utc)
    except Exception as e:
        logger.warning("Falha ao buscar próximos jogos p/ janela: {e}", e=e)
        return []
    out: list[JogoSimulado] = []
    for j in jogos:
        if j.id == excluir_id:
            continue
        if len(out) >= n:
            break
        try:
            pred = predizer(
                time_casa=j.time_casa, time_fora=j.time_fora,
                prob_casa=j.prob_casa, prob_empate=j.prob_empate, prob_fora=j.prob_fora,
                lambda_total=j.lambda_total * config.lambda_fator,
                mata_mata=eh_mata_mata(j.kickoff_utc), top_n=36,
            )
            out.append(JogoSimulado.de_predicao(pred))
        except Exception:
            continue
    return out


# checkpoints de refinamento antes do kickoff (minutos). o primeiro (60) define
# o horário da aposta inicial; é fixo, não usa config.
_CHECKPOINTS_MIN = [60, 45, 30, 15, 5]

# alerta de créditos da The Odds API: o plano free tem 500/mês e o consumo
# típico (~5 checkpoints x 2 créditos por jogo) esgota em ~2 semanas
_LIMITE_ALERTA_CREDITOS = 100


def _executar_job(
    config: Config, jogo: OddsJogo, tentativa: int = 1, escalar_falha: bool = True
) -> None:
    """Job principal de cada jogo: busca odds, prediz o placar, aposta e notifica.
    Se falhar e ainda houver tentativas, agenda retry.

    escalar_falha=False (usado pela rotina de início do dia): uma falha não vira
    alerta urgente nem agenda retries. O job T-60 e os refinamentos ainda rodam
    perto do kickoff, então um tropeço cedo não assusta você.
    """
    tz_local = ZoneInfo(config.timezone)
    kickoff_local = jogo.kickoff_utc.astimezone(tz_local)

    logger.info(
        "Job iniciado | tentativa {t}/{max} | {c} x {f} | kickoff {k}",
        t=tentativa,
        max=config.max_retries,
        c=jogo.time_casa,
        f=jogo.time_fora,
        k=kickoff_local.strftime("%d/%m %H:%M %Z"),
    )

    # se já há aposta registrada (ex. feita pela rotina de início do dia), não
    # re-aposta do zero; delega ao refinamento, que só re-aposta se a predição
    # mudou. sem isto, o T-60 rodando depois do batch matinal geraria aposta e
    # "sucesso" duplicados.
    if cache_get(f"aposta:{jogo.id}") is not None:
        logger.info(
            "Aposta já registrada para {c} x {f}, delegando ao refinamento (T-60)",
            c=jogo.time_casa,
            f=jogo.time_fora,
        )
        _refinar_palpite_job(config, jogo, min_antes_kickoff=_CHECKPOINTS_MIN[0])
        return

    # odds frescas (ignora cache, reflete escalações já divulgadas)
    jogo = _atualizar_odds(config, jogo)

    try:
        poly = buscar_probabilidades(jogo.time_casa, jogo.time_fora)
        prob_casa, prob_empate, prob_fora = ajustar_probabilidades(
            jogo.prob_casa,
            jogo.prob_empate,
            jogo.prob_fora,
            poly,
        )
    except Exception as e:
        logger.warning("Erro ao buscar Polymarket, usando só The Odds API: {e}", e=e)
        prob_casa, prob_empate, prob_fora = jogo.prob_casa, jogo.prob_empate, jogo.prob_fora

    # palpites dos oponentes via Supabase
    try:
        from bolao.fontes.supabase import extrair_palpites_jogo
        palpites_oponentes = extrair_palpites_jogo(config, jogo.time_casa, jogo.time_fora)
    except Exception as e:
        logger.warning("Falha ao extrair palpites do Supabase: {e}", e=e)
        palpites_oponentes = []

    # predição (top_n=36 para viabilizar a simulação Monte Carlo)
    try:
        predicao = predizer(
            time_casa=jogo.time_casa,
            time_fora=jogo.time_fora,
            prob_casa=prob_casa,
            prob_empate=prob_empate,
            prob_fora=prob_fora,
            lambda_total=jogo.lambda_total * config.lambda_fator,
            mata_mata=eh_mata_mata(jogo.kickoff_utc),
            top_n=36,
            palpites_oponentes=palpites_oponentes,
        )
    except Exception:
        # stacktrace completo: um bug no modelo não pode passar mudo
        logger.exception("Erro na predição")
        predicao = None

    if jogo.lambda_estimado:
        logger.warning(
            "lambda_total estimado (sem mercado de totals) para {c} x {f}, volume de gols genérico",
            c=jogo.time_casa, f=jogo.time_fora,
        )

    if predicao is None:
        if not escalar_falha:
            logger.warning(
                "Início do dia: predição falhou para {c} x {f}, T-60 tentará de novo",
                c=jogo.time_casa, f=jogo.time_fora,
            )
            return
        _lidar_com_falha(
            config, jogo, kickoff_local, None,
            "Falha ao calcular predição de placar", tentativa,
        )
        return

    # política de aposta (objetivo: terminar em 1º na subliga).
    # até a reta final: max-EV (acumular pontos) e a sugestão ousada fica só
    # como notificação. na reta final: o pick que maximiza P(campeão) decide.
    estrategia = "max-EV"
    try:
        dados_bolao: DadosBolao | None = cache_get("bolao:dados_scraped")
        if dados_bolao is not None:
            if _modo_reta_final(config):
                estrategia = _aplicar_estrategia_campeao(
                    config, jogo, predicao, dados_bolao
                ) or estrategia
            else:
                _notificar_se_ousado(config, jogo, predicao, dados_bolao, kickoff_local)
    except Exception as e:
        logger.warning("Erro na política de aposta: {e}", e=e)

    # aposta no site. sem notificação pré-aposta: o sucesso traz a análise
    # completa e a falha definitiva já inclui a predição para aposta manual.
    resultado = apostar(
        email=config.bolao_email,
        password=config.bolao_password,
        subleague=config.bolao_subleague,
        predicao=predicao,
    )

    # sucesso
    if resultado.sucesso:
        notificar_sucesso(
            cc_api_url=config.cc_api_url,
            cc_token=config.cc_token,
            time_casa=jogo.time_casa,
            time_fora=jogo.time_fora,
            kickoff_brt=kickoff_local,
            predicao=predicao,
            estrategia=estrategia,
            lambda_estimado=jogo.lambda_estimado,
        )

        aposta = predicao.melhor_placar
        kickoff_brt_str = kickoff_local.strftime("%d/%m às %H:%M")

        registrar_aposta(
            jogo_id=jogo.id,
            time_casa=jogo.time_casa,
            time_fora=jogo.time_fora,
            kickoff_utc=jogo.kickoff_utc,
            kickoff_brt_str=kickoff_brt_str,
            gols_casa=aposta.gols_casa,
            gols_fora=aposta.gols_fora,
        )

        _registrar_backtest_seguro(jogo, prob_casa, prob_empate, prob_fora, predicao)
        _agendar_refinamentos(config, jogo)
        _agendar_resultado(config, jogo.id, jogo.time_casa, jogo.time_fora, jogo.kickoff_utc)
        return

    # falha: retry ou notificação definitiva
    if not escalar_falha:
        logger.warning(
            "Início do dia: aposta falhou para {c} x {f} ({m}), sem escalar; "
            "o T-60 e os refinamentos seguem como rede de segurança",
            c=jogo.time_casa, f=jogo.time_fora, m=resultado.mensagem,
        )
        return
    _lidar_com_falha(
        config, jogo, kickoff_local, predicao, resultado.mensagem, tentativa
    )


def _modo_reta_final(config: Config) -> bool:
    """Decide se a política de aposta entra no modo P(campeão)."""
    if config.estrategia_aposta == "ev":
        return False
    if config.estrategia_aposta == "campeao":
        return True

    # "auto": na fase de grupos, max-EV acumula pontos com o torneio inteiro pela
    # frente. No mata-mata, cada jogo é decisivo e a posição no ranking é tudo, daí
    # diferencia em todos eles. O piso `eh_mata_mata` evita o falso positivo da
    # janela rolante da Odds API no fim da fase de grupos.
    return eh_mata_mata(datetime.now(UTC))


def _aplicar_estrategia_campeao(
    config: Config,
    jogo: OddsJogo,
    predicao: Predicao,
    dados: DadosBolao,
) -> str | None:
    """Substitui o palpite max-EV pelo pick da estratégia que maximiza P(1o) no
    Monte Carlo (espelhar se líder, diferenciar se atrás; a otimização decide
    sozinha com posição e perfis reais).

    Simula uma janela dos próximos jogos (não só o atual): com 1 jogo só, estando
    atrás, P(1o)=0 para todas as estratégias e nada diferencia. O pick aplicado é
    o do jogo atual (`palpites[0]`). Retorna o rótulo da estratégia, ou None.
    """
    if len(predicao.placares) < 36:
        return None
    meu_nome = _meu_nome(config)
    meus_pontos, adversarios = _adversarios_da_subliga(dados, meu_nome, _membros_filtro(config, dados))
    if not adversarios:
        logger.warning("Sem adversários da subliga, mantendo max-EV")
        return None

    try:
        jogos_sim = [JogoSimulado.de_predicao(predicao)]  # jogo atual = índice 0
        jogos_sim += _predizer_proximos(config, excluir_id=jogo.id, n=_JANELA_OTIMIZACAO - 1)
        resultado_sim = simular(jogos_sim, meus_pontos, adversarios, n_sims=5_000)
    except Exception as e:
        logger.warning("Erro na simulação da estratégia: {e}, mantendo max-EV", e=e)
        return None
    if not resultado_sim:
        return None

    melhor = resultado_sim[0]
    pick_ev = predicao.melhor_placar
    label_campeao = melhor.palpites[0]

    prob_com_ev = next(
        (r.prob_campeao for r in resultado_sim if r.palpites[0] == pick_ev.label),
        resultado_sim[-1].prob_campeao,
    )

    if label_campeao == pick_ev.label:
        logger.info(
            "P(1o) confirma o pick max-EV {p} ({pc:.1%})",
            p=pick_ev.label,
            pc=melhor.prob_campeao,
        )
        return "max-EV, confirmado por P(1o)"

    novo = next((p for p in predicao.placares if p.label == label_campeao), None)
    if novo is None:
        logger.warning("Pick '{p}' fora da grade da predição, mantendo max-EV", p=label_campeao)
        return None

    predicao.aposta = novo
    logger.info(
        "Estratégia '{e}' troca {a} -> {n} | P(1o) {pe:.1%} -> {pn:.1%}",
        e=melhor.estrategia,
        a=pick_ev.label,
        n=label_campeao,
        pe=prob_com_ev,
        pn=melhor.prob_campeao,
    )
    return f"P(campeão) - {melhor.estrategia}"


def _notificar_se_ousado(
    config: Config,
    jogo: OddsJogo,
    predicao: Predicao,
    dados: DadosBolao,
    kickoff_local: datetime,
) -> None:
    """Envia notificação se a estratégia ótima (Monte Carlo) recomenda um placar
    diferente do mais provável, para o usuário decidir se aposta ousado na mão."""
    resultado_sim = _simular_com_standings(dados, predicao, _meu_nome(config), membros=_membros_filtro(config, dados))
    if not resultado_sim:
        return

    melhor = resultado_sim[0]
    pick_ousado = melhor.palpites[0]       # jogo[0] = este jogo
    pick_provavel = predicao.melhor_placar.label

    if pick_ousado == pick_provavel:
        logger.debug(
            "Teoria dos jogos confirma pick modal {p}, sem notificação ousada",
            p=pick_provavel,
        )
        return

    # P(campeão) do pick modal, para comparar
    prob_com_provavel = next(
        (r.prob_campeao for r in resultado_sim if r.palpites[0] == pick_provavel),
        resultado_sim[-1].prob_campeao,
    )

    lider = dados.classificacao[0][0] if dados.classificacao else "líder"
    pts_lider = dados.classificacao[0][3] if dados.classificacao else 0
    pts_diferenca = dados.meus_pontos - pts_lider

    logger.info(
        "Aposta ousada recomendada: {o} (em vez de {p}) | +{g:.1%} P(campeão)",
        o=pick_ousado,
        p=pick_provavel,
        g=melhor.prob_campeao - prob_com_provavel,
    )

    try:
        notificar_aposta_ousada(
            cc_api_url=config.cc_api_url,
            cc_token=config.cc_token,
            time_casa=jogo.time_casa,
            time_fora=jogo.time_fora,
            pick_ousado=pick_ousado,
            pick_provavel=pick_provavel,
            estrategia=melhor.estrategia,
            prob_campeao_ousado=melhor.prob_campeao,
            prob_campeao_provavel=prob_com_provavel,
            minha_posicao=dados.minha_posicao,
            lider=lider,
            pts_diferenca=pts_diferenca,
        )
    except Exception as e:
        logger.warning("Falha ao enviar notificação de aposta ousada: {e}", e=e)


def _alertar_scrape_falhou(config: Config, tipo: str, motivo: str) -> None:
    """Avisa (no máximo 1x a cada 6h) que o scrape do site quebrou. Sem isso a
    fonte de ranking/perfis seca em silêncio e o usuário só descobre quando o
    ranking para de chegar."""
    if cache_get("scrape:falha_alertada") is not None:
        return
    cache_set("scrape:falha_alertada", True, ttl=6 * 3600)
    try:
        notificar_scrape_falhou(
            cc_api_url=config.cc_api_url,
            cc_token=config.cc_token,
            tipo=tipo,
            motivo=motivo[:200],
        )
    except Exception as e:
        logger.warning("Falha ao notificar scrape quebrado: {e}", e=e)


def _raspar_picks_job(config: Config, jogo_id: str) -> None:
    """Job T+5min após kickoff: raspa picks (agora visíveis) e atualiza cache."""
    logger.info("Scrape de picks | jogo {id}", id=jogo_id)
    try:
        obter_dados_bolao(config)
    except Exception as e:
        logger.warning("Scrape de picks falhou: {e}", e=e)
        _alertar_scrape_falhou(config, "os picks", str(e))


def _e_ultimo_jogo_do_dia(config: Config, kickoff_utc: datetime) -> bool:
    """Evita enxurrada de rankings em dia de vários jogos: o ranking só é
    notificado após o último jogo do dia (data local do kickoff)."""
    try:
        jogos = buscar_odds(config.odds_api_key)
    except Exception:
        return True  # sem agenda para comparar, melhor notificar que silenciar

    tz = ZoneInfo(config.timezone)
    dia = kickoff_utc.astimezone(tz).date()
    return not any(
        j.kickoff_utc > kickoff_utc and j.kickoff_utc.astimezone(tz).date() == dia
        for j in jogos
    )


def _raspar_standings_job(
    config: Config,
    jogo_id: str,
    kickoff_utc: datetime | None = None,
) -> None:
    """Job T+3h após kickoff: standings já atualizados; raspa e notifica ranking."""
    logger.info("Scrape de standings | jogo {id}", id=jogo_id)
    try:
        dados = obter_dados_bolao(config)
    except Exception as e:
        logger.warning("Scrape de standings falhou: {e}", e=e)
        _alertar_scrape_falhou(config, "os standings", str(e))
        return

    if dados is None:
        _alertar_scrape_falhou(config, "os standings", "scrape retornou vazio (login/layout?)")
        return

    if not dados.classificacao:
        logger.warning(
            "Scrape de standings retornou classificação vazia, notificação omitida"
        )
        _alertar_scrape_falhou(config, "os standings", "classificação veio vazia")
        return

    # o scrape sempre roda (mantém perfis frescos); a notificação só após o
    # último jogo do dia
    if kickoff_utc is not None and not _e_ultimo_jogo_do_dia(config, kickoff_utc):
        logger.info("Ranking raspado mas não notificado, ainda há jogos hoje")
        return

    try:
        notificar_ranking_atualizado(
            cc_api_url=config.cc_api_url,
            cc_token=config.cc_token,
            classificacao=dados.classificacao,
            minha_posicao=dados.minha_posicao,
            proximo_jogo=None,
            estrategia_recomendada=None,
        )
    except Exception as e:
        logger.warning("Falha ao notificar ranking: {e}", e=e)


def _agendar_resultado(
    config: Config,
    jogo_id: str,
    time_casa: str,
    time_fora: str,
    kickoff_utc: datetime,
) -> None:
    """Agenda (ou reagenda) o check de resultado e os jobs de scraping do bolão."""
    horario_resultado = kickoff_utc + timedelta(minutes=_BUFFER_RESULTADO_MIN)
    _scheduler_global.add_job(
        _checar_resultado_job,
        trigger=DateTrigger(run_date=horario_resultado),
        args=[config, jogo_id, time_casa, time_fora],
        id=f"resultado_{jogo_id}",
        name=f"Resultado {time_casa} x {time_fora}",
        replace_existing=True,
    )

    # T+5min: picks ficam visíveis após kickoff
    _scheduler_global.add_job(
        _raspar_picks_job,
        trigger=DateTrigger(run_date=kickoff_utc + timedelta(minutes=5)),
        args=[config, jogo_id],
        id=f"scrape_picks_{jogo_id}",
        name=f"Scrape picks {time_casa} x {time_fora}",
        replace_existing=True,
    )

    # T+3h: standings atualizados após delay do site
    _scheduler_global.add_job(
        _raspar_standings_job,
        trigger=DateTrigger(run_date=kickoff_utc + timedelta(hours=3)),
        args=[config, jogo_id, kickoff_utc],
        id=f"scrape_standings_{jogo_id}",
        name=f"Scrape standings {time_casa} x {time_fora}",
        replace_existing=True,
    )

    logger.info(
        "Check de resultado agendado para {h}",
        h=horario_resultado.astimezone(ZoneInfo(config.timezone)).strftime("%d/%m %H:%M"),
    )


def _recuperar_resultados_pendentes(config: Config) -> None:
    """No startup, lida com as apostas persistidas conforme o estado de cada jogo:

    - já notificado: ignora (um restart não reenvia o que já foi avisado);
    - encerrado e ainda não notificado: coleta o placar e junta numa única
      mensagem-tabela (em vez de uma mensagem por jogo a cada restart);
    - ainda em andamento ou futuro: reagenda o check normal.
    """
    agora = datetime.now(UTC)
    fd_token = os.getenv("FOOTBALL_DATA_TOKEN", "").strip() or None
    recuperados = 0
    pendentes_tabela: list[tuple[ApostaRegistrada, tuple[int, int]]] = []

    for aposta in listar_apostas_pendentes():
        limite = aposta.kickoff_utc + timedelta(minutes=_BUFFER_RESULTADO_MIN)

        if limite > agora:
            _agendar_resultado(
                config, aposta.jogo_id, aposta.time_casa, aposta.time_fora, aposta.kickoff_utc
            )
            recuperados += 1
            continue

        if resultado_ja_notificado(aposta.jogo_id):
            continue  # já avisado num ciclo anterior, nada a fazer

        # encerrado e não notificado: busca o placar para entrar no resumo
        placar = buscar_placar_real(
            config.odds_api_key,
            aposta.jogo_id,
            time_casa=aposta.time_casa,
            time_fora=aposta.time_fora,
            football_data_token=fd_token,
        )
        if placar is None:
            # placar ainda indisponível, deixa o check normal (com retry) cuidar
            _scheduler_global.add_job(
                _checar_resultado_job,
                trigger=DateTrigger(run_date=agora + timedelta(seconds=10)),
                args=[config, aposta.jogo_id, aposta.time_casa, aposta.time_fora],
                id=f"resultado_{aposta.jogo_id}",
                name=f"Resultado {aposta.time_casa} x {aposta.time_fora} (recuperado)",
                replace_existing=True,
            )
            recuperados += 1
        else:
            pendentes_tabela.append((aposta, placar))

    _enviar_resumo_recuperado(config, pendentes_tabela)

    if recuperados:
        logger.info("{n} check(s) de resultado recuperado(s) do cache", n=recuperados)


def _enviar_resumo_recuperado(
    config: Config, pendentes: list[tuple[ApostaRegistrada, tuple[int, int]]]
) -> None:
    """Envia uma mensagem-tabela com todos os resultados encerrados recuperados,
    em vez de uma por jogo. Marca cada um como notificado, atualiza o total e o
    backtest, e anexa a posição ao vivo na subliga."""
    if not pendentes:
        return

    itens: list[ItemResumo] = []
    total = int(cache_get("resultados:pontos_total") or 0)
    for aposta, (real_casa, real_fora) in pendentes:
        pontos, _ = calcular_pontos(
            aposta.gols_apostados_casa, aposta.gols_apostados_fora, real_casa, real_fora
        )
        total += pontos
        itens.append(ItemResumo(
            time_casa=aposta.time_casa,
            time_fora=aposta.time_fora,
            real_casa=real_casa,
            real_fora=real_fora,
            apostado_casa=aposta.gols_apostados_casa,
            apostado_fora=aposta.gols_apostados_fora,
            pontos=pontos,
        ))
        try:
            registrar_resultado_backtest(aposta.jogo_id, real_casa, real_fora, pontos)
        except Exception as e:
            logger.warning("Backtest: falha ao registrar resultado recuperado: {e}", e=e)

    cache_set("resultados:pontos_total", total, ttl=90 * 24 * 3600)

    ranking = None
    try:
        dados = carregar_dados_bolao()
        if dados is not None:
            ranking = calcular_ranking_ao_vivo(
                dados, _meu_nome(config),
                membros_filtro=_membros_filtro(config, dados),
            )
    except Exception as e:
        logger.warning("Falha ao calcular ranking no resumo: {e}", e=e)

    try:
        notificar_resumo_resultados(
            cc_api_url=config.cc_api_url,
            cc_token=config.cc_token,
            itens=itens,
            pontos_acumulados=total,
            ranking=ranking,
        )
    except Exception as e:
        logger.warning("Falha ao enviar resumo de resultados: {e}", e=e)

    for aposta, _ in pendentes:
        marcar_resultado_notificado(aposta.jogo_id)


def _ranking_ao_vivo_seguro(
    config: Config, time_casa: str, time_fora: str, placar: tuple[int, int]
) -> RankingAoVivo | None:
    """Posição ao vivo na subliga a partir dos picks já cacheados (sem scrape novo,
    poupa o servidor) mais o placar deste jogo. Nunca quebra o fluxo de resultado."""
    try:
        dados = carregar_dados_bolao()
        if dados is None:
            return None
        return calcular_ranking_ao_vivo(
            dados,
            _meu_nome(config),
            jogo_atual_casa=time_casa,
            jogo_atual_fora=time_fora,
            resultado_atual=placar,
            membros_filtro=_membros_filtro(config, dados),
        )
    except Exception as e:
        logger.warning("Falha ao calcular ranking ao vivo: {e}", e=e)
        return None


def _checar_resultado_job(
    config: Config,
    jogo_id: str,
    time_casa: str = "",
    time_fora: str = "",
    tentativa: int = 1,
    max_tentativas: int = 4,
) -> None:
    """Verifica o placar real ~115 min após o kickoff. Se o jogo ainda não encerrou,
    reagenda a si mesmo por mais 30 min (até max_tentativas adicionais, ~3h após
    kickoff).

    Roda mesmo sem aposta registrada no cache (falha definitiva na aposta
    automática): o usuário pode ter apostado manualmente e merece saber o placar
    final do mesmo jeito, por isso recebe os nomes dos times.
    """
    # idempotência: um restart não pode reenviar um resultado já avisado
    if resultado_ja_notificado(jogo_id):
        logger.info("Resultado de {id} já notificado, pulando", id=jogo_id)
        return

    aposta: ApostaRegistrada | None = cache_get(f"aposta:{jogo_id}")
    nome_casa = aposta.time_casa if aposta else time_casa
    nome_fora = aposta.time_fora if aposta else time_fora

    placar = buscar_placar_real(
        config.odds_api_key,
        jogo_id,
        time_casa=nome_casa,
        time_fora=nome_fora,
        football_data_token=os.getenv("FOOTBALL_DATA_TOKEN", "").strip() or None,
    )

    if placar is None:
        if tentativa < max_tentativas:
            horario_retry = datetime.now(UTC) + timedelta(minutes=30)
            _scheduler_global.add_job(
                _checar_resultado_job,
                trigger=DateTrigger(run_date=horario_retry),
                args=[config, jogo_id, nome_casa, nome_fora, tentativa + 1, max_tentativas],
                id=f"resultado_{jogo_id}_t{tentativa + 1}",
                name=f"Resultado {nome_casa} x {nome_fora} (t{tentativa + 1})",
            )
            logger.info(
                "Placar ainda indisponível, reagendando em 30 min (t{t}/{m})",
                t=tentativa,
                m=max_tentativas,
            )
        else:
            logger.warning(
                "Placar não disponível após {m} tentativas | {id}",
                m=max_tentativas,
                id=jogo_id,
            )
            # sem isso o resultado nunca chegaria e o usuário ficaria esperando
            try:
                notificar_placar_nao_confirmado(
                    cc_api_url=config.cc_api_url,
                    cc_token=config.cc_token,
                    time_casa=nome_casa,
                    time_fora=nome_fora,
                )
            except Exception as e:
                logger.warning("Falha ao notificar placar não confirmado: {e}", e=e)
        return

    real_casa, real_fora = placar

    if aposta is None:
        # sem aposta automática registrada (falha definitiva): informa o placar
        # e lembra que pontos dependem de eventual aposta manual
        logger.warning(
            "Resultado sem aposta registrada | {c} x {f} | {rc} x {rf}",
            c=nome_casa, f=nome_fora, rc=real_casa, rf=real_fora,
        )
        notificar_resultado_sem_aposta(
            cc_api_url=config.cc_api_url,
            cc_token=config.cc_token,
            time_casa=nome_casa,
            time_fora=nome_fora,
            real_casa=real_casa,
            real_fora=real_fora,
        )
        marcar_resultado_notificado(jogo_id)
        return

    pontos_ganhos, _ = calcular_pontos(
        aposta.gols_apostados_casa, aposta.gols_apostados_fora, real_casa, real_fora
    )

    # total corrido das apostas automáticas: contexto no resultado sem depender
    # do scrape de standings (que chega horas depois)
    total = int(cache_get("resultados:pontos_total") or 0) + pontos_ganhos
    cache_set("resultados:pontos_total", total, ttl=90 * 24 * 3600)

    notificar_resultado(
        cc_api_url=config.cc_api_url,
        cc_token=config.cc_token,
        time_casa=aposta.time_casa,
        time_fora=aposta.time_fora,
        kickoff_brt_str=aposta.kickoff_brt_str,
        apostado_casa=aposta.gols_apostados_casa,
        apostado_fora=aposta.gols_apostados_fora,
        real_casa=real_casa,
        real_fora=real_fora,
        pontos_acumulados=total,
        ranking=_ranking_ao_vivo_seguro(config, aposta.time_casa, aposta.time_fora, placar),
    )
    marcar_resultado_notificado(jogo_id)
    try:
        registrar_resultado_backtest(jogo_id, real_casa, real_fora, pontos_ganhos)
    except Exception as e:
        logger.warning("Backtest: falha ao registrar resultado: {e}", e=e)


def _lidar_com_falha(
    config: Config,
    jogo: OddsJogo,
    kickoff_local: datetime,
    predicao: Predicao | None,
    motivo: str,
    tentativa: int,
) -> None:
    """Decide entre agendar retry (com notificação) ou notificar falha definitiva."""
    logger.warning(
        "Tentativa {t} falhou | {c} x {f} | motivo: {m}",
        t=tentativa,
        c=jogo.time_casa,
        f=jogo.time_fora,
        m=motivo,
    )

    agora = datetime.now(UTC)
    minutos_ate_kickoff = (jogo.kickoff_utc - agora).total_seconds() / 60
    minutos_proximo = minutos_ate_kickoff - config.retry_interval_minutes

    if tentativa < config.max_retries and minutos_proximo > 5:
        horario_retry = agora + timedelta(minutes=config.retry_interval_minutes)
        logger.info(
            "Retry {r} agendado em {m} minutos",
            r=tentativa + 1,
            m=config.retry_interval_minutes,
        )

        # sem notificação de retry: o desfecho (sucesso ou falha definitiva
        # "após N tentativas") conta a história inteira. avisos intermediários
        # só geram ruído sem ação possível
        _scheduler_global.add_job(
            _executar_job,
            trigger=DateTrigger(run_date=horario_retry),
            args=[config, jogo, tentativa + 1],
            id=f"retry_{jogo.id}_{tentativa + 1}",
            name=f"Retry {tentativa + 1} | {jogo.time_casa} x {jogo.time_fora}",
        )
    else:
        logger.critical(
            "Falha definitiva após {t} tentativas | {c} x {f}",
            t=tentativa,
            c=jogo.time_casa,
            f=jogo.time_fora,
        )
        notificar_falha(
            cc_api_url=config.cc_api_url,
            cc_token=config.cc_token,
            time_casa=jogo.time_casa,
            time_fora=jogo.time_fora,
            kickoff_brt=kickoff_local,
            predicao=predicao,
            motivo=f"{motivo} (após {tentativa} tentativa(s))",
        )

        # falha definitiva não pode significar silêncio: os checkpoints futuros
        # viram retries naturais (refinamento sem aposta no cache executa a
        # aposta inicial), e resultado/raspagens seguem agendados (o usuário pode
        # ter apostado manualmente após o alerta). tudo idempotente
        # (replace_existing=True) caso uma tentativa posterior tenha sucesso.
        _agendar_refinamentos(config, jogo)
        _agendar_resultado(config, jogo.id, jogo.time_casa, jogo.time_fora, jogo.kickoff_utc)


def _agendar_refinamentos(config: Config, jogo: OddsJogo) -> None:
    """Agenda os checkpoints T-45, T-30, T-15, T-5 após a aposta inicial (T-60)."""
    agora = datetime.now(UTC)
    for min_antes in _CHECKPOINTS_MIN[1:]:  # pula T-60, já apostou
        horario = jogo.kickoff_utc - timedelta(minutes=min_antes)
        if horario <= agora:
            continue
        _scheduler_global.add_job(
            _refinar_palpite_job,
            trigger=DateTrigger(run_date=horario),
            args=[config, jogo, min_antes],
            id=f"refinar_{jogo.id}_{min_antes}",
            name=f"Refinamento T-{min_antes}min | {jogo.time_casa} x {jogo.time_fora}",
            replace_existing=True,
            # grace curto: um refinamento muito atrasado (após o kickoff) não faz
            # sentido, melhor pular do que re-apostar com o jogo já rolando
            misfire_grace_time=240,
        )
        logger.info(
            "Refinamento T-{m}min agendado | {c} x {f}",
            m=min_antes,
            c=jogo.time_casa,
            f=jogo.time_fora,
        )


def _refinar_palpite_job(config: Config, jogo: OddsJogo, min_antes_kickoff: int) -> None:
    """Job agendado nos checkpoints T-45/30/15/5. Recalcula odds e re-aposta só se
    o placar ótimo mudou."""
    tz_local = ZoneInfo(config.timezone)
    logger.info(
        "Refinamento T-{m}min | {c} x {f}",
        m=min_antes_kickoff,
        c=jogo.time_casa,
        f=jogo.time_fora,
    )

    # odds frescas
    jogo = _atualizar_odds(config, jogo)

    # predição atualizada
    try:
        poly = buscar_probabilidades(jogo.time_casa, jogo.time_fora)
        prob_casa, prob_empate, prob_fora = ajustar_probabilidades(
            jogo.prob_casa, jogo.prob_empate, jogo.prob_fora, poly,
        )
    except Exception as e:
        logger.warning("Polymarket indisponível no refinamento: {e}", e=e)
        prob_casa, prob_empate, prob_fora = jogo.prob_casa, jogo.prob_empate, jogo.prob_fora

    # palpites dos oponentes: mesma fonte do T-60 para manter a ponderação de TJ
    try:
        from bolao.fontes.supabase import extrair_palpites_jogo
        palpites_oponentes = extrair_palpites_jogo(config, jogo.time_casa, jogo.time_fora)
    except Exception as e:
        logger.warning("Falha ao extrair palpites do Supabase no refinamento: {e}", e=e)
        palpites_oponentes = []

    try:
        predicao = predizer(
            time_casa=jogo.time_casa,
            time_fora=jogo.time_fora,
            prob_casa=prob_casa,
            prob_empate=prob_empate,
            prob_fora=prob_fora,
            lambda_total=jogo.lambda_total * config.lambda_fator,
            mata_mata=eh_mata_mata(jogo.kickoff_utc),
            top_n=36,  # grade completa para reaplicar a política P(campeão)
            palpites_oponentes=palpites_oponentes,
        )
    except Exception as e:
        logger.error("Erro na predição do refinamento: {e}", e=e)
        return

    # reaplica a mesma política do T-60, senão o refinamento reverteria o pick
    # diferenciado (otimizada) de volta para max-EV
    try:
        dados_bolao: DadosBolao | None = cache_get("bolao:dados_scraped")
        if dados_bolao is not None and _modo_reta_final(config):
            _aplicar_estrategia_campeao(config, jogo, predicao, dados_bolao)
    except Exception as e:
        logger.warning("Erro na política de aposta (refinamento): {e}", e=e)

    novo = predicao.melhor_placar

    # compara com aposta atual no cache
    aposta_atual: ApostaRegistrada | None = cache_get(f"aposta:{jogo.id}")
    if aposta_atual is None:
        # cache expirou ou aposta nunca foi registrada, faz aposta inicial aqui
        logger.warning(
            "T-{m}min: sem aposta registrada para {c} x {f}, executando aposta inicial",
            m=min_antes_kickoff,
            c=jogo.time_casa,
            f=jogo.time_fora,
        )
        _executar_job(config, jogo, tentativa=1)
        return

    if (novo.gols_casa == aposta_atual.gols_apostados_casa
            and novo.gols_fora == aposta_atual.gols_apostados_fora):
        logger.info(
            "T-{m}min: predição inalterada, mantendo {c} x {f}",
            m=min_antes_kickoff,
            c=novo.gols_casa,
            f=novo.gols_fora,
        )
        return

    # placar mudou, re-aposta
    logger.info(
        "T-{m}min: palpite mudou {a}x{b} -> {c}x{d}, re-apostando",
        m=min_antes_kickoff,
        a=aposta_atual.gols_apostados_casa,
        b=aposta_atual.gols_apostados_fora,
        c=novo.gols_casa,
        d=novo.gols_fora,
    )

    resultado = apostar(
        email=config.bolao_email,
        password=config.bolao_password,
        subleague=config.bolao_subleague,
        predicao=predicao,
    )

    if resultado.sucesso:
        # atualiza cache com novo palpite
        kickoff_local = jogo.kickoff_utc.astimezone(tz_local)
        registrar_aposta(
            jogo_id=jogo.id,
            time_casa=jogo.time_casa,
            time_fora=jogo.time_fora,
            kickoff_utc=jogo.kickoff_utc,
            kickoff_brt_str=kickoff_local.strftime("%d/%m às %H:%M"),
            gols_casa=novo.gols_casa,
            gols_fora=novo.gols_fora,
        )
        _registrar_backtest_seguro(jogo, prob_casa, prob_empate, prob_fora, predicao)
        try:
            notificar_palpite_atualizado(
                cc_api_url=config.cc_api_url,
                cc_token=config.cc_token,
                time_casa=jogo.time_casa,
                time_fora=jogo.time_fora,
                gols_casa_anterior=aposta_atual.gols_apostados_casa,
                gols_fora_anterior=aposta_atual.gols_apostados_fora,
                predicao=predicao,
                min_antes_kickoff=min_antes_kickoff,
            )
        except Exception as e:
            logger.warning("Falha ao notificar palpite atualizado: {e}", e=e)
    else:
        logger.error(
            "Falha ao re-apostar no refinamento T-{m}min: {err}",
            m=min_antes_kickoff,
            err=resultado.mensagem,
        )
        try:
            notificar_acao_manual_necessaria(
                cc_api_url=config.cc_api_url,
                cc_token=config.cc_token,
                time_casa=jogo.time_casa,
                time_fora=jogo.time_fora,
                gols_casa=novo.gols_casa,
                gols_fora=novo.gols_fora,
                min_antes_kickoff=min_antes_kickoff,
                motivo=resultado.mensagem,
            )
        except Exception as e:
            logger.warning("Falha ao enviar alerta de ação manual: {e}", e=e)


def _recuperar_refinamentos_pendentes(config: Config) -> None:
    """No startup, reagenda refinamentos futuros para apostas já registradas.
    Cobre restarts entre a aposta inicial (T-60) e o kickoff."""
    agora = datetime.now(UTC)
    recuperados = 0

    for aposta in listar_apostas_pendentes():
        if aposta.kickoff_utc <= agora + timedelta(minutes=5):
            continue  # muito perto ou já passou, nada a reagendar

        # reconstrói um OddsJogo mínimo para passar ao agendador
        # (odds serão atualizadas no momento do job)
        jogo_stub = OddsJogo(
            id=aposta.jogo_id,
            time_casa=aposta.time_casa,
            time_fora=aposta.time_fora,
            kickoff_utc=aposta.kickoff_utc,
            prob_casa=0.0,
            prob_empate=0.0,
            prob_fora=0.0,
            lambda_total=2.5,
        )
        _agendar_refinamentos(config, jogo_stub)
        recuperados += 1

    if recuperados:
        logger.info("Refinamentos reagendados para {n} jogo(s) recuperado(s)", n=recuperados)


def _alertar_creditos_baixos(config: Config) -> None:
    """Notifica (no máximo 1x por dia) quando os créditos da The Odds API caem
    abaixo do limite, para o usuário criar outra chave gratuita e trocar no .env."""
    restantes = cache_get(CHAVE_CREDITOS)
    if restantes is None or restantes > _LIMITE_ALERTA_CREDITOS:
        return
    if cache_get("odds_api:alerta_creditos_enviado") is not None:
        return

    try:
        notificar_creditos_baixos(
            cc_api_url=config.cc_api_url,
            cc_token=config.cc_token,
            restantes=int(restantes),
            limite=_LIMITE_ALERTA_CREDITOS,
        )
        # dedup: alerta de novo só amanhã se a chave continuar a mesma
        cache_set("odds_api:alerta_creditos_enviado", True, ttl=24 * 3600)
    except Exception as e:
        logger.warning("Falha ao alertar créditos baixos: {e}", e=e)


def _atualizar_odds(config: Config, jogo: OddsJogo) -> OddsJogo:
    """Re-busca odds frescas do jogo na hora da aposta. Em falha, mantém as odds
    capturadas no agendamento."""
    try:
        jogos = buscar_odds(config.odds_api_key, ignorar_cache=True)
        _alertar_creditos_baixos(config)
        atual = next((j for j in jogos if j.id == jogo.id), None)
        if atual is not None:
            logger.info(
                "Odds atualizadas | casa {a:.1%}->{b:.1%} | empate {c:.1%}->{d:.1%}",
                a=jogo.prob_casa,
                b=atual.prob_casa,
                c=jogo.prob_empate,
                d=atual.prob_empate,
            )
            return atual
        logger.warning("Jogo {id} ausente na resposta, usando odds do agendamento", id=jogo.id)
    except Exception as e:
        logger.warning("Falha ao atualizar odds, usando odds do agendamento: {e}", e=e)
    return jogo


# referência global ao scheduler (necessária para agendar retries dentro de jobs)
_scheduler_global: BlockingScheduler = cast(BlockingScheduler, None)


def _agendar_jogos(config: Config, notificar_novos: bool = False) -> None:
    """Busca os jogos futuros e (re)agenda um job de aposta para cada um.
    Idempotente: jogos já agendados são atualizados (replace_existing), jogos
    novos (ex: mata-mata recém-definido) entram na agenda. Roda no startup e
    periodicamente via re-scan. Com notificar_novos=True avisa por Telegram os
    jogos que ainda não estavam na agenda (não disparar no startup)."""
    tz = ZoneInfo(config.timezone)
    jogos = buscar_odds(config.odds_api_key)
    _alertar_creditos_baixos(config)

    if not jogos:
        logger.critical("Nenhum jogo encontrado! Verifique a ODDS_API_KEY.")
        return

    novos_lista: list[tuple[str, str, str]] = []
    apostas_registradas = {a.jogo_id for a in listar_apostas_pendentes()}

    for jogo in jogos:
        horario_job = jogo.kickoff_utc - timedelta(minutes=_CHECKPOINTS_MIN[0])
        agora = datetime.now(UTC)

        if horario_job <= agora:
            if agora < jogo.kickoff_utc and jogo.id not in apostas_registradas:
                # T-60 perdido (daemon reiniciou tarde) mas o jogo ainda não começou.
                # tenta recuperar agendando _executar_job no próximo checkpoint disponível
                kickoff_local = jogo.kickoff_utc.astimezone(tz)
                proximo = next(
                    (m for m in _CHECKPOINTS_MIN[1:]
                     if jogo.kickoff_utc - timedelta(minutes=m) > agora),
                    None,
                )
                if proximo is not None:
                    horario_recuperacao = jogo.kickoff_utc - timedelta(minutes=proximo)
                    logger.warning(
                        "T-60 perdido para '{c} x {f}', agendando tentativa em T-{m}min",
                        c=jogo.time_casa,
                        f=jogo.time_fora,
                        m=proximo,
                    )
                    _scheduler_global.add_job(
                        _executar_job,
                        trigger=DateTrigger(run_date=horario_recuperacao),
                        args=[config, jogo, 1],
                        id=f"jogo_{jogo.id}",
                        name=f"{jogo.time_casa} x {jogo.time_fora} | recuperação T-{proximo}min",
                        replace_existing=True,
                    )
                    notificar_recuperacao_janela(
                        config.cc_api_url,
                        config.cc_token,
                        jogo.time_casa,
                        jogo.time_fora,
                        proximo,
                    )
                else:
                    # todos os checkpoints passaram, só resta alertar manualmente
                    logger.warning(
                        "T-60 perdido e sem checkpoints futuros para '{c} x {f}'",
                        c=jogo.time_casa,
                        f=jogo.time_fora,
                    )
                    notificar_falha(
                        config.cc_api_url,
                        config.cc_token,
                        jogo.time_casa,
                        jogo.time_fora,
                        kickoff_local,
                        None,
                        "Daemon reiniciou após todos os checkpoints. Aposte manualmente.",
                    )
            else:
                logger.debug(
                    "Pulando '{c} x {f}', horário de aposta já passou",
                    c=jogo.time_casa,
                    f=jogo.time_fora,
                )
            continue

        kickoff_local = jogo.kickoff_utc.astimezone(tz)
        ja_agendado = _scheduler_global.get_job(f"jogo_{jogo.id}") is not None

        _scheduler_global.add_job(
            _executar_job,
            trigger=DateTrigger(run_date=horario_job),
            args=[config, jogo, 1],
            id=f"jogo_{jogo.id}",
            name=f"{jogo.time_casa} x {jogo.time_fora} | {kickoff_local.strftime('%d/%m %H:%M')}",
            replace_existing=True,
        )

        if not ja_agendado:
            kickoff_str = kickoff_local.strftime("%d/%m %H:%M")
            novos_lista.append((jogo.time_casa, jogo.time_fora, kickoff_str))
            logger.info(
                "Agendado: {c} x {f} | kickoff {k} | aposta às {a}",
                c=jogo.time_casa,
                f=jogo.time_fora,
                k=kickoff_str,
                a=horario_job.astimezone(tz).strftime("%H:%M"),
            )

    n_novos = len(novos_lista)
    logger.info("Agenda atualizada | {n} jogos novos", n=n_novos)

    if notificar_novos and novos_lista:
        try:
            notificar_novos_jogos(config.cc_api_url, config.cc_token, novos_lista)
        except Exception as e:
            logger.warning("Falha ao notificar novos jogos: {e}", e=e)


# Janela para detectar restart em loop (Restart=always do systemd a cada 30s)
_JANELA_LOOP_S = 900


def _notificar_startup(config: Config, recuperados: int) -> None:
    """Par da mensagem "daemon caiu" do systemd: confirma no Telegram que o serviço
    voltou, com resumo da agenda. Anti-spam: reinícios em sequência não repetem a
    mensagem; no 3o restart na janela, um único alerta urgente de loop (o restart
    automático não está resolvendo)."""
    reinicios = int(cache_get("startup:contagem") or 0) + 1
    cache_set("startup:contagem", reinicios, ttl=_JANELA_LOOP_S)

    try:
        if reinicios >= 3:
            if cache_get("startup:loop_alertado") is None:
                cache_set("startup:loop_alertado", True, ttl=3600)
                notificar_reinicio_em_loop(
                    cc_api_url=config.cc_api_url,
                    cc_token=config.cc_token,
                    reinicios=reinicios,
                    janela_min=_JANELA_LOOP_S // 60,
                )
            return

        proxima: str | None = None
        n_jogos = 0
        try:
            jogos = buscar_odds(config.odds_api_key)
            agora = datetime.now(UTC)
            futuros = sorted(
                (j for j in jogos if j.kickoff_utc > agora),
                key=lambda j: j.kickoff_utc,
            )
            n_jogos = len(futuros)
            if futuros:
                prox = futuros[0]
                t60 = (
                    prox.kickoff_utc - timedelta(minutes=_CHECKPOINTS_MIN[0])
                ).astimezone(ZoneInfo(config.timezone))
                proxima = (
                    f"{prox.time_casa} x {prox.time_fora}, {t60.strftime('%d/%m às %H:%M')}"
                )
        except Exception as e:
            logger.warning("Startup: falha ao montar resumo da agenda: {e}", e=e)

        creditos = cache_get(CHAVE_CREDITOS)
        notificar_daemon_no_ar(
            cc_api_url=config.cc_api_url,
            cc_token=config.cc_token,
            n_jogos=n_jogos,
            proxima_aposta=proxima,
            recuperados=recuperados,
            creditos=int(creditos) if creditos is not None else None,
        )
    except Exception as e:
        # notificação de startup nunca pode derrubar o daemon
        logger.warning("Falha ao notificar startup: {e}", e=e)


def _resumo_diario_job(config: Config) -> None:
    """Resumo das 09:00: agenda do dia, posição na subliga e créditos. Dobra como
    heartbeat: se esta mensagem não chegar, o daemon está travado ou morto, mesmo
    que o systemd não tenha detectado crash."""
    tz = ZoneInfo(config.timezone)
    agora = datetime.now(UTC)
    hoje = agora.astimezone(tz).date()

    jogos_hoje: list[tuple[str, str, str, str]] = []
    try:
        jogos = buscar_odds(config.odds_api_key)
        for j in sorted(jogos, key=lambda j: j.kickoff_utc):
            k_local = j.kickoff_utc.astimezone(tz)
            if k_local.date() == hoje and j.kickoff_utc > agora:
                t60 = k_local - timedelta(minutes=_CHECKPOINTS_MIN[0])
                jogos_hoje.append((
                    j.time_casa,
                    j.time_fora,
                    k_local.strftime("%H:%M"),
                    t60.strftime("%H:%M"),
                ))
    except Exception as e:
        logger.warning("Resumo diário: falha ao buscar jogos: {e}", e=e)

    dados: DadosBolao | None = cache_get("bolao:dados_scraped")
    creditos = cache_get(CHAVE_CREDITOS)

    notificar_resumo_diario(
        cc_api_url=config.cc_api_url,
        cc_token=config.cc_token,
        jogos_hoje=jogos_hoje,
        minha_posicao=dados.minha_posicao if dados else None,
        meus_pontos=dados.meus_pontos if dados else None,
        creditos=int(creditos) if creditos is not None else None,
    )


def _apostar_dia_job(config: Config) -> None:
    """Aposta em todos os jogos da janela do dia (kickoff nas próximas
    `janela_aposta_dia_horas`) que ainda não têm palpite registrado.

    Roda no startup e diariamente cedo (`hora_aposta_diaria`), garantindo um
    palpite com folga máxima: os refinamentos T-45/30/15/5 e o job T-60 atualizam
    conforme o kickoff se aproxima. É a defesa contra o bug que deixava jogos de
    00:00 sem aposta: apostar cedo, com a janela do site folgada, em vez de só no
    aperto do T-60.

    Falhas aqui não escalam (escalar_falha=False): o T-60 ainda tentará perto do
    jogo, então um tropeço às 08:00 não vira alerta urgente prematuro. Jogos já
    apostados são pulados, então dá para re-rodar a cada restart.
    """
    tz = ZoneInfo(config.timezone)
    agora = datetime.now(UTC)
    limite = agora + timedelta(hours=config.janela_aposta_dia_horas)

    try:
        jogos = sorted(buscar_odds(config.odds_api_key), key=lambda j: j.kickoff_utc)
    except Exception as e:
        logger.warning("Aposta do dia: falha ao buscar jogos: {e}", e=e)
        return

    ja_apostados = {a.jogo_id for a in listar_apostas_pendentes()}
    alvos = [
        j for j in jogos
        if agora < j.kickoff_utc <= limite and j.id not in ja_apostados
    ]

    if not alvos:
        logger.info(
            "Aposta do dia: nenhum jogo novo na janela de {h}h",
            h=config.janela_aposta_dia_horas,
        )
        return

    logger.info("Aposta do dia | apostando {n} jogo(s) cedo", n=len(alvos))
    for jogo in alvos:
        kickoff_local = jogo.kickoff_utc.astimezone(tz)
        logger.info(
            "Aposta inicial: {c} x {f} | kickoff {k}",
            c=jogo.time_casa,
            f=jogo.time_fora,
            k=kickoff_local.strftime("%d/%m %H:%M"),
        )
        try:
            _executar_job(config, jogo, tentativa=1, escalar_falha=False)
        except Exception as e:
            logger.exception(
                "Aposta do dia falhou para {c} x {f}: {e}",
                c=jogo.time_casa, f=jogo.time_fora, e=e,
            )


def main() -> None:
    """Ponto de entrada do daemon: inicializa o scheduler e agenda os jogos do
    torneio."""
    global _scheduler_global

    logger.add(
        "/var/log/bolao/bolao.log",
        rotation="1 day",
        retention="30 days",
        level="INFO",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    )

    logger.info("Bolão do torneio, daemon iniciando")
    config = carregar_config()
    validar_config_runtime(config)      # timezone + chave da API + tabela de recursos
    limpar_diagnosticos_antigos()       # não deixar traces velhos encherem o disco

    scheduler = BlockingScheduler(
        timezone=config.timezone,
        # grace generoso: um job que perdeu o horário por restart/stall ainda roda.
        # coalesce evita rajada de execuções acumuladas. refinamentos têm grace
        # curto próprio (240s).
        job_defaults={"misfire_grace_time": 3600, "coalesce": True},
    )
    _scheduler_global = scheduler

    logger.info("Buscando jogos do torneio na The Odds API...")
    _agendar_jogos(config, notificar_novos=False)

    # reagenda checks de resultado e refinamentos perdidos num restart anterior
    recuperados = len(listar_apostas_pendentes())
    _recuperar_resultados_pendentes(config)
    _recuperar_refinamentos_pendentes(config)

    # re-scan periódico: captura jogos novos (mata-mata, repescagem)
    # notificar_novos=True para avisar quando encontrar jogos inéditos
    scheduler.add_job(
        _agendar_jogos,
        trigger=IntervalTrigger(hours=6),
        args=[config, True],
        id="rescan_jogos",
        name="Re-scan de jogos novos",
    )

    # resumo das 09:00: agenda do dia + heartbeat (ausência = daemon travado)
    scheduler.add_job(
        _resumo_diario_job,
        trigger=CronTrigger(hour=9, minute=0, timezone=config.timezone),
        args=[config],
        id="resumo_diario",
        name="Resumo diário do bolão",
    )

    # aposta de início do dia: todos os jogos da janela ganham palpite com folga.
    # roda no startup (one-shot logo após o boot, sem travar a inicialização) e
    # diariamente cedo. refinamentos T-45/30/15/5 e o T-60 (rede de segurança)
    # atualizam depois. é a defesa contra os jogos de 00:00 que ficavam sem aposta.
    if config.apostar_inicio_dia:
        scheduler.add_job(
            _apostar_dia_job,
            trigger=DateTrigger(run_date=datetime.now(UTC) + timedelta(seconds=20)),
            args=[config],
            id="apostar_dia_startup",
            name="Aposta do dia (startup)",
        )
        scheduler.add_job(
            _apostar_dia_job,
            trigger=CronTrigger(
                hour=config.hora_aposta_diaria, minute=0, timezone=config.timezone
            ),
            args=[config],
            id="apostar_dia",
            name="Aposta inicial dos jogos do dia",
        )
        logger.info(
            "Aposta de início do dia ativa | diária às {h}h | janela {j}h",
            h=config.hora_aposta_diaria, j=config.janela_aposta_dia_horas,
        )

    # "Voltei" no Telegram: fecha o ciclo com o alerta "caiu" do systemd
    _notificar_startup(config, recuperados)

    logger.info("Daemon ativo | re-scan 6h | resumo diário 09:00")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Daemon encerrado")


if __name__ == "__main__":
    main()
