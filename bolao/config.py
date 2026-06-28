"""Configuração central do bolão: lê variáveis de ambiente do .env e expõe
como dataclass tipada.
"""
import os
from dataclasses import dataclass
from pathlib import Path

import httpx
from dotenv import load_dotenv
from loguru import logger

_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(_env_path)

# URL base do site do bolão, backend e esporte ficam no .env (variam por bolão).
URL_BASE = os.getenv("BOLAO_BASE_URL", "")
SUPABASE_REST_URL = os.getenv("SUPABASE_URL", "")
ODDS_SPORT = os.getenv("ODDS_SPORT_KEY", "soccer_fifa_world_cup")
SUBLEAGUE_NOME = os.getenv("BOLAO_SUBLEAGUE_NOME", "Bolão")
PROJECT_TAG = os.getenv("BOLAO_PROJECT_TAG", "bolao")


@dataclass(frozen=True)
class Config:
    # The Odds API
    odds_api_key: str

    # Credenciais do site do bolão
    bolao_email: str
    bolao_password: str
    bolao_subleague: str        # UUID da subliga para apostas e login
    bolao_subleague_nome: str   # nome exibido na tab de picks

    # Daemon cc (notificações Telegram)
    cc_api_url: str
    cc_token: str

    # Comportamento
    max_retries: int
    retry_interval_minutes: int
    timezone: str
    ambiente: str  # "local" ou "producao"; em produção, CC_TOKEN é obrigatório

    # Política de aposta. Objetivo: terminar em 1º na subliga.
    #   "ev":      sempre max-EV (acumular pontos)
    #   "campeao": sempre o pick que maximiza P(campeão) no Monte Carlo
    #   "auto":    max-EV até a reta final; depois, P(campeão) (duas fases)
    estrategia_aposta: str = "auto"
    # Quantos jogos restantes no calendário definem a "reta final" no modo auto
    jogos_para_modo_final: int = 10
    # correção multiplicativa do volume de gols do mercado (lambda_total). abaixo
    # de 1.0 desloca o placar modal para margem-1 (1x0 em vez de 2x0); no backtest
    # sem vazamento, 0.85 a 0.90 deu +5 pts e +2 cravadas. 1.0 = neutro.
    lambda_fator: float = 1.0
    # meu nome exato na classificação (BOLAO_NOME). Quando setado, casa por ele em
    # vez do heurístico do prefixo do e-mail; necessário se o nome na plataforma
    # não começa com o usuário do e-mail (apelido). Vazio = usa a heurística.
    bolao_nome: str = ""
    # meu user_id (UUID) na plataforma (BOLAO_USER_ID). Quando setado, o caminho
    # HTTP me identifica por ele e segue qualquer troca de nome, sem precisar
    # atualizar BOLAO_NOME. É o identificador que não muda; nome/e-mail são fallback.
    bolao_user_id: str = ""
    # Membros da subliga (BOLAO_MEMBROS, separados por vírgula). Quando o scraper
    # raspa a liga inteira, a simulação P(campeão) e o ranking ao vivo filtram para
    # estes nomes. Vazio = sem filtro (usa toda a classificação).
    bolao_membros: tuple[str, ...] = ()
    # busca os dados do bolão (standings/picks) via HTTP em vez de raspar o DOM com
    # Playwright, o que corta o pico de CPU dos jobs de leitura. Cai no scraping se
    # o HTTP falhar. USAR_HTTP_SCRAPE=false força o scraping (desligar rápido).
    usar_http_scrape: bool = True
    # Aposta de todos os jogos do dia logo cedo (folga máxima antes do kickoff); os
    # refinamentos T-45/30/15/5 e o job T-60 atualizam conforme o jogo se aproxima.
    # Desligar com APOSTAR_INICIO_DIA=false.
    apostar_inicio_dia: bool = True
    # Hora local (0-23) do batch diário "apostar tudo do dia".
    hora_aposta_diaria: int = 8
    # Janela à frente (horas) do batch: cobre o dia + a virada (jogos de 00:00 do
    # dia seguinte). 28h pega, de um run às 08:00, até ~12:00 do dia seguinte.
    janela_aposta_dia_horas: int = 28


_PLACEHOLDER_CC = "seu_token_do_cc_aqui"


def carregar_config() -> Config:
    """
    Carrega e valida as variáveis de ambiente.

    CC_TOKEN é opcional localmente (só avisa no log).
    É obrigatório em produção (servidor com o daemon cc).
    """
    erros = []

    def _req(chave: str) -> str:
        valor = os.getenv(chave, "").strip()
        if not valor:
            erros.append(f"  - {chave} não está definida no .env")
        return valor

    ambiente = os.getenv("BOLAO_AMBIENTE", "local").strip().lower()

    def _cc_token() -> str:
        """CC_TOKEN: opcional localmente (só avisa); obrigatório em produção (fatal)."""
        valor = os.getenv("CC_TOKEN", "").strip()
        if not valor or valor == _PLACEHOLDER_CC:
            if ambiente == "producao":
                erros.append(
                    "  - CC_TOKEN é obrigatório em produção (BOLAO_AMBIENTE=producao); "
                    "sem ele as notificações Telegram não funcionam"
                )
            else:
                logger.warning(
                    "CC_TOKEN não configurado, notificações Telegram desativadas. "
                    "Configure antes de fazer deploy no servidor."
                )
            return ""
        return valor

    config = Config(
        odds_api_key=_req("ODDS_API_KEY"),
        bolao_email=_req("BOLAO_EMAIL"),
        bolao_password=_req("BOLAO_PASSWORD"),
        bolao_subleague=_req("BOLAO_SUBLEAGUE"),
        bolao_subleague_nome=os.getenv("BOLAO_SUBLEAGUE_NOME", "Bolão"),
        cc_api_url=os.getenv("CC_API_URL", "http://localhost:8765"),
        cc_token=_cc_token(),
        max_retries=int(os.getenv("MAX_RETRIES", "3")),
        retry_interval_minutes=int(os.getenv("RETRY_INTERVAL_MINUTES", "10")),
        timezone=os.getenv("TIMEZONE", "America/Sao_Paulo"),
        ambiente=ambiente,
        estrategia_aposta=os.getenv("ESTRATEGIA_APOSTA", "auto").strip().lower(),
        jogos_para_modo_final=int(os.getenv("JOGOS_PARA_MODO_FINAL", "10")),
        # Clamp defensivo: um typo em LAMBDA_FATOR não pode destruir o modelo de gols.
        lambda_fator=min(max(float(os.getenv("LAMBDA_FATOR", "1.0")), 0.5), 1.5),
        bolao_nome=os.getenv("BOLAO_NOME", "").strip(),
        bolao_user_id=os.getenv("BOLAO_USER_ID", "").strip(),
        bolao_membros=tuple(
            n.strip() for n in os.getenv("BOLAO_MEMBROS", "").split(",") if n.strip()
        ),
        usar_http_scrape=os.getenv("USAR_HTTP_SCRAPE", "true").strip().lower()
        not in ("false", "0", "no"),
        apostar_inicio_dia=os.getenv("APOSTAR_INICIO_DIA", "true").strip().lower()
        not in ("false", "0", "no"),
        hora_aposta_diaria=int(os.getenv("HORA_APOSTA_DIARIA", "8")),
        janela_aposta_dia_horas=int(os.getenv("JANELA_APOSTA_DIA_HORAS", "28")),
    )

    # BOLAO_BASE_URL e SUPABASE_URL são lidas em nível de módulo (URL_BASE,
    # SUPABASE_REST_URL) porque vários fetchers as importam direto. Validamos aqui
    # para falhar no boot, e não em runtime: sem elas o login, a aposta e a leitura
    # de palpites quebram com URL vazia, depois do daemon já ter subido.
    _req("BOLAO_BASE_URL")
    _req("SUPABASE_URL")

    if config.estrategia_aposta not in ("ev", "campeao", "auto"):
        erros.append(
            f"  - ESTRATEGIA_APOSTA inválida: '{config.estrategia_aposta}' "
            f"(use ev, campeao ou auto)"
        )

    if erros:
        msg = "Variáveis de ambiente obrigatórias faltando:\n" + "\n".join(erros)
        logger.critical(msg)
        raise OSError(msg)

    logger.info(
        "Configuração carregada | ambiente={a} | retries={r} | cc={cc}",
        a=config.ambiente,
        r=config.max_retries,
        cc="on" if config.cc_token else "desativado",
    )
    return config


def validar_config_runtime(config: Config) -> None:
    """Valida timezone e chave da API no startup, e loga os recursos opcionais
    (LLM, 2ª fonte de placares) ligados. Aborta (OSError) se algo essencial falhar.
    """
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    erros: list[str] = []

    try:
        ZoneInfo(config.timezone)
    except (ZoneInfoNotFoundError, ValueError):
        erros.append(f"  - TIMEZONE inválido: '{config.timezone}'")

    # Só uma chave definitivamente inválida (401) bloqueia o boot. Blip de rede,
    # 429 (créditos) ou 5xx viram aviso, o daemon sobe e tenta de novo nos jobs.
    # Bloquear o boot por falha transitória causaria loop de restart no systemd.
    try:
        resp = httpx.get(
            "https://api.the-odds-api.com/v4/sports",
            params={"apiKey": config.odds_api_key},
            timeout=10.0,
        )
        if resp.status_code == 401:
            erros.append("  - ODDS_API_KEY rejeitada (401) pela The Odds API")
        elif resp.is_success:
            logger.info(
                "The Odds API ok | créditos restantes: {c}",
                c=resp.headers.get("x-requests-remaining", "?"),
            )
        else:
            logger.warning(
                "The Odds API respondeu {s} no startup, seguindo; jobs tentarão de novo",
                s=resp.status_code,
            )
    except httpx.HTTPError as e:
        logger.warning(
            "The Odds API inacessível no startup ({e}), seguindo; jobs tentarão de novo", e=e
        )

    if erros:
        msg = "Validação de runtime falhou:\n" + "\n".join(erros)
        logger.critical(msg)
        raise OSError(msg)

    def _ativo(chave: str) -> str:
        return "ativo" if os.getenv(chave, "").strip() else "ausente"

    logger.info(
        "Recursos opcionais | LLM (Anthropic): {llm} | 2ª fonte de placares: {fd}",
        llm=_ativo("ANTHROPIC_API_KEY"),
        fd=_ativo("FOOTBALL_DATA_TOKEN"),
    )
