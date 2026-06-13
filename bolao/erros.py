"""Hierarquia de erros do bolão.

Classificar a falha (login, seletor, overlay, aposta não confirmada) deixa o
agendador decidir entre retry e ação manual, e a notificação dizer qual etapa
quebrou. São exceções puras, sem I/O; quem levanta anexa contexto na mensagem.
"""
from __future__ import annotations


class ErroBolao(Exception):
    """Raiz de todos os erros previsíveis do fluxo de aposta/scrape."""


class ErroLogin(ErroBolao):
    """Login rejeitado ou página não saiu de /login (credencial, 2FA, bloqueio)."""


class ErroOverlay(ErroBolao):
    """Popup/overlay desconhecido bloqueou a interação e não fechou."""


class ErroCardNaoEncontrado(ErroBolao):
    """O jogo não foi localizado entre os cards visíveis na página."""


class ErroModalPalpite(ErroBolao):
    pass


class ErroVerificacaoIncerta(ErroBolao):
    """A aposta foi enviada mas não deu para confirmar que o site persistiu.

    O palpite pode ter sido salvo, então o agendador trata como "confirme
    manualmente" em vez de registrar sucesso falso ou re-apostar cego.
    """
