"""Matching resiliente de nomes de seleção entre fontes.

A The Odds API manda nomes em inglês e o site exibe em português; comparar
texto exato quebrava com grafia ou acento fora do dicionário. Aqui normaliza
(sem acento, casefold) e casa por similaridade (rapidfuzz). O dicionário EN
para PT é só um acelerador: país novo cai no fuzzy em vez de quebrar.

Módulo puro (sem I/O), reutilizado por apostador, fontes e resultados.
"""
from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz

# The Odds API usa inglês, o site exibe em português. Apelidos conhecidos
# aceleram o match exato; ausência aqui não quebra nada, só cai no fuzzy.
_NOME_PT: dict[str, str] = {
    "South Korea": "Coreia do Sul",
    "Czech Republic": "República Tcheca",
    "Mexico": "México",
    "South Africa": "África do Sul",
    "Switzerland": "Suíça",
    "Canada": "Canadá",
    "Scotland": "Escócia",
    "Brazil": "Brasil",
    "Morocco": "Marrocos",
    "Ivory Coast": "Costa do Marfim",
    "Ecuador": "Equador",
    "Germany": "Alemanha",
    "Japan": "Japão",
    "Sweden": "Suécia",
    "Tunisia": "Tunísia",
    "Netherlands": "Holanda",
    "Paraguay": "Paraguai",
    "Australia": "Austrália",
    "Turkey": "Turquia",
    "USA": "Estados Unidos",
    "Norway": "Noruega",
    "France": "França",
    "Iraq": "Iraque",
    "Cape Verde": "Cabo Verde",
    "Saudi Arabia": "Arábia Saudita",
    "Uruguay": "Uruguai",
    "Spain": "Espanha",
    "New Zealand": "Nova Zelândia",
    "Belgium": "Bélgica",
    "Egypt": "Egito",
    "Iran": "Irã",
    "Croatia": "Croácia",
    "Ghana": "Gana",
    "Panama": "Panamá",
    "England": "Inglaterra",
    "Colombia": "Colômbia",
    "DR Congo": "Congo",
    "Uzbekistan": "Uzbequistão",
    "Algeria": "Argélia",
    "Austria": "Áustria",
    "Jordan": "Jordânia",
    "Qatar": "Catar",
    "Nigeria": "Nigéria",
    "Cameroon": "Camarões",
    "Denmark": "Dinamarca",
    "Poland": "Polônia",
    "Wales": "País de Gales",
    "Serbia": "Sérvia",
    "Slovakia": "Eslováquia",
    "Greece": "Grécia",
    "Romania": "Romênia",
    "Ukraine": "Ucrânia",
    "Bosnia & Herzegovina": "Bósnia e Herzegovina",
    "Bosnia and Herzegovina": "Bósnia e Herzegovina",
}

# Sufixos de clube/categoria que poluem o match (raro em seleções, mas barato).
_SUFIXOS = (" fc", " sc", " sub-23", " sub-20", " u23", " u20")

# Abreviações/variantes que o fuzzy sozinho não casa (sem token em comum).
# Ex: "USA" e "united states" não compartilham nenhuma palavra.
_ALIASES_EN: dict[str, str] = {
    "usa": "united states",
    "united states": "usa",
    "uae": "united arab emirates",
    "south korea": "korea republic",
    "ir iran": "iran",
}

# Seleções com mais de uma grafia PT em uso. O dicionário acima guarda só a
# forma coloquial; a formal (ex: "Países Baixos" em vez de "Holanda") não tem
# token em comum, então o fuzzy não casa. Todas as grafias entram nas formas
# reconhecidas (ver _formas).
_GRAFIAS_PT_EXTRA: dict[str, tuple[str, ...]] = {
    "Netherlands": ("Países Baixos",),
}


def traduzir_pt(nome: str) -> str:
    return _NOME_PT.get(nome, nome)


def normalizar(nome: str) -> str:
    """Remove acentos, colapsa espaços, casefold e tira sufixos: base do match."""
    sem_acento = "".join(
        c for c in unicodedata.normalize("NFKD", nome) if not unicodedata.combining(c)
    )
    base = re.sub(r"\s+", " ", sem_acento).strip().casefold()
    for sufixo in _SUFIXOS:
        if base.endswith(sufixo):
            base = base[: -len(sufixo)].strip()
    return base


def casar_identidade(
    nomes: list[str], bolao_email: str, bolao_nome: str = ""
) -> str | None:
    """
    Acha o MEU nome na classificação do bolão.

    Prioriza `BOLAO_NOME` (casamento exato, case-insensitive mas preservando
    acento, ex.: "Joao Silva" vs "João Silva"). Sem ele, cai no heurístico: o 1º
    nome casa como prefixo do e-mail (jdoe123 -> "Joao Doe").
    Retorna None se nada casar.
    """
    if bolao_nome.strip():
        alvo = bolao_nome.strip().casefold()
        for n in nomes:
            if n.strip().casefold() == alvo:
                return n
    prefixo = bolao_email.casefold()
    for n in nomes:
        partes = n.split()
        if partes and prefixo.startswith(partes[0].casefold()):
            return n
    return None


def _formas(nome_en: str) -> set[str]:
    """Formas normalizadas conhecidas de uma seleção (inglês + tradução PT + aliases)."""
    formas = {normalizar(nome_en), normalizar(traduzir_pt(nome_en))}
    for grafia in _GRAFIAS_PT_EXTRA.get(nome_en, ()):
        formas.add(normalizar(grafia))
    for forma in list(formas):
        if forma in _ALIASES_EN:
            formas.add(normalizar(_ALIASES_EN[forma]))
    return formas


def nome_presente(nome_en: str, texto: str, minimo: int = 90) -> bool:
    """
    True se a seleção aparece no texto de um card.

    Tenta substring de qualquer forma conhecida (rápido e exato após normalizar);
    se falhar, usa similaridade parcial (rapidfuzz) para tolerar grafias novas.
    """
    alvo = normalizar(texto)
    formas = _formas(nome_en)
    if any(forma and forma in alvo for forma in formas):
        return True
    return any(forma and fuzz.partial_ratio(forma, alvo) >= minimo for forma in formas)


def casar_nome(alvo: str, candidatos: list[str], minimo: int = 86) -> str | None:
    """
    Melhor candidato cujo nome casa com `alvo` por `token_set_ratio`, ou None.

    Compara o alvo tanto na forma crua quanto na tradução PT conhecida, contra
    cada candidato normalizado. Usado para casar nomes entre fontes (Polymarket,
    football-data) sem depender de dicionário completo.
    """
    formas_alvo = _formas(alvo)
    melhor: str | None = None
    melhor_score = float(minimo)
    for cand in candidatos:
        cand_norm = normalizar(cand)
        score = max(fuzz.token_set_ratio(f, cand_norm) for f in formas_alvo)
        if score >= melhor_score:
            melhor_score = score
            melhor = cand
    return melhor
