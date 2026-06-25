"""
Carrega o histórico de jogos internacionais do CSV local
(pesquisa/dados/international_results.csv), sem rede, para o Elo rodar igual toda
vez. Placar 'NA' é jogo ainda não disputado.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# CSV versionado em pesquisa/dados/ (parents[1] = pesquisa/).
_CSV_PADRAO = (
    Path(__file__).resolve().parents[1] / "dados" / "international_results.csv"
)

# Valor da coluna `tournament` para jogos da fase final da Copa do Mundo.
COPA_DO_MUNDO = "FIFA World Cup"


@dataclass(frozen=True)
class Partida:
    """Um jogo internacional. gols_* é None para jogos ainda não disputados."""

    data: date
    time_casa: str
    time_fora: str
    gols_casa: int | None
    gols_fora: int | None
    torneio: str
    neutro: bool  # True = campo neutro (sem vantagem de mando)

    @property
    def disputada(self) -> bool:
        return self.gols_casa is not None and self.gols_fora is not None

    @property
    def resultado(self) -> str:
        """'C' (casa vence), 'E' (empate) ou 'F' (fora vence). Exige jogo disputado."""
        if self.gols_casa is None or self.gols_fora is None:
            raise ValueError("resultado indefinido para jogo não disputado")
        if self.gols_casa > self.gols_fora:
            return "C"
        if self.gols_casa == self.gols_fora:
            return "E"
        return "F"


def carregar_partidas(caminho: Path | None = None) -> list[Partida]:
    """Lê o CSV e devolve as partidas em ordem de data (o Elo precisa da ordem)."""
    csv_path = caminho or _CSV_PADRAO
    partidas: list[Partida] = []
    with open(csv_path, encoding="utf-8") as f:
        for linha in csv.DictReader(f):
            partidas.append(
                Partida(
                    data=date.fromisoformat(linha["date"]),
                    time_casa=linha["home_team"],
                    time_fora=linha["away_team"],
                    gols_casa=_parse_gols(linha["home_score"]),
                    gols_fora=_parse_gols(linha["away_score"]),
                    torneio=linha["tournament"],
                    neutro=linha["neutral"].strip().upper() == "TRUE",
                )
            )
    partidas.sort(key=lambda p: p.data)
    return partidas


def _parse_gols(valor: str) -> int | None:
    bruto = valor.strip()
    if not bruto or bruto.upper() == "NA":
        return None
    try:
        return int(bruto)
    except ValueError:
        return None


