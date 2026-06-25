"""
Baseline de mercado para a Copa 2022: forecast pré-jogo do FiveThirtyEight (SPI).

Não achei odds grátis de bookmaker para seleções, e o forecast do 538 serve bem:
é probabilidade pré-jogo, livre, e traz times, 1X2, gols projetados e placar real
no mesmo arquivo. Leio só as colunas pré-jogo (prob1/probtie/prob2, proj_score*);
score1/score2 são alvo. As colunas pós-jogo (xg, nsxg, adj_score) ficam de fora.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

_CSV_PADRAO = (
    Path(__file__).resolve().parents[1] / "dados" / "wc2022_538_matches.csv"
)


@dataclass(frozen=True)
class JogoMercado:
    """Um jogo da Copa 2022 com o forecast pré-jogo do 538 e o placar real."""

    data: date
    time1: str
    time2: str
    prob1: float       # P(time1 vence), pré-jogo
    prob_empate: float
    prob2: float       # P(time2 vence), pré-jogo
    proj1: float       # gols projetados time1 (pré-jogo)
    proj2: float       # gols projetados time2 (pré-jogo)
    gols1: int         # placar real: ALVO, nunca feature
    gols2: int

    @property
    def lambda_total(self) -> float:
        """Volume de gols projetado: lambda comum às opções na métrica secundária."""
        return self.proj1 + self.proj2

    @property
    def resultado(self) -> str:
        """'C' (time1 vence), 'E' (empate) ou 'F' (time2 vence)."""
        if self.gols1 > self.gols2:
            return "C"
        if self.gols1 == self.gols2:
            return "E"
        return "F"


def carregar_mercado_538(caminho: Path | None = None) -> list[JogoMercado]:
    """Lê o CSV do 538 e devolve os 64 jogos com forecast pré-jogo e placar."""
    csv_path = caminho or _CSV_PADRAO
    jogos: list[JogoMercado] = []
    with open(csv_path, encoding="utf-8-sig") as f:
        for linha in csv.DictReader(f):
            # pula linhas sem placar (não deveria haver em 2022)
            if not linha["score1"].strip() or not linha["score2"].strip():
                continue
            jogos.append(
                JogoMercado(
                    data=date.fromisoformat(linha["date"]),
                    time1=linha["team1"],
                    time2=linha["team2"],
                    prob1=float(linha["prob1"]),
                    prob_empate=float(linha["probtie"]),
                    prob2=float(linha["prob2"]),
                    proj1=float(linha["proj_score1"]),
                    proj2=float(linha["proj_score2"]),
                    gols1=int(float(linha["score1"])),
                    gols2=int(float(linha["score2"])),
                )
            )
    return jogos
