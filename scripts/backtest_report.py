"""Relatório de backtest: qualidade das predições vs. resultados reais."""
from __future__ import annotations

from bolao.backtest import (
    calcular_metricas_backtest,
    calcular_rps,
    listar_registros_backtest,
)


def main() -> None:
    registros = listar_registros_backtest()
    if not registros:
        print("\nNenhum registro de backtest encontrado.")
        print("Os registros são criados automaticamente após cada aposta.\n")
        return

    completos = [r for r in registros if r.completo]
    pendentes = [r for r in registros if not r.completo]

    print(f"\n{'-' * 70}")
    print("  Backtest do bolão")
    print(f"{'-' * 70}")
    print(f"  Jogos com resultado: {len(completos)}   Pendentes: {len(pendentes)}")

    if completos:
        print()
        header = f"  {'Jogo':<22} {'Aposta':<7} {'Real':<7} {'Pts':>3}  {'E[pts]':>6}  {'RPS api':>7}  {'RPS mod':>7}"
        print(header)
        print(f"  {'-' * 22} {'-' * 7} {'-' * 7} {'-' * 3}  {'-' * 6}  {'-' * 7}  {'-' * 7}")

        for r in completos:
            assert r.gols_reais_casa is not None and r.gols_reais_fora is not None
            jogo_str = f"{r.time_casa[:10]} x {r.time_fora[:10]}"
            rps_api = calcular_rps(
                r.prob_casa_api, r.prob_empate_api, r.prob_fora_api,
                r.gols_reais_casa, r.gols_reais_fora,
            )
            rps_mod = calcular_rps(
                r.prob_casa_modelo, r.prob_empate_modelo, r.prob_fora_modelo,
                r.gols_reais_casa, r.gols_reais_fora,
            )
            pts_str = f"+{r.pontos_ganhos}" if r.pontos_ganhos else " 0"
            real_str = r.label_real or "-"
            print(
                f"  {jogo_str:<22} {r.label_apostado:<7} {real_str:<7} {pts_str:>3}"
                f"  {r.pontos_esperados:>6.2f}  {rps_api:>7.4f}  {rps_mod:>7.4f}"
            )

        print()
        m = calcular_metricas_backtest(registros)
        print(f"  {'-' * 70}")
        print(
            f"  Total: {m['n_jogos']} jogos | "
            f"Pontos: {m['pts_total']} (esperado: {m['pts_esperados']:.1f}) | "
            f"Cravadas: {m['n_cravadas']}  Resultados: {m['n_resultados']}  Erros: {m['n_erros']}"
        )
        print()
        rps_api_val = float(m["rps_medio_api"])
        rps_mod_val = float(m["rps_medio_modelo"])
        print(f"  RPS médio sem Polymarket: {rps_api_val:.4f}")
        print(f"  RPS médio com Polymarket: {rps_mod_val:.4f}  (uniforme ~= 0.222)")

        delta = rps_api_val - rps_mod_val
        if abs(delta) < 0.005:
            veredicto = "Polymarket quase não mexe no RPS"
        elif delta > 0:
            veredicto = f"Polymarket ajuda: -{delta:.4f} no RPS"
        else:
            veredicto = f"Polymarket atrapalha: +{-delta:.4f} no RPS"
        print(f"  -> {veredicto}")
        print(f"  Brier médio da aposta: {m['brier_medio']:.4f}")
        print()

    if pendentes:
        print(f"  Aguardando resultado ({len(pendentes)} jogos):")
        for r in pendentes:
            print(f"    {r.time_casa} x {r.time_fora:<14}  apostado: {r.label_apostado}")
        print()

    print(f"{'-' * 70}\n")


if __name__ == "__main__":
    main()
