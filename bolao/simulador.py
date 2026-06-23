"""Simulador Monte Carlo do bolão: otimiza P(ser campeão), não E[pontos].

O bolão é um jogo relativo, palpitar igual a todo mundo não gera vantagem.
Simula milhares de desfechos dos próximos jogos e os palpites dos adversários,
e compara estratégias pela probabilidade de terminar em 1º lugar.

Estratégias avaliadas:
  - max_ev: maximiza E[pontos] jogo a jogo (ótimo "solo")
  - espelhar: copia o palpite modal da multidão (minimiza variância, certo
    quando se está na liderança)
  - diferenciar: melhor E[pontos] excluindo o palpite modal (certo quando atrás)
  - otimizada: hill-climbing sobre os candidatos maximizando P(campeão)

Modelo de multidão: cada adversário palpita o placar mais provável com
probabilidade `aderencia`; senão sorteia entre os 4 seguintes. Vale enquanto
não há palpites reais raspados do site.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from bolao.modelo import Predicao

if TYPE_CHECKING:
    from bolao.fontes.site import DadosBolao

# Quantos placares por jogo entram como candidatos no hill-climbing
_TOP_CANDIDATOS = 6

_CLASSE_RESULTADO = {"casa": 0, "empate": 1, "fora": 2}

# Mínimo de jogos no histórico para confiar no perfil empírico de um
# adversário; abaixo disso o modelo de multidão hipotético é menos enviesado
_MIN_JOGOS_PERFIL = 3


@dataclass
class PerfilAdversario:
    """
    Comportamento de palpite de um adversário, estimado dos picks reais
    raspados do site (visíveis após o prazo de cada jogo).

    Como não guardamos as odds dos jogos passados, o "favorito" de cada jogo
    histórico é aproximado pelo lado majoritário entre os palpites não-empate
    da subliga.
    """

    nome: str
    n_jogos: int
    p_empate: float            # fração de palpites em empate (suavizada)
    p_lado_majoritario: float  # dado não-empate, fração no lado da maioria
    formas_vitoria: dict[tuple[int, int], float] = field(default_factory=dict)
    # (gols vencedor, gols perdedor) -> frequência relativa, ex: (2,1) -> 0.4
    formas_empate: dict[int, float] = field(default_factory=dict)
    # k do empate kxk -> frequência relativa


@dataclass
class Adversario:
    """Um participante do bolão com sua pontuação atual."""

    nome: str
    pontos: float
    aderencia: float = 0.7  # prob de palpitar o placar modal (multidão típica)
    picks_fixos: dict[str, tuple[int, int]] = field(default_factory=dict)
    # chave: "time_casa x time_fora" (em PT), valor: (gols_casa, gols_fora)
    # Quando disponível (jogo já iniciado), usa o pick real em vez do modelo.
    perfil: PerfilAdversario | None = None
    # Perfil empírico calibrado com picks reais; quando presente e com
    # histórico suficiente, substitui o modelo de multidão hipotético.


@dataclass
class JogoSimulado:
    """Grade completa de placares de um jogo, em arrays para simulação."""

    time_casa: str
    time_fora: str
    probs: np.ndarray       # (36,) probabilidade de cada placar
    e_pontos: np.ndarray    # (36,) E[pontos] de palpitar cada placar
    labels: list[str]       # (36,) rótulo "G x G" de cada placar
    resultados: np.ndarray  # (36,) classe do placar: 0=casa, 1=empate, 2=fora

    @classmethod
    def de_predicao(cls, pred: Predicao) -> JogoSimulado:
        """Constrói a partir de uma Predicao com a grade completa (top_n=36)."""
        if len(pred.placares) < 36:
            raise ValueError(
                f"Predicao precisa da grade completa (top_n=36), veio {len(pred.placares)}"
            )
        probs = np.array([p.probabilidade for p in pred.placares])
        return cls(
            time_casa=pred.time_casa,
            time_fora=pred.time_fora,
            probs=probs / probs.sum(),
            e_pontos=np.array([p.pontos_esperados for p in pred.placares]),
            labels=[p.label for p in pred.placares],
            resultados=np.array(
                [_CLASSE_RESULTADO[p.resultado] for p in pred.placares]
            ),
        )


@dataclass
class ResultadoSimulacao:
    """Desempenho de uma estratégia de palpites nas simulações."""

    estrategia: str
    palpites: list[str]        # palpite escolhido por jogo
    pontos_esperados: float    # média de pontos ganhos nos próximos jogos
    prob_campeao: float        # P(1º lugar); empate no topo conta meio título


@dataclass
class ResultadoPolitica:
    """Desempenho de uma política de quando diferenciar do campo: P(1º), P(top-3)
    e posição média."""

    nome: str
    palpites: list[str]
    pontos_esperados: float
    prob_campeao: float
    prob_top3: float
    pos_media: float


@dataclass
class _Arena:
    """Desfechos sorteados de uma rodada, para pontuar estratégias diferentes
    nas mesmas realizações."""

    reais: np.ndarray       # (n_sims, n_jogos) índice do placar real sorteado
    pontos_adv: np.ndarray  # (n_adv, n_sims) pontuação final de cada adversário


def _ganhos_de(
    picks: list[int], jogos: list[JogoSimulado], reais: np.ndarray
) -> np.ndarray:
    """Pontos ganhos por um vetor de palpites (um por jogo) em cada simulação."""
    n_sims = reais.shape[0]
    g_total = np.zeros(n_sims)
    for g, jogo in enumerate(jogos):
        g_total += _pontuar(np.full(n_sims, picks[g]), reais[:, g], jogo.resultados)
    return g_total


def _preparar_arena(
    jogos: list[JogoSimulado],
    adversarios: list[Adversario],
    n_sims: int,
    seed: int | None,
) -> _Arena:
    """Sorteia os placares reais de cada simulação e a pontuação final de cada
    adversário (base atual + pontos sorteados nos jogos da janela)."""
    rng = np.random.default_rng(seed)

    reais = np.column_stack([
        rng.choice(len(j.probs), size=n_sims, p=j.probs) for j in jogos
    ])  # (n_sims, n_jogos)

    if not adversarios:
        return _Arena(reais=reais, pontos_adv=np.empty((0, n_sims)))

    pontos_adv = np.zeros((len(adversarios), n_sims))
    for k, adv in enumerate(adversarios):
        total = np.full(n_sims, float(adv.pontos))
        for g, jogo in enumerate(jogos):
            chave = f"{jogo.time_casa} x {jogo.time_fora}"
            if chave in adv.picks_fixos:
                # pick real conhecido (jogo já iniciado), determinístico
                gc, gf = adv.picks_fixos[chave]
                label = f"{gc} x {gf}"
                if label in jogo.labels:
                    idx = jogo.labels.index(label)
                else:
                    # Placar fora da grade (> max gols modelados): cai no modal
                    idx = int(np.argmax(jogo.probs))
                picks_adv = np.full(n_sims, idx)
            else:
                if adv.perfil is not None and adv.perfil.n_jogos >= _MIN_JOGOS_PERFIL:
                    dist = _dist_empirica(jogo, adv.perfil)
                else:
                    dist = _dist_multidao(jogo.probs, adv.aderencia)
                picks_adv = rng.choice(len(jogo.probs), size=n_sims, p=dist)
            total += _pontuar(picks_adv, reais[:, g], jogo.resultados)
        pontos_adv[k] = total

    return _Arena(reais=reais, pontos_adv=pontos_adv)


def _melhor_adv(arena: _Arena, n_sims: int) -> np.ndarray:
    """Pontuação do melhor adversário em cada simulação (-inf se não há adversário)."""
    if arena.pontos_adv.size:
        return np.asarray(arena.pontos_adv.max(axis=0))
    return np.full(n_sims, -np.inf)


def simular(
    jogos: list[JogoSimulado],
    meus_pontos: float,
    adversarios: list[Adversario],
    n_sims: int = 10_000,
    seed: int | None = None,
) -> list[ResultadoSimulacao]:
    """Roda o Monte Carlo e retorna as estratégias ordenadas por P(campeão)."""
    if not jogos:
        raise ValueError("Nenhum jogo para simular")

    n_jogos = len(jogos)
    arena = _preparar_arena(jogos, adversarios, n_sims, seed)
    reais = arena.reais
    melhor_adv = _melhor_adv(arena, n_sims)

    # monta os perfis de estratégia
    perfis = _perfis_basicos(jogos)

    # O hill-climbing parte do melhor perfil básico (não sempre de max_ev),
    # senão pode ficar preso num ótimo local pior que espelhar/diferenciar
    semente = max(
        perfis.values(),
        key=lambda picks: _prob_campeao(meus_pontos + _ganhos_de(picks, jogos, reais), melhor_adv),
    )
    perfis["otimizada"] = _otimizar(jogos, reais, melhor_adv, meus_pontos, semente)

    # avalia cada perfil nas mesmas simulações
    resultados = []
    for nome, perfil in perfis.items():
        ganhos = _ganhos_de(perfil, jogos, reais)
        totais = meus_pontos + ganhos
        resultados.append(ResultadoSimulacao(
            estrategia=nome,
            palpites=[jogos[g].labels[perfil[g]] for g in range(n_jogos)],
            pontos_esperados=float(ganhos.mean()),
            prob_campeao=_prob_campeao(totais, melhor_adv),
        ))

    resultados.sort(key=lambda r: r.prob_campeao, reverse=True)

    logger.info(
        "Monte Carlo: {s} sims | {j} jogos | {a} adversários | melhor={e} ({p:.1%})",
        s=n_sims,
        j=n_jogos,
        a=len(adversarios),
        e=resultados[0].estrategia,
        p=resultados[0].prob_campeao,
    )
    return resultados


def _pontuar(picks: np.ndarray, reais: np.ndarray, resultados: np.ndarray) -> np.ndarray:
    """Pontos do bolão: +3 cravada, +1 acertou só o resultado, 0 erro."""
    cravada = picks == reais
    resultado_certo = resultados[picks] == resultados[reais]
    return np.asarray(3 * cravada + (resultado_certo & ~cravada))


def _prob_campeao(meus_totais: np.ndarray, melhor_adv: np.ndarray) -> float:
    """P(1º lugar). Empate no topo divide o título (conta como meio)."""
    return float(
        np.mean(meus_totais > melhor_adv) + 0.5 * np.mean(meus_totais == melhor_adv)
    )


def _dist_multidao(probs: np.ndarray, aderencia: float) -> np.ndarray:
    """
    Distribuição de palpites da multidão: massa `aderencia` no placar modal,
    o restante repartido entre os 4 placares seguintes proporcionalmente.
    """
    dist = np.zeros_like(probs)
    top = np.argsort(probs)[::-1][:5]
    dist[top[0]] = aderencia
    resto = probs[top[1:]]
    dist[top[1:]] = (1 - aderencia) * resto / resto.sum()
    return dist


def estimar_perfis(dados: DadosBolao, meu_nome: str) -> dict[str, PerfilAdversario]:
    """
    Estima o perfil de palpite de cada adversário a partir dos picks reais
    dos jogos encerrados. Suavização de Laplace evita que histórico curto
    vire certeza (prior: ~25% empate, ~2/3 no lado majoritário).
    """
    stats: dict[str, dict] = defaultdict(lambda: {
        "n": 0, "empates": 0, "nao_empate": 0, "maioria": 0,
        "formas_v": Counter(), "formas_e": Counter(),
    })

    for jogo in dados.palpites_por_jogo:
        picks = jogo.picks
        if len(picks) < 2:
            continue  # sem multidão não há lado majoritário a estimar

        casa = sum(1 for gc, gf in picks.values() if gc > gf)
        fora = sum(1 for gc, gf in picks.values() if gc < gf)
        maioria = None if casa == fora else ("casa" if casa > fora else "fora")

        for nome, (gc, gf) in picks.items():
            s = stats[nome]
            s["n"] += 1
            if gc == gf:
                s["empates"] += 1
                s["formas_e"][gc] += 1
            else:
                s["nao_empate"] += 1
                lado = "casa" if gc > gf else "fora"
                if maioria is not None and lado == maioria:
                    s["maioria"] += 1
                s["formas_v"][(max(gc, gf), min(gc, gf))] += 1

    perfis: dict[str, PerfilAdversario] = {}
    for nome, s in stats.items():
        if nome.lower() == meu_nome.lower():
            continue
        n = s["n"]
        total_v = sum(s["formas_v"].values())
        total_e = sum(s["formas_e"].values())
        perfis[nome] = PerfilAdversario(
            nome=nome,
            n_jogos=n,
            p_empate=(s["empates"] + 1) / (n + 4),
            p_lado_majoritario=(s["maioria"] + 2) / (s["nao_empate"] + 3),
            formas_vitoria={k: v / total_v for k, v in s["formas_v"].items()} if total_v else {},
            formas_empate={k: v / total_e for k, v in s["formas_e"].items()} if total_e else {},
        )

    logger.info("Perfis estimados de {n} adversário(s) com picks reais", n=len(perfis))
    return perfis


def _dist_empirica(jogo: JogoSimulado, perfil: PerfilAdversario) -> np.ndarray:
    """
    Distribuição de palpites de um adversário calibrada com seu histórico:
    massa por classe de resultado (empate / lado favorito / zebra) segundo o
    perfil, repartida entre os placares pelas formas que ele costuma apostar.
    O favorito do jogo vem do modelo (lado com maior probabilidade somada).
    """
    p_casa = float(jogo.probs[jogo.resultados == 0].sum())
    p_fora = float(jogo.probs[jogo.resultados == 2].sum())
    favorito = 0 if p_casa >= p_fora else 2
    zebra = 2 if favorito == 0 else 0

    massa_por_classe = {
        1: perfil.p_empate,
        favorito: (1 - perfil.p_empate) * perfil.p_lado_majoritario,
        zebra: (1 - perfil.p_empate) * (1 - perfil.p_lado_majoritario),
    }

    dist = np.zeros_like(jogo.probs)
    for classe, massa in massa_por_classe.items():
        if massa <= 0:
            continue
        idxs = np.where(jogo.resultados == classe)[0]
        pesos = np.zeros(len(idxs))
        for k, idx in enumerate(idxs):
            # a grade vem ordenada por probabilidade, os gols saem do label
            gc, gf = (int(g) for g in jogo.labels[int(idx)].split(" x "))
            if classe == 1:
                pesos[k] = perfil.formas_empate.get(gc, 0.0)
            else:
                pesos[k] = perfil.formas_vitoria.get((max(gc, gf), min(gc, gf)), 0.0)
        if pesos.sum() <= 0:
            pesos = jogo.probs[idxs]  # forma nunca vista: segue o modelo
        dist[idxs] = massa * pesos / pesos.sum()

    return np.asarray(dist / dist.sum())


def _perfis_basicos(jogos: list[JogoSimulado]) -> dict[str, list[int]]:
    """Perfis fixos de estratégia: max_ev, espelhar e diferenciar."""
    max_ev = [int(np.argmax(j.e_pontos)) for j in jogos]
    espelhar = [int(np.argmax(j.probs)) for j in jogos]

    diferenciar = []
    for j in jogos:
        modal = int(np.argmax(j.probs))
        ordem = np.argsort(j.e_pontos)[::-1]
        alternativo = next(int(i) for i in ordem if int(i) != modal)
        diferenciar.append(alternativo)

    return {"max_ev": max_ev, "espelhar": espelhar, "diferenciar": diferenciar}


def _otimizar(
    jogos: list[JogoSimulado],
    reais: np.ndarray,
    melhor_adv: np.ndarray,
    meus_pontos: float,
    inicial: list[int],
    congelar_ate: int = 0,
) -> list[int]:
    """
    Hill-climbing por coordenada: para cada jogo, testa os _TOP_CANDIDATOS
    placares de maior E[pontos] e mantém a troca que aumentar P(campeão)
    nas simulações. Como avalia sempre nas mesmas simulações e parte do
    melhor perfil básico, o resultado nunca é pior que o ponto de partida.

    `congelar_ate`: índices de jogo `< congelar_ate` ficam fixos no palpite
    `inicial` (candidato único), usado para a política 'tarde', que joga
    max-EV no prefixo e só diferencia no sufixo da reta final.
    """
    n_sims = reais.shape[0]
    n_jogos = len(jogos)

    # Pré-computa os pontos de cada candidato em cada simulação
    candidatos: list[np.ndarray] = []
    pontos_cand: list[np.ndarray] = []  # (n_cand, n_sims) por jogo
    for g, jogo in enumerate(jogos):
        if g < congelar_ate:
            idxs = np.array([inicial[g]])  # jogo congelado: sem alternativas
        else:
            idxs = np.argsort(jogo.e_pontos)[::-1][:_TOP_CANDIDATOS]
            if inicial[g] not in idxs:
                idxs = np.append(idxs, inicial[g])
        candidatos.append(idxs)
        pontos_cand.append(np.stack([
            _pontuar(np.full(n_sims, int(i)), reais[:, g], jogo.resultados)
            for i in idxs
        ]))

    # Posição do palpite inicial dentro dos candidatos de cada jogo
    escolha = []
    for g in range(n_jogos):
        pos = np.where(candidatos[g] == inicial[g])[0]
        escolha.append(int(pos[0]) if len(pos) else 0)

    totais = meus_pontos + sum(pontos_cand[g][escolha[g]] for g in range(n_jogos))
    atual = _prob_campeao(totais, melhor_adv)

    for _ in range(3):  # passes de melhoria
        melhorou = False
        for g in range(n_jogos):
            base = totais - pontos_cand[g][escolha[g]]
            for c in range(len(candidatos[g])):
                if c == escolha[g]:
                    continue
                novo = _prob_campeao(base + pontos_cand[g][c], melhor_adv)
                if novo > atual + 1e-12:
                    atual = novo
                    escolha[g] = c
                    melhorou = True
            totais = base + pontos_cand[g][escolha[g]]
        if not melhorou:
            break

    return [int(candidatos[g][escolha[g]]) for g in range(n_jogos)]


def comparar_politicas_horizonte(
    jogos: list[JogoSimulado],
    meus_pontos: float,
    adversarios: list[Adversario],
    idx_gatilho: int,
    n_sims: int = 10_000,
    seed: int | None = None,
) -> list[ResultadoPolitica]:
    """Compara três políticas de quando diferenciar do campo, nos mesmos sorteios:

      nunca (max_ev): máx E[pontos] em todos os jogos (o que o daemon faz hoje)
      tarde: max-EV até `idx_gatilho`, otimizada no sufixo (gatilho atual)
      agora: otimizada (foge do consenso) desde o 1º jogo

    `idx_gatilho` é o índice (0-based) do 1º jogo em que 'tarde' passa a
    diferenciar; se `idx_gatilho >= len(jogos)`, 'tarde' vira 'nunca'.
    """
    if not jogos:
        raise ValueError("Nenhum jogo para simular")

    n_sims = int(n_sims)
    arena = _preparar_arena(jogos, adversarios, n_sims, seed)
    reais = arena.reais
    melhor_adv = _melhor_adv(arena, n_sims)

    max_ev = [int(np.argmax(j.e_pontos)) for j in jogos]

    # 'agora' parte do melhor perfil básico (igual ao daemon) p/ não cair em
    # ótimo local; 'tarde' parte do max-EV e só pode mexer no sufixo.
    perfis = _perfis_basicos(jogos)
    semente = max(
        perfis.values(),
        key=lambda picks: _prob_campeao(meus_pontos + _ganhos_de(picks, jogos, reais), melhor_adv),
    )
    agora = _otimizar(jogos, reais, melhor_adv, meus_pontos, semente)

    idx = max(0, min(idx_gatilho, len(jogos)))
    tarde = _otimizar(jogos, reais, melhor_adv, meus_pontos, max_ev, congelar_ate=idx)

    politicas = {
        "nunca (max_ev)": max_ev,
        "tarde (gatilho atual)": tarde,
        "agora": agora,
    }

    out: list[ResultadoPolitica] = []
    for nome, picks in politicas.items():
        ganhos = _ganhos_de(picks, jogos, reais)
        totais = meus_pontos + ganhos
        if arena.pontos_adv.size:
            # Posição = 1 + nº de adversários estritamente à frente (empate = melhor caso)
            pos = 1 + (arena.pontos_adv > totais[None, :]).sum(axis=0)
        else:
            pos = np.ones(n_sims, dtype=int)
        out.append(ResultadoPolitica(
            nome=nome,
            palpites=[jogos[g].labels[picks[g]] for g in range(len(jogos))],
            pontos_esperados=float(ganhos.mean()),
            prob_campeao=_prob_campeao(totais, melhor_adv),
            prob_top3=float(np.mean(pos <= 3)),
            pos_media=float(pos.mean()),
        ))

    logger.info(
        "Políticas | gatilho @ jogo {i}/{n} | P(1º): nunca={a:.1%} tarde={t:.1%} agora={g:.1%}",
        i=idx, n=len(jogos),
        a=out[0].prob_campeao, t=out[1].prob_campeao, g=out[2].prob_campeao,
    )
    return out
