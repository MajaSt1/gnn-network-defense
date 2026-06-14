"""Symulator propagacji ataku na grafie."""
from __future__ import annotations

from typing import Optional, Set

import numpy as np
import networkx as nx

__all__ = ['simulate_attack']


def simulate_attack(
    G: nx.Graph,
    hardened: Set[int] = frozenset(),
    source: Optional[int] = None,
    model: str = 'SI',
    beta: float = 0.3,
    gamma: float = 0.05,
    resistance: float = 0.7,
    delay: int = 2,
    max_steps: int = 100,
    seed: Optional[int] = None,
) -> dict:
    """Symulacja ataku SI lub SIR z uwzględnieniem utwardzonych węzłów.

    Efekty obrony działają niezależnie:
        - resistance: prawdopodobieństwo zarażenia utwardzonego węzła spada do
          beta * (1 - resistance).
        - delay: zarażony utwardzony węzeł zaczyna rozsiewać wirusa dopiero
          po `delay` krokach.

    Args:
        G:          Graf NetworkX (węzły 0..n-1).
        hardened:   Zbiór indeksów utwardzonych węzłów.
        source:     Węzeł startowy ataku; losowy jeśli None.
        model:      'SI' (brak zdrowienia) lub 'SIR' (zdrowienie z częstością gamma).
        beta:       Bazowe prawdopodobieństwo zarażenia przez krawędź na krok.
        gamma:      Prawdopodobieństwo wyzdrowienia na krok (tylko SIR).
        resistance: Redukcja prawdopodobieństwa zarażenia dla utwardzonych węzłów.
        delay:      Kroki opóźnienia rozsiewania dla zarażonego utwardzonego węzła.
        max_steps:  Maksymalna liczba kroków symulacji.
        seed:       Ziarno losowości.

    Returns:
        Słownik z kluczami:
            'infected_per_step' — liczba aktualnie zarażonych węzłów w każdym kroku,
            'total_infected'    — łączna liczba węzłów, które kiedykolwiek były zarażone,
            'steps'             — liczba wykonanych kroków.
    """
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()

    if source is None:
        source = int(rng.integers(0, n))

    # 0 = podatny, 1 = zarażony, 2 = wyleczony
    state = np.zeros(n, dtype=np.int8)
    state[source] = 1
    infected_at = np.full(n, -1, dtype=np.int32)
    infected_at[source] = 0
    ever_infected = np.zeros(n, dtype=bool)
    ever_infected[source] = True

    hardened_set: Set[int] = set(hardened)
    adj = [list(G.neighbors(v)) for v in range(n)]

    infected_per_step = [1]

    for step in range(1, max_steps + 1):
        active = list(np.where(state == 1)[0])
        new_infections: Set[int] = set()
        recoveries: list = []

        for u in active:
            # utwardzony węzeł czeka z rozsiewaniem przez `delay` kroków
            if u in hardened_set and step - infected_at[u] < delay:
                continue

            for v in adj[u]:
                if state[v] == 0:
                    p = beta * (1.0 - resistance) if v in hardened_set else beta
                    if rng.random() < p:
                        new_infections.add(v)

            if model == 'SIR' and rng.random() < gamma:
                recoveries.append(u)

        for v in new_infections:
            if state[v] == 0:
                state[v] = 1
                infected_at[v] = step
                ever_infected[v] = True

        for u in recoveries:
            state[u] = 2

        infected_per_step.append(int(np.sum(state == 1)))

        active_after = list(np.where(state == 1)[0])

        if model == 'SIR':
            if not active_after:
                break
        else:
            if not new_infections:
                # zatrzymaj tylko jeśli żaden utwardzony węzeł nie czeka jeszcze
                # na wyjście z opóźnienia przy podatnych sąsiadach
                has_pending = any(
                    u in hardened_set
                    and step - infected_at[u] < delay
                    and any(state[v] == 0 for v in adj[u])
                    for u in active_after
                )
                if not has_pending:
                    break

    return {
        'infected_per_step': infected_per_step,
        'total_infected': int(ever_infected.sum()),
        'steps': len(infected_per_step),
    }
