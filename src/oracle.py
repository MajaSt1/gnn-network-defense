"""Wyrocznia wyboru węzłów do utwardzenia."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import networkx as nx
from tqdm import tqdm

from .attack import simulate_attack

__all__ = ['monte_carlo_score', 'node_marginal_gains', 'greedy_oracle']


def monte_carlo_score(
    G: nx.Graph,
    hardened: frozenset = frozenset(),
    n_sim: int = 50,
    **attack_kwargs,
) -> float:
    """Szacowana oczekiwana liczba zarażonych węzłów (niższa = lepsza obrona).

    Każda symulacja losuje nowy węzeł startowy, więc wynik odzwierciedla
    podatność całego grafu, nie jednego punktu wejścia.
    Klucz 'seed' jest pomijany, żeby każda z n_sim prób była niezależna.

    Args:
        G:               Graf NetworkX.
        hardened:        Zbiór indeksów utwardzonych węzłów.
        n_sim:           Liczba symulacji Monte Carlo.
        **attack_kwargs: Parametry przekazywane do simulate_attack.

    Returns:
        Średnia liczba zarażonych węzłów (float).
    """
    kwargs = {k: v for k, v in attack_kwargs.items() if k != 'seed'}
    totals = [
        simulate_attack(G, hardened=hardened, **kwargs)['total_infected']
        for _ in range(n_sim)
    ]
    return float(np.mean(totals))


def node_marginal_gains(
    G: nx.Graph,
    n_sim: int = 50,
    **attack_kwargs,
) -> np.ndarray:
    """Indywidualny gain z utwardzenia każdego węzła osobno.

    gain[v] = zarażeni_bez_obrony − zarażeni_gdy_tylko_v_jest_utwardzony

    Wartości ujemne (szum Monte Carlo) są ucinane do zera.

    Args:
        G:               Graf NetworkX.
        n_sim:           Liczba symulacji Monte Carlo na węzeł.
        **attack_kwargs: Parametry przekazywane do monte_carlo_score.

    Returns:
        Tablica float32 kształtu (n,).
    """
    n = G.number_of_nodes()
    baseline = monte_carlo_score(G, hardened=frozenset(), n_sim=n_sim, **attack_kwargs)
    gains = np.zeros(n, dtype=np.float32)
    for v in range(n):
        score = monte_carlo_score(G, hardened=frozenset({v}), n_sim=n_sim, **attack_kwargs)
        gains[v] = baseline - score
    return np.maximum(gains, 0.0)


def greedy_oracle(
    G: nx.Graph,
    k: int,
    n_sim: int = 50,
    verbose: bool = True,
    **attack_kwargs,
) -> Tuple[List[int], List[float]]:
    """Zachłanny wybór k węzłów minimalizujący liczbę zarażonych.

    W każdej rundzie wybierany jest węzeł, którego dodanie do aktualnie
    utwardzonych daje największy marginalny spadek oczekiwanej liczby zarażonych.
    Jest to zachłanne przybliżenie optymalizacji funkcji submodularnej.

    Args:
        G:               Graf NetworkX.
        k:               Budżet — liczba węzłów do utwardzenia.
        n_sim:           Liczba symulacji Monte Carlo na kandydata.
        verbose:         Pasek postępu tqdm.
        **attack_kwargs: Parametry przekazywane do monte_carlo_score.

    Returns:
        selected: Lista k indeksów węzłów w kolejności wyboru.
        gains:    Marginalny gain osiągnięty w każdej rundzie (>= 0).
    """
    n = G.number_of_nodes()
    selected: List[int] = []
    gains: List[float] = []
    current_hardened: set = set()

    current_score = monte_carlo_score(
        G, hardened=frozenset(), n_sim=n_sim, **attack_kwargs
    )

    rounds = tqdm(range(k), desc='Greedy oracle') if verbose else range(k)
    for _ in rounds:
        best_v: int = -1
        best_gain: float = -np.inf
        best_score: float = current_score

        for v in range(n):
            if v in current_hardened:
                continue
            score = monte_carlo_score(
                G, hardened=frozenset(current_hardened | {v}), n_sim=n_sim, **attack_kwargs
            )
            gain = current_score - score
            if gain > best_gain:
                best_gain = gain
                best_v = v
                best_score = score

        if best_v == -1:
            break

        selected.append(best_v)
        gains.append(float(max(best_gain, 0.0)))
        current_hardened.add(best_v)
        current_score = best_score

    return selected, gains
