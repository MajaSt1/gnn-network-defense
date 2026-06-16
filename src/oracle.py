"""Wyrocznia wyboru węzłów do utwardzenia."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import networkx as nx
from tqdm import tqdm

from .attack import simulate_attack

__all__ = ['monte_carlo_score', 'node_marginal_gains_crn', 'greedy_oracle']


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


def node_marginal_gains_crn(
    G: nx.Graph,
    n_sim: int = 150,
    beta: float = 0.3,
    resistance: float = 1.0,
    delay: int = 2,
    max_steps: int = 100,
    model: str = 'SI',
    gamma: float = 0.05,
    base_seed: int = 0,
) -> np.ndarray:
    """Gain z utwardzenia węzła liczony metodą *common random numbers* (CRN).

    DLACZEGO TO ISTNIEJE: naiwne liczenie gainu porównuje baseline
    i wariant "v utwardzony" na NIEZALEŻNYCH losowaniach ataku. Efekt
    utwardzenia jednego węzła z n jest mały i tonie w wariancji Monte Carlo
    — etykiety wychodzą prawie nieskorelowane same ze sobą (nie da się ich
    nauczyć). CRN losuje JEDEN wspólny zestaw `n_sim` ziaren i mierzy
    baseline oraz każdy wariant "v utwardzony" na TYCH SAMYCH przebiegach.
    Wspólny szum się skraca, zostaje czysty efekt obrony.

    Dodatkowo domyślnie resistance=1.0 (pełne uodpornienie węzła) — sygnał
    jest wtedy wyraźniejszy niż przy częściowej odporności.

    gain[v] = E[zarażeni_bez_obrony − zarażeni_gdy_v_utwardzony]  (te same ziarna)

    Args:
        G:          Graf NetworkX.
        n_sim:      Liczba wspólnych przebiegów Monte Carlo.
        beta:       Prawdopodobieństwo zarażenia przez krawędź.
        resistance: Redukcja zarażalności utwardzonego węzła (1.0 = pełna).
        delay:      Opóźnienie rozsiewania utwardzonego węzła.
        max_steps:  Maksymalna liczba kroków symulacji.
        model:      'SI' lub 'SIR'.
        gamma:      Częstość zdrowienia (tylko SIR).
        base_seed:  Ziarno generujące wspólny zestaw ziaren symulacji.

    Returns:
        Tablica float32 kształtu (n,), wartości ujemne ucięte do zera.
    """
    n = G.number_of_nodes()
    seeds = np.random.default_rng(base_seed).integers(0, 2**31, size=n_sim)
    ak = dict(model=model, beta=beta, gamma=gamma,
              resistance=resistance, delay=delay, max_steps=max_steps)

    baseline = np.array([
        simulate_attack(G, hardened=frozenset(), seed=int(s), **ak)['total_infected']
        for s in seeds
    ])
    gains = np.zeros(n, dtype=np.float32)
    for v in range(n):
        with_v = np.array([
            simulate_attack(G, hardened=frozenset({v}), seed=int(s), **ak)['total_infected']
            for s in seeds
        ])
        gains[v] = float((baseline - with_v).mean())
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
