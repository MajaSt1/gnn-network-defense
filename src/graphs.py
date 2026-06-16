"""Generowanie grafów reprezentujących sieci komputerowe."""
from __future__ import annotations

import numpy as np
import networkx as nx

__all__ = ['generate_ba_graph', 'generate_two_cluster_graph', 'node_features']


def generate_ba_graph(n: int, m: int, seed=None) -> nx.Graph:
    """Graf Barabási–Albert z węzłami 0..n-1.

    Args:
        n:    Liczba węzłów.
        m:    Liczba krawędzi dołączanych przez nowy węzeł (kontroluje gęstość hubów).
        seed: Ziarno losowości.
    """
    return nx.barabasi_albert_graph(n, m, seed=seed)


def generate_two_cluster_graph(n: int, m: int = 3, n_bridges: int = 2, seed=None) -> nx.Graph:
    """Dwa klastry Barabási–Albert połączone wąskimi "mostami".

    Po co taki graf: na zwykłym BA cała struktura sprowadza się do hubów,
    więc obrona "utwardź węzły o największym stopniu" jest niemal optymalna —
    sieć neuronowa nie ma czym przebić heurystyki. Tutaj kluczowe są WĘZŁY
    MOSTOWE: mają mały stopień (degree je przegapia), ale ich utwardzenie
    blokuje przeskok ataku między klastrami. Dobra obrona musi patrzeć na
    pozycję węzła w grafie (betweenness), a nie tylko na liczbę sąsiadów —
    i tu GNN, który widzi obie cechy, pokazuje przewagę.

    Args:
        n:         Łączna liczba węzłów (dzielona po równo na dwa klastry).
        m:         Parametr BA każdego klastra.
        n_bridges: Liczba krawędzi-mostów między klastrami.
        seed:      Ziarno losowości.

    Returns:
        Graf NetworkX z węzłami 0..n-1 (spójny).
    """
    rng = np.random.default_rng(seed)
    n_a = n // 2
    n_b = n - n_a
    a = nx.barabasi_albert_graph(n_a, m, seed=int(rng.integers(2**31)))
    b = nx.barabasi_albert_graph(n_b, m, seed=int(rng.integers(2**31)))
    G = nx.disjoint_union(a, b)  # klaster B dostaje numery n_a..n-1
    for _ in range(n_bridges):
        u = int(rng.integers(0, n_a))
        v = int(n_a + rng.integers(0, n_b))
        G.add_edge(u, v)
    return G


def node_features(G: nx.Graph) -> np.ndarray:
    """Macierz cech węzłów (n, 4) używana jako wejście GNN.

    Kolumny (wszystkie w zakresie [0, 1]):
        0 — stopień / (n-1)
        1 — betweenness centrality
        2 — współczynnik klastrowania
        3 — closeness centrality

    Args:
        G: Graf NetworkX z węzłami 0..n-1.

    Returns:
        Tablica float32 kształtu (n, 4).
    """
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    max_deg = max(n - 1, 1)

    degree = np.array([G.degree(v) / max_deg for v in nodes], dtype=np.float32)

    bc = nx.betweenness_centrality(G, normalized=True)
    betweenness = np.array([bc[v] for v in nodes], dtype=np.float32)

    cc = nx.clustering(G)
    clustering = np.array([cc[v] for v in nodes], dtype=np.float32)

    cl = nx.closeness_centrality(G)
    closeness = np.array([cl[v] for v in nodes], dtype=np.float32)

    return np.stack([degree, betweenness, clustering, closeness], axis=1)
