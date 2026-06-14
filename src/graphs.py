"""Generowanie grafów reprezentujących sieci komputerowe."""
from __future__ import annotations

import numpy as np
import networkx as nx

__all__ = ['generate_ba_graph', 'node_features']


def generate_ba_graph(n: int, m: int, seed=None) -> nx.Graph:
    """Graf Barabási–Albert z węzłami 0..n-1.

    Args:
        n:    Liczba węzłów.
        m:    Liczba krawędzi dołączanych przez nowy węzeł (kontroluje gęstość hubów).
        seed: Ziarno losowości.
    """
    return nx.barabasi_albert_graph(n, m, seed=seed)


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
