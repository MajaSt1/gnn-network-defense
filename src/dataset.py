"""Budowa zbioru danych dla GNN.

Format obiektu Data:
  x          — cechy węzłów (n, 4): [stopień, betweenness, klastrowanie, closeness],
               wszystkie znormalizowane do [0, 1],
  edge_index — krawędzie (2, 2E): obie orientacje (graf nieskierowany),
  y          — etykiety wyroczni (n,): gain z utwardzenia węzła,
               znormalizowany max-skalowaniem do [0, 1] wewnątrz każdego grafu.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from .graphs import generate_ba_graph, generate_two_cluster_graph, node_features
from .oracle import node_marginal_gains_crn

__all__ = ['build_dataset', 'save_dataset', 'load_dataset', 'generate_and_save']


def build_dataset(
    n_graphs: int = 100,
    n_nodes: int = 50,
    m: int = 3,
    n_sim: int = 30,
    graph_type: str = 'two_cluster',
    model: str = 'SI',
    beta: float = 0.3,
    gamma: float = 0.05,
    resistance: float = 1.0,
    delay: int = 2,
    max_steps: int = 100,
    seed: Optional[int] = None,
) -> List[Data]:
    """Generuje zbiór grafów z etykietami wyroczni dla GNN.

    Dla każdego grafu:
      1. Losuje graf (BA lub dwa klastry + most).
      2. Oblicza 4 cechy strukturalne per węzeł.
      3. Szacuje gain z utwardzenia każdego węzła osobno (Monte Carlo).
      4. Normalizuje gainy do [0, 1] — to są etykiety y.

    Args:
        n_graphs:   Liczba grafów w zbiorze.
        n_nodes:    Liczba węzłów w każdym grafie.
        m:          Parametr BA (krawędzie dołączane przez nowy węzeł).
        n_sim:      Symulacje Monte Carlo na węzeł przy wyznaczaniu etykiet.
        graph_type: 'ba' (Barabási–Albert) lub 'two_cluster' (dwa klastry
                    połączone mostem — tu GNN pokazuje przewagę nad degree).
        model:      Model ataku — 'SI' lub 'SIR'.
        beta:       Bazowe prawdopodobieństwo zarażenia przez krawędź.
        gamma:      Prawdopodobieństwo wyzdrowienia (tylko SIR).
        resistance: Uodpornienie utwardzonego węzła przy liczeniu etykiet
                    (1.0 = pełne; mocniejszy, czytelniejszy sygnał dla GNN).
        delay:      Opóźnienie rozsiewania utwardzonych węzłów (kroki).
        max_steps:  Maksymalna liczba kroków symulacji.
        seed:       Ziarno losowości dla powtarzalności.

    Returns:
        Lista obiektów Data z polami x (n, 4), edge_index (2, 2E), y (n,).
    """
    rng = np.random.default_rng(seed)
    dataset: List[Data] = []

    for _ in tqdm(range(n_graphs), desc='Generowanie datasetu'):
        graph_seed = int(rng.integers(0, 2**31))
        if graph_type == 'two_cluster':
            G = generate_two_cluster_graph(n_nodes, m, seed=graph_seed)
        else:
            G = generate_ba_graph(n_nodes, m, seed=graph_seed)

        feats = node_features(G)
        # Etykiety metodą common random numbers — sygnał wyraźnie mocniejszy
        # niż przy niezależnych losowaniach (patrz oracle.py).
        gains = node_marginal_gains_crn(
            G, n_sim=n_sim, beta=beta, resistance=resistance,
            delay=delay, max_steps=max_steps, model=model, gamma=gamma,
            base_seed=graph_seed,
        )

        max_g = float(gains.max())
        y = (gains / max_g).astype(np.float32) if max_g > 0 else gains.astype(np.float32)

        if G.number_of_edges() > 0:
            src, dst = zip(*G.edges())
            src, dst = list(src), list(dst)
            edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        dataset.append(Data(
            x=torch.tensor(feats, dtype=torch.float),
            edge_index=edge_index,
            y=torch.tensor(y, dtype=torch.float),
        ))

    return dataset


def save_dataset(dataset: List[Data], path: str) -> None:
   
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, path)


def load_dataset(path: str) -> List[Data]:

    return torch.load(path, weights_only=False)


def generate_and_save(
    path: str = 'data/dataset.pt',
    n_graphs: int = 200,
    n_nodes: int = 50,
    m: int = 3,
    n_sim: int = 30,
    graph_type: str = 'two_cluster',
    model: str = 'SI',
    beta: float = 0.3,
    gamma: float = 0.05,
    resistance: float = 1.0,
    delay: int = 2,
    max_steps: int = 100,
    seed: int = 42,
) -> None:

    dataset = build_dataset(
        n_graphs=n_graphs, n_nodes=n_nodes, m=m, n_sim=n_sim,
        graph_type=graph_type, model=model, beta=beta, gamma=gamma,
        resistance=resistance, delay=delay, max_steps=max_steps,
        seed=seed,
    )
    save_dataset(dataset, path)
    d = dataset[0]
    print(f"Zapisano {len(dataset)} grafow -> {path}")
    print(f"Format: x={tuple(d.x.shape)}, edge_index={tuple(d.edge_index.shape)}, y={tuple(d.y.shape)}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generuj dataset i zapisz do pliku .pt')
    parser.add_argument('--path',       default='data/dataset.pt')
    parser.add_argument('--n_graphs',   type=int,   default=200)
    parser.add_argument('--n_nodes',    type=int,   default=50)
    parser.add_argument('--m',          type=int,   default=3)
    parser.add_argument('--n_sim',      type=int,   default=30)
    parser.add_argument('--graph_type', default='two_cluster', choices=['ba', 'two_cluster'])
    parser.add_argument('--model',      default='SI', choices=['SI', 'SIR'])
    parser.add_argument('--beta',       type=float, default=0.3)
    parser.add_argument('--gamma',      type=float, default=0.05)
    parser.add_argument('--resistance', type=float, default=1.0)
    parser.add_argument('--delay',      type=int,   default=2)
    parser.add_argument('--seed',       type=int,   default=42)
    args = parser.parse_args()

    generate_and_save(**vars(args))
