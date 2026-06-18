"""Ewaluacja strategii obrony.

Odpowiedzialność (Osoba B):
- porównanie obrony wskazanej przez GNN z:
    * wyrocznią (górna granica jakości),
    * baseline'ami: degree, betweenness, losowo,
- metryki: szybkość propagacji (dotknięci do kroku T) oraz finalny zasięg
  ataku, a także czas decyzji,
- generowanie wykresów porównawczych.

POMYSŁ: każda strategia wybiera k węzłów do utwardzenia. Główną miarą jakości
obrony jest liczba węzłów dotkniętych atakiem DO KROKU T ('affected_within_T',
mniej = wolniejsza propagacja = lepiej) — reaguje na oba parametry obrony
(barierę resistance i opóźnienie delay). Raportujemy też finalny zasięg
('total_infected'). GNN powinien zbliżyć się do wyroczni, ale podejmować
decyzję natychmiast, a nie w sekundy/minuty.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Tuple

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data

from .model import DefenseGCN, predict
from .oracle import monte_carlo_score, greedy_oracle

__all__ = [
    'data_to_graph',
    'select_gnn', 'select_degree', 'select_betweenness',
    'select_random', 'select_oracle',
    'evaluate_strategy', 'compare_strategies', 'plot_comparison',
]

# Parametry ataku użyte przy budowie datasetu (dataset.py). Trzymamy je
# spójne, żeby ewaluacja mierzyła obronę w tych samych warunkach, w jakich
# powstały etykiety wyroczni.
DEFAULT_ATTACK = dict(
    model='SI', beta=0.3, gamma=0.05,
    resistance=0.6, delay=5, horizon=10, max_steps=100,
)


def data_to_graph(data: Data) -> nx.Graph:
    """Odtwarza graf NetworkX z obiektu PyG Data.

    edge_index trzyma obie orientacje krawędzi (graf nieskierowany),
    więc nx.Graph naturalnie scala duplikaty w jedną krawędź.
    """
    G = nx.Graph()
    G.add_nodes_from(range(data.x.size(0)))
    edges = data.edge_index.t().tolist()
    G.add_edges_from(edges)
    return G


# --------------------------------------------------------------------------
#  Strategie wyboru k węzłów do utwardzenia.
#  Każda zwraca listę indeksów węzłów. Sygnatura wspólna: (data, G, model, k).
# --------------------------------------------------------------------------

def select_gnn(data: Data, G: nx.Graph, model: DefenseGCN, k: int) -> List[int]:
    """k węzłów z najwyższym score przewidzianym przez GNN."""
    scores = predict(model, data).numpy()
    return [int(v) for v in np.argsort(scores)[::-1][:k]]


def select_degree(data: Data, G: nx.Graph, model: DefenseGCN, k: int) -> List[int]:
    """k węzłów o największym stopniu (klasyczny baseline)."""
    deg = dict(G.degree())
    return sorted(deg, key=deg.get, reverse=True)[:k]


def select_betweenness(data: Data, G: nx.Graph, model: DefenseGCN, k: int) -> List[int]:
    """k węzłów o najwyższej betweenness centrality."""
    bc = nx.betweenness_centrality(G, normalized=True)
    return sorted(bc, key=bc.get, reverse=True)[:k]


def select_random(data: Data, G: nx.Graph, model: DefenseGCN, k: int,
                  seed: int | None = None) -> List[int]:
    """k losowych węzłów (dolny punkt odniesienia)."""
    rng = np.random.default_rng(seed)
    return [int(v) for v in rng.choice(G.number_of_nodes(), size=k, replace=False)]


def select_oracle(data: Data, G: nx.Graph, model: DefenseGCN, k: int,
                  n_sim: int = 300) -> List[int]:
    """k węzłów wybranych zachłannie przez wyrocznię (górna granica).

    Uwaga: na grafach BA stopień węzła (degree) jest już bliski optymalnemu,
    więc wyrocznia musi wykonać dużo symulacji (n_sim~300/rundę), żeby
    rzetelnie pobić ten baseline — stąd jej duży koszt czasowy.
    """
    selected, _ = greedy_oracle(G, k=k, n_sim=n_sim, verbose=False, **DEFAULT_ATTACK)
    return selected


# --------------------------------------------------------------------------
#  Pomiar jakości strategii.
# --------------------------------------------------------------------------

def evaluate_strategy(
    G: nx.Graph,
    hardened: List[int],
    n_sim: int = 50,
) -> Tuple[float, float]:
    """Jakość obrony przy danym zbiorze utwardzonych węzłów.

    Niższe wartości = lepsza obrona. Liczone tą samą metodą Monte Carlo,
    której używała wyrocznia, więc wyniki są bezpośrednio porównywalne.

    Returns:
        affected_T: śr. liczba węzłów dotkniętych do kroku T (główna miara,
                    czuła na oba parametry obrony),
        total:      śr. finalny zasięg ataku (metryka pomocnicza).
    """
    hardened_set = frozenset(int(v) for v in hardened)
    affected_T = monte_carlo_score(
        G, hardened=hardened_set, n_sim=n_sim,
        metric='affected_within_T', **DEFAULT_ATTACK,
    )
    total = monte_carlo_score(
        G, hardened=hardened_set, n_sim=n_sim,
        metric='total_infected', **DEFAULT_ATTACK,
    )
    return affected_T, total


def compare_strategies(
    dataset: List[Data],
    model: DefenseGCN,
    k: int = 5,
    n_sim: int = 50,
    include_oracle: bool = True,
    oracle_n_sim: int = 300,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Porównuje wszystkie strategie na zbiorze grafów testowych.

    Dla każdej strategii i każdego grafu:
      1. wybiera k węzłów do utwardzenia (mierząc czas decyzji),
      2. szacuje liczbę zarażonych (Monte Carlo).
    Wyniki uśredniane po grafach.

    Args:
        dataset:        Lista grafów testowych (Data).
        model:          Wytrenowany GNN.
        k:              Budżet obrony (liczba utwardzanych węzłów).
        n_sim:          Symulacje Monte Carlo przy ocenie obrony.
        include_oracle: Czy liczyć wyrocznię (wolna — można wyłączyć).
        oracle_n_sim:   Symulacje na rundę wewnątrz wyroczni.
        seed:           Ziarno dla strategii losowej.

    Returns:
        Słownik: nazwa_strategii -> {
            'affected':     śr. dotkniętych do kroku T (główna miara),
            'affected_std': odch. std tej miary,
            'total':        śr. finalny zasięg ataku (pomocnicza),
            'time':         śr. czas decyzji [s]}.
    """
    # baseline bez żadnej obrony — punkt odniesienia "ile zaraża niechroniona sieć"
    strategies: Dict[str, Callable] = {
        'GNN': select_gnn,
        'degree': select_degree,
        'betweenness': select_betweenness,
        'random': lambda d, G, m, k: select_random(d, G, m, k, seed=seed),
    }
    if include_oracle:
        strategies['oracle'] = lambda d, G, m, k: select_oracle(d, G, m, k, n_sim=oracle_n_sim)

    results: Dict[str, Dict[str, float]] = {}
    graphs = [data_to_graph(d) for d in dataset]

    # punkt odniesienia: sieć zupełnie bez obrony
    no_def = [evaluate_strategy(G, [], n_sim=n_sim) for G in graphs]
    aff0 = [a for a, _ in no_def]
    tot0 = [t for _, t in no_def]
    results['none'] = {
        'affected': float(np.mean(aff0)),
        'affected_std': float(np.std(aff0)),
        'total': float(np.mean(tot0)),
        'time': 0.0,
    }

    for name, select_fn in strategies.items():
        affected: List[float] = []
        total: List[float] = []
        times: List[float] = []
        for data, G in zip(dataset, graphs):
            t0 = time.perf_counter()
            hardened = select_fn(data, G, model, k)
            times.append(time.perf_counter() - t0)
            a, t = evaluate_strategy(G, hardened, n_sim=n_sim)
            affected.append(a)
            total.append(t)

        results[name] = {
            'affected': float(np.mean(affected)),
            'affected_std': float(np.std(affected)),
            'total': float(np.mean(total)),
            'time': float(np.mean(times)),
        }

    return results


def plot_comparison(results: Dict[str, Dict[str, float]], k: int = 5, save_path: str | None = None):
    """Wykres słupkowy: średnia liczba węzłów dotkniętych do kroku T per strategia.

    Im niższy słupek, tym wolniejsza propagacja ataku (lepsza obrona). 'none'
    pokazuje sieć bez obrony, 'oracle' to praktyczna górna granica jakości.
    """
    import matplotlib.pyplot as plt

    # kolejność od najgorszej (none) do najlepszej obrony czytamy z danych
    order = ['none', 'random', 'degree', 'betweenness', 'GNN', 'oracle']
    names = [n for n in order if n in results]
    means = [results[n]['affected'] for n in names]
    stds = [results[n]['affected_std'] for n in names]

    colors = {
        'none': '#7f7f7f', 'random': '#bcbd22', 'degree': '#1f77b4',
        'betweenness': '#9467bd', 'GNN': '#d62728', 'oracle': '#2ca02c',
    }
    bar_colors = [colors.get(n, '#333333') for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, means, yerr=stds, capsize=4, color=bar_colors, alpha=0.85)
    ax.set_ylabel('Śr. liczba węzłów dotkniętych do kroku T (mniej = wolniejszy atak)')
    ax.set_title(f'Skuteczność obrony przy budżecie k = {k} utwardzonych węzłów')
    ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f'Zapisano wykres -> {save_path}')
    return fig, ax


def print_report(results: Dict[str, Dict[str, float]], k: int = 5) -> None:
    """Wypisuje tabelę wyników w konsoli."""
    print(f"\n=== Porównanie strategii obrony (k={k}) ===")
    print(f"{'strategia':<14}{'dotknięci do T (śr±std)':<26}"
          f"{'finalny zasięg':<16}{'czas decyzji [ms]':<18}")
    print("-" * 74)
    for name, r in results.items():
        affected = f"{r['affected']:.1f} ± {r['affected_std']:.1f}"
        print(f"{name:<14}{affected:<26}{r['total']:<16.1f}{r['time'] * 1e3:>10.2f}")


if __name__ == '__main__':
    import argparse

    from .dataset import load_dataset
    from .model import load_model

    parser = argparse.ArgumentParser(description='Ewaluacja strategii obrony GNN vs baseline')
    parser.add_argument('--dataset', default='data/dataset.pt')
    parser.add_argument('--model', default='data/model.pt')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--n_sim', type=int, default=50)
    parser.add_argument('--n_test', type=int, default=20, help='Liczba grafów testowych')
    parser.add_argument('--no_oracle', action='store_true', help='Pomiń wyrocznię (szybciej)')
    parser.add_argument('--plot', default='data/comparison.png')
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    test_set = dataset[-args.n_test:]
    model = load_model(args.model, in_dim=test_set[0].x.size(1))

    results = compare_strategies(
        test_set, model, k=args.k, n_sim=args.n_sim,
        include_oracle=not args.no_oracle,
    )
    print_report(results, k=args.k)
    plot_comparison(results, k=args.k, save_path=args.plot)
