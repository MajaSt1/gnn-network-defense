"""Wizualizacja propagacji ataku — efekt „wow".

Zestawienie obok siebie: ten sam atak na sieci BEZ obrony i na sieci z obroną
GNN. Widać, jak utwardzone węzły hamują rozprzestrzenianie się ataku.

UWAGA: attack.py zwraca tylko LICZBY zarażonych na krok. Do animacji musimy
wiedzieć, KTÓRE węzły są zarażone w każdym kroku, dlatego poniżej jest
`simulate_attack_trace` — wierna kopia semantyki attack.py (resistance + delay),
która dodatkowo zapisuje stany wszystkich węzłów.
"""
from __future__ import annotations

from typing import List, Optional, Set

import numpy as np
import networkx as nx

__all__ = ['simulate_attack_trace', 'compare_defense_animation']

# kolory stanów węzłów
C_SUSCEPTIBLE = '#bfd7ea'   # podatny — jasnoniebieski
C_INFECTED = '#d62728'      # zarażony — czerwony
C_RECOVERED = '#7f7f7f'     # wyleczony — szary
C_HARDENED_EDGE = '#2ca02c' # obwódka węzła utwardzonego — zielony


def simulate_attack_trace(
    G: nx.Graph,
    hardened: Set[int] = frozenset(),
    source: Optional[int] = None,
    model: str = 'SI',
    beta: float = 0.3,
    gamma: float = 0.05,
    resistance: float = 0.6,
    delay: int = 5,
    horizon: int = 10,
    max_steps: int = 100,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """Jak simulate_attack, ale zwraca PEŁNĄ historię stanów do animacji.

    `horizon` jest przyjmowany dla zgodności z sygnaturą simulate_attack
    (animacja ma pełną historię, więc nie liczy metryki do kroku T).

    Returns:
        Lista tablic stanów (po jednej na krok). Każda tablica ma kształt (n,)
        z wartościami: 0 = podatny, 1 = zarażony, 2 = wyleczony.
    """
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    if source is None:
        source = int(rng.integers(0, n))

    state = np.zeros(n, dtype=np.int8)
    state[source] = 1
    infected_at = np.full(n, -1, dtype=np.int32)
    infected_at[source] = 0
    hardened_set: Set[int] = set(hardened)
    adj = [list(G.neighbors(v)) for v in range(n)]

    history = [state.copy()]

    for step in range(1, max_steps + 1):
        active = list(np.where(state == 1)[0])
        new_infections: Set[int] = set()
        recoveries: list = []

        for u in active:
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
        for u in recoveries:
            state[u] = 2

        history.append(state.copy())
        active_after = list(np.where(state == 1)[0])

        if model == 'SIR':
            if not active_after:
                break
        else:
            if not new_infections:
                has_pending = any(
                    u in hardened_set and step - infected_at[u] < delay
                    and any(state[v] == 0 for v in adj[u])
                    for u in active_after
                )
                if not has_pending:
                    break

    return history


def _node_colors(state: np.ndarray) -> List[str]:
    palette = {0: C_SUSCEPTIBLE, 1: C_INFECTED, 2: C_RECOVERED}
    return [palette[int(s)] for s in state]


def compare_defense_animation(
    G: nx.Graph,
    hardened: List[int],
    source: Optional[int] = None,
    interval: int = 600,
    save_path: Optional[str] = None,
    seed: int = 0,
    **attack_kwargs,
):
    """Zestawienie obok siebie: sieć BEZ obrony vs sieć z obroną GNN.

    Ten sam węzeł startowy i to samo ziarno losowości po obu stronach, żeby
    różnica wynikała WYŁĄCZNIE z obrony, nie z losowości symulacji.

    Args:
        save_path: jeśli kończy się '.gif' -> zapis przez Pillow,
                   inaczej ffmpeg; None -> tylko zwraca animację.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    pos = nx.spring_layout(G, seed=42)
    if source is None:
        source = int(np.random.default_rng(seed).integers(0, G.number_of_nodes()))

    hardened_set = set(hardened)
    sizes = [120 + 40 * G.degree(v) for v in G.nodes()]
    # obwódki: zielone wokół utwardzonych (tylko na panelu z obroną)
    edge_def = [C_HARDENED_EDGE if v in hardened_set else 'black' for v in G.nodes()]
    width_def = [3.0 if v in hardened_set else 0.3 for v in G.nodes()]

    # ten sam atak (to samo source + seed) na obu sieciach
    histories = [
        simulate_attack_trace(G, hardened=set(), source=source, seed=seed, **attack_kwargs),
        simulate_attack_trace(G, hardened=hardened_set, source=source, seed=seed, **attack_kwargs),
    ]
    labels = ['BEZ obrony', 'obrona GNN']
    n_frames = max(len(h) for h in histories)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    scatters = []
    for i, ax in enumerate(axes):
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2)
        sc = nx.draw_networkx_nodes(
            G, pos, ax=ax, node_size=sizes, node_color=_node_colors(histories[i][0]),
            edgecolors=(edge_def if i == 1 else 'black'),
            linewidths=(width_def if i == 1 else 0.3),
        )
        ax.axis('off')
        scatters.append(sc)

    def update(frame):
        for i, (sc, hist, ax) in enumerate(zip(scatters, histories, axes)):
            state = hist[min(frame, len(hist) - 1)]   # krótsza historia zamarza na ostatniej klatce
            sc.set_color(_node_colors(state))
            ax.set_title(f'{labels[i]}\nkrok {frame} | dotkniętych: {int(np.sum(state > 0))}')
        return scatters

    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)
    fig.suptitle('Propagacja ataku: ten sam atak, dwie sieci', fontsize=14)

    if save_path:
        writer = 'pillow' if save_path.endswith('.gif') else None
        anim.save(save_path, writer=writer, fps=max(1, 1000 // interval))
        print(f'Zapisano animację porównawczą -> {save_path}')
    return anim


if __name__ == '__main__':
    import argparse

    from .dataset import load_dataset
    from .model import load_model
    from .evaluate import data_to_graph, select_gnn, DEFAULT_ATTACK

    parser = argparse.ArgumentParser(description='Wizualizacja propagacji ataku')
    parser.add_argument('--dataset', default='data/dataset.pt')
    parser.add_argument('--model', default='data/model.pt')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--graph_idx', type=int, default=-1, help='Indeks grafu w zbiorze')
    parser.add_argument('--out', default='data/compare.gif')
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    data = dataset[args.graph_idx]
    G = data_to_graph(data)
    model = load_model(args.model, in_dim=data.x.size(1))
    hardened = select_gnn(data, G, model, args.k)

    print(f'Utwardzone węzły (GNN): {hardened}')
    compare_defense_animation(G, hardened, save_path=args.out, **DEFAULT_ATTACK)
