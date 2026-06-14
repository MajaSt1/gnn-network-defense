"""Pakiet źródłowy projektu gnn-network-defense."""

from .graphs import generate_ba_graph, node_features
from .attack import simulate_attack
from .oracle import monte_carlo_score, node_marginal_gains, greedy_oracle
from .dataset import build_dataset, save_dataset, load_dataset
