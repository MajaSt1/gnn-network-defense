
"""Model grafowej sieci neuronowej (GCN).

Odpowiedzialność (Osoba B):
- architektura GCN (warstwy GCNConv z PyTorch Geometric) dla zadania
  predykcji per-węzeł (score utwardzenia),
- pętla treningowa, funkcja straty, optymalizator,
- zapis/wczytanie wytrenowanego modelu.

ZADANIE: regresja na poziomie węzła. Dla każdego węzła model przewiduje
jedną liczbę w [0, 1] — "jak bardzo opłaca się go utwardzić" — i uczy się
imitować etykiety wyroczni (y z dataset.py).
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv

__all__ = ['DefenseGCN', 'train', 'evaluate', 'predict', 'save_model', 'load_model']


class DefenseGCN(nn.Module):
    """Dwuwarstwowy GCN do regresji per-węzeł.

    Przepływ informacji:
        x (n, in_dim)
          │  GCNConv 1  — każdy węzeł miesza swoje cechy z cechami sąsiadów
          ▼
        h (n, hidden)  → ReLU → dropout
          │  GCNConv 2  — drugie mieszanie: zasięg rośnie do sąsiadów sąsiadów
          ▼
        h (n, hidden)  → ReLU
          │  Linear     — sprowadza do 1 liczby na węzeł
          ▼
        out (n,)       → sigmoid → wartości w [0, 1]
    """

    def __init__(self, in_dim: int = 4, hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        # GCNConv(in, out): warstwa "message passing" — agreguje sąsiadów
        # i przepuszcza przez transformację liniową. To jest serce GNN.
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        # Zwykła warstwa liniowa: hidden cech -> 1 liczba (score) na węzeł.
        self.head = nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # --- 1. warstwa: węzeł zbiera informacje od bezpośrednich sąsiadów ---
        h = self.conv1(x, edge_index)
        h = F.relu(h)                                  # nieliniowość
        h = F.dropout(h, p=self.dropout, training=self.training)  # regularyzacja

        # --- 2. warstwa: zasięg rośnie o kolejny "krok" w grafie ---
        h = self.conv2(h, edge_index)
        h = F.relu(h)

        # --- głowica: z wektora cech robimy jedną liczbę na węzeł ---
        out = self.head(h).squeeze(-1)                 # (n, 1) -> (n,)

        # sigmoid ściska wynik do [0, 1] — tak jak znormalizowane etykiety y
        return torch.sigmoid(out)


def train(
    train_set: List[Data],
    val_set: List[Data] | None = None,
    in_dim: int = 4,
    hidden: int = 32,
    dropout: float = 0.2,
    lr: float = 1e-2,
    weight_decay: float = 5e-4,
    epochs: int = 100,
    batch_size: int = 16,
    device: str | None = None,
    verbose: bool = True,
) -> Tuple[DefenseGCN, dict]:
    """Trenuje GCN, by imitował etykiety wyroczni.

    Args:
        train_set:    Lista obiektów Data (graf + cechy + etykiety) do treningu.
        val_set:      Opcjonalny zbiór walidacyjny (grafy NIEwidziane w treningu).
        in_dim:       Liczba cech wejściowych na węzeł (u nas 4).
        hidden:       Szerokość warstw ukrytych.
        dropout:      Prawdopodobieństwo dropoutu (regularyzacja).
        lr:           Krok uczenia (learning rate) dla optymalizatora Adam.
        weight_decay: Kara L2 na wagi (przeciw przeuczeniu).
        epochs:       Liczba przejść przez cały zbiór treningowy.
        batch_size:   Liczba grafów w jednej paczce.
        device:       'cuda' / 'cpu'; auto-wykrywane jeśli None.
        verbose:      Wypisywanie postępu co kilka epok.

    Returns:
        model:   Wytrenowany model.
        history: Słownik z listami strat: {'train_loss': [...], 'val_loss': [...]}.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    model = DefenseGCN(in_dim=in_dim, hidden=hidden, dropout=dropout).to(device)

    # Adam: rozsądny domyślny optymalizator. weight_decay = regularyzacja L2.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # MSE: karzemy KWADRAT różnicy między predykcją a etykietą wyroczni.
    # Pasuje do regresji (przewidujemy liczbę ciągłą, nie klasę).
    loss_fn = nn.MSELoss()

    # DataLoader z PyG sam skleja kilka grafów w jedną wielką "paczkę"
    # (rozłączne grafy = jeden duży graf bez krawędzi między nimi).
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, epochs + 1):
        # tryb treningowy: dropout aktywny
        model.train()
        epoch_loss = 0.0
        n_seen = 0

        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()                 # 1. wyzeruj gradienty z poprzedniego kroku
            pred = model(batch.x, batch.edge_index)  # 2. forward: predykcja
            loss = loss_fn(pred, batch.y)         # 3. policz błąd vs wyrocznia
            loss.backward()                       # 4. backward: policz gradienty
            optimizer.step()                      # 5. popraw wagi w stronę mniejszego błędu

            # ważymy stratę liczbą węzłów w paczce (różne grafy, różne rozmiary)
            epoch_loss += loss.item() * batch.num_nodes
            n_seen += batch.num_nodes

        history['train_loss'].append(epoch_loss / n_seen)

        if val_set is not None:
            history['val_loss'].append(evaluate(model, val_set, device=device))

        if verbose and (epoch % 10 == 0 or epoch == 1):
            msg = f"epoka {epoch:3d} | train loss {history['train_loss'][-1]:.4f}"
            if val_set is not None:
                msg += f" | val loss {history['val_loss'][-1]:.4f}"
            print(msg)

    return model, history


@torch.no_grad()  # bez liczenia gradientów — szybciej, mniej pamięci
def evaluate(model: DefenseGCN, dataset: List[Data], device: str | None = None) -> float:
    """Średnia strata MSE na podanym zbiorze (bez aktualizacji wag).

    Używane do walidacji: mierzy, jak model radzi sobie na grafach,
    których NIE widział w treningu (= czy generalizuje).
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()                                   # tryb ewaluacji: dropout wyłączony
    loss_fn = nn.MSELoss()

    total, n_seen = 0.0, 0
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index)
        loss = loss_fn(pred, batch.y)
        total += loss.item() * batch.num_nodes
        n_seen += batch.num_nodes
    return total / n_seen


@torch.no_grad()
def predict(model: DefenseGCN, data: Data, device: str | None = None) -> torch.Tensor:
    """Zwraca score utwardzenia (n,) dla pojedynczego grafu.

    To jest "produkt" projektu: dla nowego grafu model w ułamku sekundy
    daje ranking węzłów do obrony — zamiast wolnej wyroczni Monte Carlo.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    data = data.to(device)
    return model(data.x, data.edge_index).cpu()


def save_model(model: DefenseGCN, path: str) -> None:
    """Zapisuje wagi modelu do pliku."""
    torch.save(model.state_dict(), path)


def load_model(path: str, in_dim: int = 4, hidden: int = 32) -> DefenseGCN:
    """Wczytuje wagi do nowej instancji modelu (architektura musi się zgadzać)."""
    model = DefenseGCN(in_dim=in_dim, hidden=hidden)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model
