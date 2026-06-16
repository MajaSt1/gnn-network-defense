"""Skrypt treningowy: wczytuje dataset, trenuje GCN i zapisuje model.

Spina część osoby A (dataset) z modelem osoby B. Po treningu zapisuje
wagi do data/model.pt oraz wykres krzywej uczenia (train vs val loss),
który pokazuje, czy model się uczy i czy nie przeucza.

Użycie:
    python -m src.train                       # parametry domyślne
    python -m src.train --epochs 200 --hidden 64
"""
from __future__ import annotations

import argparse

from .dataset import load_dataset
from .model import train, save_model


def plot_history(history: dict, save_path: str) -> None:
    """Krzywa uczenia: strata treningowa i walidacyjna w funkcji epoki."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history['train_loss'], label='train loss')
    if history.get('val_loss'):
        ax.plot(history['val_loss'], label='val loss')
    ax.set_xlabel('epoka')
    ax.set_ylabel('MSE')
    ax.set_title('Krzywa uczenia GCN')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f'Zapisano krzywą uczenia -> {save_path}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Trening GCN obrony sieci')
    parser.add_argument('--dataset', default='data/dataset.pt')
    parser.add_argument('--out', default='data/model.pt')
    parser.add_argument('--history_plot', default='data/learning_curve.png')
    parser.add_argument('--val_frac', type=float, default=0.2,
                        help='Ułamek grafów na walidację (reszta na trening)')
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    n_val = max(1, int(len(dataset) * args.val_frac))
    train_set, val_set = dataset[:-n_val], dataset[-n_val:]
    in_dim = train_set[0].x.size(1)
    print(f'Dataset: {len(dataset)} grafów  ->  trening {len(train_set)} / walidacja {len(val_set)}')

    model, history = train(
        train_set, val_set=val_set, in_dim=in_dim,
        hidden=args.hidden, dropout=args.dropout, lr=args.lr,
        weight_decay=args.weight_decay, epochs=args.epochs,
        batch_size=args.batch_size, verbose=True,
    )

    save_model(model, args.out)
    print(f'Zapisano model -> {args.out}')
    plot_history(history, args.history_plot)


if __name__ == '__main__':
    main()
