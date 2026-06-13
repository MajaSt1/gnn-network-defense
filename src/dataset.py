"""Budowa zbioru danych dla GNN.

Odpowiedzialność (Osoba A):
- generowanie wielu grafów o podobnej strukturze,
- wyznaczenie etykiet wyrocznią,
- zapis do obiektów torch_geometric.data.Data(x, edge_index, y).

STYK MIĘDZY OSOBAMI: format Data ustalany wspólnie.
  x          - cechy węzłów (np. stopień, betweenness, ...),
  edge_index - krawędzie grafu,
  y          - etykiety wyroczni (score utwardzenia per węzeł).
"""
