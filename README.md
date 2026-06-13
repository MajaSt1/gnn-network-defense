# gnn-network-defense

Zabezpieczanie sieci komputerowych z wykorzystaniem grafowej sieci neuronowej (GNN).
Projekt na przedmiot **Matematyczne Fundamenty Informatyki** (temat 13).

## Cel projektu

Sieć komputerowa jest reprezentowana jako graf. Trenujemy grafową sieć neuronową (GCN),
która dla każdego wierzchołka przewiduje, jak bardzo warto go **utwardzić**
(zwiększyć odporność), aby **opóźnić propagację szybko rozprzestrzeniającego się ataku**.

Sieć neuronowa uczy się imitować kosztowną **wyrocznię** (zachłanny wybór węzłów
oceniany symulacjami Monte Carlo), a następnie generalizuje na nowe grafy —
podejmując decyzję natychmiast, zamiast w sekundy/minuty.

### Parametry bezpieczeństwa wierzchołka

Utwardzony węzeł zyskuje **dwa** zabezpieczenia:

1. **Odporność na infekcję** — niższe prawdopodobieństwo zarażenia od sąsiada (`p → p·(1−r)`).
2. **Opóźnienie propagacji** — nawet po zarażeniu przekazuje atak dalej z opóźnieniem.

### Główne założenia

- **Grafy:** model Barabási–Albert (sieci scale-free z hubami, jak realne sieci komputerowe).
- **Model ataku:** propagacja typu SI / SIR po krawędziach grafu.
- **Budżet obrony:** utwardzamy tylko *k* wybranych węzłów.
- **Ewaluacja:** porównanie GNN z wyrocznią oraz baseline'ami (degree, betweenness, losowo).

## Struktura projektu

```
gnn-network-defense/
├── src/
│   ├── graphs.py       # generowanie grafów Barabási–Albert
│   ├── attack.py       # symulator propagacji ataku (SI/SIR) z parametrami obrony
│   ├── oracle.py       # zachłanna wyrocznia + ocena Monte Carlo
│   ├── dataset.py      # budowa zbioru grafów + etykiet dla GNN (PyG Data)
│   ├── model.py        # architektura GCN i pętla treningowa
│   ├── evaluate.py     # porównanie GNN vs wyrocznia vs baseline
│   └── visualize.py    # wizualizacja i animacja propagacji ataku
├── data/               # wygenerowane datasety (ignorowane przez git)
├── notebooks/          # eksperymenty, wykresy
├── requirements.txt
└── README.md
```

## Instalacja

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Podział pracy

Projekt realizują **2 osoby**. Podział przebiega wzdłuż naturalnej granicy:
osoba A dostarcza dane, osoba B je uczy i prezentuje. Stykiem jest format danych
(`torch_geometric.data.Data(x, edge_index, y)`) — ustalany wspólnie na początku.

### Osoba A — dane i atak
- `graphs.py` — generator grafów Barabási–Albert
- `attack.py` — symulator ataku SI/SIR z dwoma parametrami obrony
- `oracle.py` — zachłanna wyrocznia + ocena Monte Carlo
- `dataset.py` — generowanie zbioru grafów wraz z etykietami

### Osoba B — uczenie i prezentacja
- `model.py` — architektura GCN + pętla treningowa
- `evaluate.py` — porównanie GNN z wyrocznią i baseline'ami
- `visualize.py` — wizualizacja i animacja propagacji (efekt „wow")
- przygotowanie slajdów prezentacji

## Harmonogram (13–20.06)

| Termin | Zadanie | Odp. |
|--------|---------|------|
| 13–14.06 | generator grafów + symulator ataku | A |
| 14–15.06 | wyrocznia + generowanie datasetu | A |
| 15–16.06 | GCN + pętla treningowa | B |
| 17.06 | ewaluacja (GNN vs wyrocznia vs baseline) | B |
| 18.06 | wizualizacja + animacja propagacji | B |
| 19.06 | porządkowanie kodu, README, slajdy | A + B |
| 20.06 | bufor / poprawki / próba prezentacji | A + B |
