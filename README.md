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

Zgodnie z treścią zadania utwardzony węzeł zyskuje **dwa** parametry bezpieczeństwa,
które działają niezależnie:

1. **Odporność na infekcję** (`resistance`) — niższe prawdopodobieństwo zarażenia
   od sąsiada (`p → p·(1−r)`). To „twarda bariera": ogranicza, *które* węzły padną.
2. **Opóźnienie propagacji** (`delay`) — nawet po zarażeniu utwardzony węzeł
   przekazuje atak dalej dopiero po kilku krokach. To „spowalniacz": nie zmienia
   finalnego zasięgu, ale *opóźnia* rozprzestrzenianie się ataku.

Domyślnie `resistance=0.6` (bariera **częściowa**, nie pełna) — to celowy wybór:
przy pełnej odporności utwardzony węzeł nigdy się nie zaraża, więc opóźnienie nie
miałoby czego spowalniać. Częściowa odporność sprawia, że *oba* parametry są aktywne.

### Miara jakości obrony

Główną miarą jest **liczba węzłów dotkniętych atakiem do kroku T** (`affected_within_T`,
domyślnie T=10) — mniej = wolniejsza propagacja = lepsza obrona. Ta metryka jest
czuła na **oba** parametry: barierę (mniej węzłów pada) i opóźnienie (atak rozchodzi
się wolniej, więc do kroku T dociera do mniejszej liczby węzłów). Raportujemy też
**finalny zasięg** ataku (`total_infected`) jako metrykę pomocniczą — reaguje on
tylko na barierę, bo opóźnienie nie zmienia tego, ilu węzłów ostatecznie padnie.

### Główne założenia

- **Grafy:** domyślnie *dwa klastry Barabási–Albert połączone mostami*
  (`two_cluster`). Mosty mają niski stopień, więc naiwna obrona „utwardź huby"
  (degree) je przegapia — to tutaj GNN pokazuje przewagę. Dostępny też czysty
  model BA (`--graph_type ba`).
- **Model ataku:** propagacja typu SI / SIR po krawędziach grafu.
- **Budżet obrony:** utwardzamy tylko *k* wybranych węzłów.
- **Ewaluacja:** porównanie GNN z wyrocznią oraz baseline'ami (degree, betweenness, losowo).

### Wyniki (k=5, grafy dwuklastrowe)

Główna miara: średnia liczba węzłów **dotkniętych atakiem do kroku T=10**
(mniej = wolniejsza propagacja). Dla kontekstu podajemy też finalny zasięg ataku:

| strategia | dotknięci do T | finalny zasięg | czas decyzji |
|-----------|---------------:|---------------:|-------------:|
| brak obrony   | 33.9 | 35.7 | — |
| losowo        | 28.2 | 34.6 | 0.04 ms |
| **degree**    | 22.6 | 31.6 | 0.03 ms |
| **GNN**       | **19.4** | 26.5 | 0.66 ms |
| betweenness   | 18.4 | 25.2 | 2.35 ms |
| wyrocznia     | 19.2 | 27.0 | 16 247 ms |

GNN bije naiwny baseline `degree` o ~3 węzły i **dorównuje wyroczni** (19.4 vs 19.2),
podejmując decyzję **~24 000× szybciej** (0.66 ms vs ~16 s). Sieć sama nauczyła się
z cech, że kluczowe są węzły mostowe — czego `degree` nie widzi. Jedna ręczna miara,
betweenness, jest odrobinę lepsza (18.4), ale to gotowa reguła; model nie dostał jej
z góry, tylko sam doszedł do roli węzłów mostowych z surowych cech.

## Struktura projektu

```
gnn-network-defense/
├── src/
│   ├── graphs.py       # generowanie grafów (BA + dwuklastrowe z mostem)
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
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install torch_geometric
```

Cały pipeline odpala się czterema komendami:

```bash
# 1. dataset (jednorazowo, ~10 min — wyrocznia liczy etykiety Monte Carlo)
python -m src.dataset --n_graphs 200 --n_nodes 50 --n_sim 150 --seed 42

# 2. trening GCN (-> data/model.pt + krzywa uczenia)
python -m src.train --epochs 200

# 3. ewaluacja: GNN vs wyrocznia vs baseline (-> data/comparison.png)
python -m src.evaluate --k 5 --n_sim 100 --n_test 20

# 4. animacja propagacji: bez obrony vs obrona GNN (-> data/compare.gif)
python -m src.visualize --k 5
```

Domyślnie używane są grafy dwuklastrowe. Dla czystego modelu Barabási–Albert
dodaj `--graph_type ba` w kroku 1. Ewaluacja z wyrocznią jest wolna
(`--no_oracle` ją pomija).

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
