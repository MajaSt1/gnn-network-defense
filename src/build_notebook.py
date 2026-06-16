"""Generuje prezentację jako notebook Jupyter (prezentacja.ipynb).

Komórki markdown niosą narrację (te same treści co slajdy), komórki kodu
wyświetlają artefakty: krzywą uczenia, wykres porównawczy i animowany GIF
propagacji ataku. Po wykonaniu (nbconvert) obrazki są osadzone w pliku,
więc notebook „działa” od razu po otwarciu.

Narracja: naiwna obrona (degree) zawodzi → GNN sam odkrywa, że kluczowe są
węzły mostowe → dorównuje wolnej wyroczni, decydując natychmiast.

Użycie:
    python -m src.build_notebook        # -> prezentacja.ipynb
"""
from __future__ import annotations

import argparse

import nbformat as nbf


def build(out_path: str) -> None:
    nb = nbf.v4.new_notebook()
    cells = []
    md = lambda t: cells.append(nbf.v4.new_markdown_cell(t))
    code = lambda t: cells.append(nbf.v4.new_code_cell(t))

    # --- Tytuł ---
    md(
        "# Zabezpieczanie sieci komputerowych grafową siecią neuronową (GNN)\n"
        "\n"
        "**Wiktoria Białasik · Maja Stefanowicz**\n"
        "\n"
        "Matematyczne Fundamenty Informatyki · temat 13\n"
        "\n"
        "Mamy ograniczony budżet i możemy lepiej zabezpieczyć tylko kilka komputerów "
        "w sieci. Chcemy, żeby sieć neuronowa sama wskazała, które z nich wybrać, "
        "tak aby atak rozprzestrzeniał się jak najwolniej."
    )

    # --- 1. Problem ---
    md(
        "## 1. Problem\n"
        "\n"
        "Sieć komputerową przedstawiamy jako graf: komputery to wierzchołki, a połączenia "
        "między nimi to krawędzie. Atak (na przykład robak sieciowy) zaczyna się w jednym "
        "komputerze i przenosi się po połączeniach na kolejne, podobnie jak choroba zakaźna.\n"
        "\n"
        "Nie da się jednakowo zabezpieczyć całej sieci, bo to kosztuje. Możemy mocniej "
        "utwardzić tylko *k* wybranych komputerów. Cały problem sprowadza się więc do "
        "pytania: **które komputery wybrać, żeby atak rozszedł się jak najwolniej?**"
    )

    # --- 2. Model: graf i atak ---
    md(
        "## 2. Jak wygląda sieć i jak działa atak\n"
        "\n"
        "Testujemy nasze podejście na grafach złożonych z dwóch gęstych skupisk "
        "komputerów połączonych kilkoma pojedynczymi łączami (mostami). Taka struktura "
        "przypomina dwie podsieci spięte wąskim gardłem i dobrze pokazuje, gdzie obrona "
        "jest naprawdę ważna.\n"
        "\n"
        "Atak symulujemy modelem epidemicznym (SI/SIR): w każdym kroku zarażony komputer "
        "z pewnym prawdopodobieństwem zaraża swoich sąsiadów.\n"
        "\n"
        "Utwardzenie komputera daje mu dwie korzyści:\n"
        "\n"
        "1. **mniejszą podatność** — trudniej go zarazić,\n"
        "2. **opóźnienie** — nawet jeśli zostanie zarażony, przez jakiś czas nie zaraża dalej."
    )
    code(
        "# Przykładowy graf: dwa klastry + mosty\n"
        "import matplotlib.pyplot as plt\n"
        "import networkx as nx\n"
        "from src.graphs import generate_two_cluster_graph\n"
        "\n"
        "G = generate_two_cluster_graph(40, m=3, n_bridges=2, seed=1)\n"
        "pos = nx.spring_layout(G, seed=42)\n"
        "deg = dict(G.degree())\n"
        "sizes = [80 + 40 * deg[v] for v in G.nodes()]\n"
        "plt.figure(figsize=(8, 5))\n"
        "nx.draw_networkx_edges(G, pos, alpha=0.3)\n"
        "nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color='#bfd7ea')\n"
        "plt.title('Dwa skupiska komputerów połączone mostami')\n"
        "plt.axis('off'); plt.show()"
    )

    # --- 3. Wyrocznia ---
    md(
        "## 3. Punkt odniesienia: wyrocznia\n"
        "\n"
        "Żeby wiedzieć, jak dobry jest nasz model, potrzebujemy punktu odniesienia. "
        "Zbudowaliśmy więc wyrocznię, która znajduje bardzo dobry zestaw komputerów do "
        "utwardzenia. Robi to metodą prób i błędów: wielokrotnie symuluje atak dla różnych "
        "wyborów i sprawdza, który wypada najlepiej.\n"
        "\n"
        "Problem w tym, że jest bardzo wolna. Dla jednego grafu potrzebuje około 12 sekund, "
        "bo musi przeprowadzić tysiące symulacji. W praktyce nie da się jej tak używać.\n"
        "\n"
        "Dlatego traktujemy wyrocznię jak nauczyciela: pokazuje ona modelowi dobre "
        "odpowiedzi podczas treningu, a my chcemy, żeby sieć neuronowa nauczyła się "
        "podejmować podobne decyzje samodzielnie i znacznie szybciej."
    )

    # --- 4. Pomysł: GNN ---
    md(
        "## 4. Nasz model\n"
        "\n"
        "Każdy komputer opisujemy czterema liczbami mówiącymi o jego położeniu w sieci: "
        "liczbą połączeń, tym jak często leży na drogach między innymi komputerami, jak "
        "bardzo jego sąsiedzi są ze sobą połączeni oraz jak blisko jest do reszty sieci.\n"
        "\n"
        "Na tych danych uczymy grafową sieć neuronową (GCN). Dla każdego komputera zwraca "
        "ona liczbę od 0 do 1 mówiącą, jak bardzo opłaca się go utwardzić. Sieć uczy się, "
        "porównując swoje odpowiedzi z podpowiedziami wyroczni.\n"
        "\n"
        "Najważniejsza cecha takiej sieci jest taka, że przy liczeniu wyniku dla danego "
        "komputera bierze ona pod uwagę także jego sąsiadów. Dzięki temu nie patrzy na "
        "komputer w oderwaniu, ale rozumie jego rolę w całej strukturze. To pozwala jej "
        "zauważyć most: komputer, który ma niewiele połączeń, ale leży dokładnie między "
        "dwoma skupiskami."
    )

    # --- 5. Trening ---
    md(
        "## 5. Trening\n"
        "\n"
        "Model trenowaliśmy na 200 grafach, z których część odłożyliśmy na walidację, żeby "
        "sprawdzać go na sieciach, których nie widział podczas nauki. Na wykresie widać, że "
        "błąd spada i się stabilizuje, a wynik na walidacji nadąża za treningiem. To znaczy, "
        "że model faktycznie się uczy i nie zapamiętuje danych na pamięć.\n"
        "\n"
        "Po drodze trafiliśmy na ciekawy problem: na początku model wszystkim komputerom "
        "dawał niemal taką samą ocenę, bo podpowiedzi wyroczni były zbyt zaszumione. "
        "Poprawiliśmy sposób ich liczenia i wtedy model zaczął naprawdę rozróżniać komputery."
    )
    code(
        "from IPython.display import Image\n"
        "Image('data/learning_curve.png', width=720)"
    )

    # --- 6. Wyniki (kluczowy) ---
    md(
        "## 6. Wyniki\n"
        "\n"
        "Porównaliśmy nasz model z prostszymi sposobami wyboru komputerów do utwardzenia. "
        "W każdym przypadku utwardzamy 5 komputerów i sprawdzamy, ile z nich średnio zaraża "
        "atak. Im mniej, tym lepsza obrona.\n"
        "\n"
        "| sposób wyboru | średnio zarażonych | czas decyzji |\n"
        "|-----------|---------:|-------------:|\n"
        "| brak obrony | 34.9 | — |\n"
        "| losowo | 27.9 | 0.03 ms |\n"
        "| najwięcej połączeń | 22.0 | 0.02 ms |\n"
        "| **nasz model (GNN)** | **16.6** | 0.73 ms |\n"
        "| pozycja pośrednia | 15.5 | 2.41 ms |\n"
        "| wyrocznia | 15.0 | 12 208 ms |\n"
        "\n"
        "Najprostszy pomysł, czyli utwardzanie komputerów z największą liczbą połączeń, "
        "wypada słabo. Pomija mosty, bo te mają mało połączeń, a właśnie przez nie atak "
        "przedostaje się z jednego skupiska do drugiego.\n"
        "\n"
        "Nasz model radzi sobie znacznie lepiej. Sam nauczył się, że ważne są komputery "
        "leżące na styku skupisk, i osiąga wynik bliski wyroczni. Co najważniejsze, decyzję "
        "podejmuje w ułamku milisekundy, podczas gdy wyrocznia potrzebuje na to kilkunastu "
        "sekund.\n"
        "\n"
        "Dla uczciwości: jedna z prostych miar (pozycja pośrednia, czyli betweenness) jest "
        "tu odrobinę lepsza od naszego modelu. Warto jednak pamiętać, że model nie dostał "
        "tej reguły z góry, tylko sam doszedł do niej na podstawie danych."
    )
    code(
        "from IPython.display import Image\n"
        "Image('data/comparison.png', width=720)"
    )

    # --- 7. Wizualizacja ataku ---
    md(
        "## 7. Zobaczmy to w działaniu\n"
        "\n"
        "Na koniec ta sama symulacja ataku puszczona na dwóch identycznych sieciach. "
        "Atak startuje z tego samego komputera i przebiega tak samo. Jedyna różnica polega "
        "na tym, że po prawej stronie utwardziliśmy komputery wskazane przez nasz model.\n"
        "\n"
        "Po lewej stronie atak rozlewa się na obie części sieci. Po prawej zatrzymuje się "
        "na mostach i nie przedostaje się do drugiego skupiska."
    )
    code(
        "from IPython.display import Image\n"
        "Image('data/compare.gif')   # animacja odtwarza się w notebooku"
    )

    # --- 8. Wnioski ---
    md(
        "## 8. Podsumowanie\n"
        "\n"
        "Udało nam się nauczyć sieć neuronową wybierania komputerów do zabezpieczenia tak, "
        "żeby spowolnić atak. Model sam doszedł do tego, że kluczowe są komputery łączące "
        "różne części sieci, i radzi sobie prawie tak dobrze jak wolna wyrocznia, podejmując "
        "decyzję natychmiast.\n"
        "\n"
        "Trzeba pamiętać o ograniczeniach. Pracowaliśmy na sieciach generowanych "
        "sztucznie, a obrona jest jednorazowa, czyli wybieramy komputery raz, przed atakiem.\n"
        "\n"
        "Naturalne kierunki rozwoju to większe i bardziej realistyczne sieci, obrona "
        "reagująca już w trakcie ataku oraz uwzględnienie kierunku połączeń.\n"
        "\n"
        "Dziękujemy za uwagę."
    )

    nb['cells'] = cells
    nb.metadata['kernelspec'] = {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Zapisano notebook -> {out_path} ({len(cells)} komórek)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generuj prezentację jako notebook')
    parser.add_argument('--out', default='prezentacja.ipynb')
    args = parser.parse_args()
    build(args.out)
