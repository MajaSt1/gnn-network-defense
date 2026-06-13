"""Symulator propagacji ataku na grafie.

Odpowiedzialność (Osoba A):
- model propagacji SI / SIR po krawędziach grafu,
- uwzględnienie dwóch parametrów obrony utwardzonych węzłów:
    1) odporność na infekcję (niższe p zarażenia),
    2) opóźnienie propagacji (późniejsze przekazanie ataku),
- metryki przebiegu ataku (liczba zarażonych w czasie, czas do nasycenia).
"""
