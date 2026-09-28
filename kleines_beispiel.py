from spalten import erzeuge_spalten
from solver import loese_modell

artikel = [1, 2, 3]
aufzuege = ["L1", "L2"]

pods_pro_artikel = {
    1: ["A", "B"],
    2: ["C"],
    3: ["D"],
}

belastung = {
    "A": 3,
    "B": 4,
    "C": 5,
    "D": 2,
}

distanz = {
    "A": {"L1": 8, "L2": 6},
    "B": {"L1": 10, "L2": 5},
    "C": {"L1": 12, "L2": 9},
    "D": {"L1": 7, "L2": 11},
}

spalten = erzeuge_spalten(
    artikel=artikel, aufzuege=aufzuege, pods_pro_artikel=pods_pro_artikel,
    belastung=belastung, distanz=distanz,
)

print(f"Anzahl erzeugter Spalten: {len(spalten)}")

ergebnis = loese_modell(spalten=spalten, artikel=artikel, aufzuege=aufzuege)

print("Status:", ergebnis["status"])
print("Maximale Belastung", max(spalte["belastung"] for spalte in ergebnis["gewaehlte_spalten"]))
print("Gesamtdistanz:", ergebnis["gesamtdistanz"])
print("Gewählte Spalten:")
for spalte in ergebnis["gewaehlte_spalten"]:
    print(
        f"  Aufzug {spalte['aufzug']}: "
        f"Artikel {spalte['artikel']}, Pods {spalte['pods']}, "
        f"Belastung {spalte['belastung']}, Distanz {spalte['distanz']}"
    )
