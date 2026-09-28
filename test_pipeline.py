from instanzgenerator import erzeuge_variable_instanz
from spalten import erzeuge_spalten
from solver import loese_modell

from daten import (
    ARTIKEL_BEREICH,
    STANDORTE_BEREICH,
    PODS_PRO_ARTIKEL_BEREICH,
    BELASTUNG_BEREICH,
    DISTANZ_BEREICH,
    BASIS_SEED,
    TOLERANZ,
    MAX_AUFZUEGE,
)

instanz = erzeuge_variable_instanz(
    seed=BASIS_SEED,
    artikel_bereich=ARTIKEL_BEREICH,
    standorte_bereich=STANDORTE_BEREICH,
    pods_pro_artikel_bereich=PODS_PRO_ARTIKEL_BEREICH,
    belastung_bereich=BELASTUNG_BEREICH,
    distanz_bereich=DISTANZ_BEREICH,
)

print(f"Instanz erzeugt: {instanz['anzahl_artikel']} Artikel, "
      f"{instanz['anzahl_aufzuege']} Standorte, "
      f"{instanz['anzahl_pods_gesamt']} Pods")

spalten = erzeuge_spalten(
    artikel=instanz["artikel"],
    aufzuege=instanz["aufzuege"],
    pods_pro_artikel=instanz["pods_pro_artikel"],
    belastung=instanz["belastung"],
    distanz=instanz["distanz"],
)

print(f"Spalten erzeugt: {len(spalten)}")

ergebnis = loese_modell(
    spalten=spalten,
    artikel=instanz["artikel"],
    aufzuege=instanz["aufzuege"],
    toleranz=TOLERANZ,
    max_aufzuege=MAX_AUFZUEGE,
)

print("Status:", ergebnis["status"])
print("Maximale Belastung:", ergebnis["maximale_belastung"])
print("Gesamtdistanz:", ergebnis["gesamtdistanz"])
print(f"Gewählte Spalten: {len(ergebnis['gewaehlte_spalten'])} von {len(spalten)}")
for spalte in ergebnis["gewaehlte_spalten"]:
    print(f"  Aufzug {spalte['aufzug']}: Artikel {spalte['artikel']}, Pods {spalte['pods']}")
