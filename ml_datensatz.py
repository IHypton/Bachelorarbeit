import csv

from instanzgenerator import erzeuge_variable_instanz
from spalten import erzeuge_spalten
from solver import loese_modell
from ml_features import berechne_spaltenfeatures


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

ANZAHL_INSTANZEN = 1000


def erzeuge_datensatz():
    daten = []
    poollimit_instanzen = []

    for i in range(ANZAHL_INSTANZEN):
        seed = BASIS_SEED + i

        instanz = erzeuge_variable_instanz(
            seed=seed,
            artikel_bereich=ARTIKEL_BEREICH,
            standorte_bereich=STANDORTE_BEREICH,
            pods_pro_artikel_bereich=PODS_PRO_ARTIKEL_BEREICH,
            belastung_bereich=BELASTUNG_BEREICH,
            distanz_bereich=DISTANZ_BEREICH,
        )

        spalten = erzeuge_spalten(
            artikel=instanz["artikel"],
            aufzuege=instanz["aufzuege"],
            pods_pro_artikel=instanz["pods_pro_artikel"],
            belastung=instanz["belastung"],
            distanz=instanz["distanz"],
        )

        ergebnis = loese_modell(
            spalten=spalten,
            artikel=instanz["artikel"],
            aufzuege=instanz["aufzuege"],
            toleranz=TOLERANZ,
            max_aufzuege=MAX_AUFZUEGE,
            alle_optima=True,
            max_pool_loesungen=1000,
        )

        if ergebnis["status"] != "Optimal":
            print(f"Instanz {i} (Seed {seed}): keine optimale Lösung, übersprungen")
            continue

        if ergebnis["pool_limit_erreicht"]:
            poollimit_instanzen.append(seed)

        optimale_ids = ergebnis["optimale_spalten_ids"]

        feature_tabelle = berechne_spaltenfeatures(
            instanz=instanz,
            spalten=spalten,
            max_aufzuege=MAX_AUFZUEGE,
        )

        for spalte, features in zip(spalten, feature_tabelle.to_dict(orient="records"), strict=True):
            daten.append({
                "instanz_seed": seed,
                "spalten_id": spalte["id"],
                **features,
                "label": int(spalte["id"] in optimale_ids),
            })

        print(
            f"Instanz {i+1}/{ANZAHL_INSTANZEN} (Seed {seed}): "
            f"{len(spalten)} Spalten, {len(optimale_ids)} positive Labels"
        )

    return daten, poollimit_instanzen


def main():
    daten, poollimit_instanzen = erzeuge_datensatz()

    if not daten:
        raise RuntimeError("Es wurden keine Daten erzeugt.")

    with open("ml_datensatz.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(daten[0].keys()))
        writer.writeheader()
        writer.writerows(daten)

    positive = sum(zeile["label"] for zeile in daten)
    print()
    print(f"Fertig: {len(daten)} Zeilen, davon {positive} positiv ({positive/len(daten):.2%})")
    print("Gespeichert in: ml_datensatz.csv")

    if poollimit_instanzen:
        print()
        print(
            f"WARNUNG: Bei {len(poollimit_instanzen)} Instanzen wurde das "
            "Poollimit erreicht. Dort ist die Menge der optimalen Spalten "
            "und damit das Label moeglicherweise unvollstaendig. "
            "max_pool_loesungen erhoehen und neu erzeugen."
        )
        print(f"Betroffene Seeds: {poollimit_instanzen}")
    else:
        print("Poollimit bei keiner Instanz erreicht; Labels sind vollstaendig.")


if __name__ == "__main__":
    main()
