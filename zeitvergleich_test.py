import time

import joblib
import pandas as pd

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

LR_MODELL_DATEI = "modell_logistische_regression.joblib"

ANZAHL_TEST_INSTANZEN = 100
TEST_SEED_START = BASIS_SEED + 100_000


def lade_modell(dateiname):
    return joblib.load(dateiname)


def reduziere_spalten(spalten, instanz, modell_paket):
    modell = modell_paket["modell"]
    feature_spalten = modell_paket["feature_spalten"]
    schwellenwert = modell_paket["schwellenwert"]

    alle_features = berechne_spaltenfeatures(instanz=instanz, spalten=spalten, max_aufzuege=MAX_AUFZUEGE)
    fehlende_features = [feature for feature in feature_spalten if feature not in alle_features.columns]
    if fehlende_features:
        raise ValueError(
            "Das gespeicherte Modell verwendet ein altes Feature-Schema. "
            "Bitte ml_datensatz.py und danach ml_training.py erneut "
            f"ausfuehren. Fehlend: {fehlende_features}"
        )
    X = alle_features[feature_spalten]

    wahrscheinlichkeiten = modell.predict_proba(X)[:, 1]
    markiert = wahrscheinlichkeiten >= schwellenwert

    return [spalte for spalte, ja in zip(spalten, markiert) if ja]


def main():
    lr_paket = lade_modell(LR_MODELL_DATEI)

    ergebnisse = []

    for i in range(ANZAHL_TEST_INSTANZEN):
        seed = TEST_SEED_START + i

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

        start = time.perf_counter()
        ergebnis_voll = loese_modell(
            spalten=spalten,
            artikel=instanz["artikel"],
            aufzuege=instanz["aufzuege"],
            toleranz=TOLERANZ,
            max_aufzuege=MAX_AUFZUEGE,
        )
        zeit_voll = time.perf_counter() - start

        start = time.perf_counter()
        spalten_lr = reduziere_spalten(spalten, instanz, lr_paket)
        ergebnis_lr = loese_modell(
            spalten=spalten_lr,
            artikel=instanz["artikel"],
            aufzuege=instanz["aufzuege"],
            toleranz=TOLERANZ,
            max_aufzuege=MAX_AUFZUEGE,
        )
        zeit_lr = time.perf_counter() - start

        gleich_lr = (
            ergebnis_lr["status"] == "Optimal"
            and ergebnis_voll["status"] == "Optimal"
            and abs(ergebnis_lr["maximale_belastung"] - ergebnis_voll["maximale_belastung"]) < 1e-4
            and abs(ergebnis_lr["gesamtdistanz"] - ergebnis_voll["gesamtdistanz"]) < 1e-4
        )

        ergebnisse.append({
            "seed": seed,
            "anzahl_spalten_voll": len(spalten),
            "anzahl_spalten_lr": len(spalten_lr),
            "zeit_voll": zeit_voll,
            "zeit_lr": zeit_lr,
            "gleiche_loesung_lr": gleich_lr,
        })

        print(
            f"Instanz {i+1}/{ANZAHL_TEST_INSTANZEN} (Seed {seed}): "
            f"{len(spalten)} Spalten -> LR {len(spalten_lr)} | "
            f"Zeit voll {zeit_voll:.4f}s, LR {zeit_lr:.4f}s"
        )

    drucke_zusammenfassung(ergebnisse)


def drucke_zusammenfassung(ergebnisse):
    df = pd.DataFrame(ergebnisse)

    zeit_voll_gesamt = df["zeit_voll"].sum()
    zeit_lr_gesamt = df["zeit_lr"].sum()

    print()
    print("=" * 56)
    print("Zusammenfassung Zeitvergleich")
    print("=" * 56)
    print(f"{'':<28}{'Ohne Filter':>14}{'Mit LR':>14}")
    print("-" * 56)
    print(f"{'Gesamtzeit (s)':<28}{zeit_voll_gesamt:>14.4f}{zeit_lr_gesamt:>14.4f}")
    print(
        f"{'Zeit pro Instanz (s)':<28}"
        f"{df['zeit_voll'].mean():>14.4f}"
        f"{df['zeit_lr'].mean():>14.4f}"
    )
    print(
        f"{'Speedup ggue. ohne Filter':<28}"
        f"{'1.00x':>14}"
        f"{zeit_voll_gesamt / zeit_lr_gesamt:>13.2f}x"
    )
    print(
        f"{'Spalten im Schnitt':<28}"
        f"{df['anzahl_spalten_voll'].mean():>14.1f}"
        f"{df['anzahl_spalten_lr'].mean():>14.1f}"
    )
    print("-" * 56)

    treffer_lr = df["gleiche_loesung_lr"].sum()
    anzahl = len(df)

    print()
    print("Loesungsqualitaet (stimmt Ergebnis mit der echten Optimalloesung ueberein?)")
    print("-" * 56)
    print(f"LR: {treffer_lr}/{anzahl} Instanzen exakt optimal ({treffer_lr/anzahl:.1%})")
    print("-" * 56)


if __name__ == "__main__":
    main()
