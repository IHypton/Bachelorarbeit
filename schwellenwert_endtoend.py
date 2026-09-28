import time

import joblib
import pandas as pd

from daten import (
    ARTIKEL_BEREICH,
    BASIS_SEED,
    BELASTUNG_BEREICH,
    DISTANZ_BEREICH,
    MAX_AUFZUEGE,
    PODS_PRO_ARTIKEL_BEREICH,
    STANDORTE_BEREICH,
    TOLERANZ,
)
from instanzgenerator import erzeuge_variable_instanz
from ml_features import berechne_spaltenfeatures
from solver import loese_modell
from spalten import erzeuge_spalten

LR_MODELL_DATEI = "modell_logistische_regression.joblib"
AUSGABE_DATEI = "schwellenwert_endtoend.md"

ANZAHL_TEST_INSTANZEN = 100
TEST_SEED_START = BASIS_SEED + 100_000

SCHWELLENWERTE = (0.30, 0.50, 0.70, 0.80, 0.85, 0.90, 0.95)

VERGLEICHS_TOLERANZ = 1e-4


def ist_gleichwertig(ergebnis, referenz):
    if ergebnis["status"] != "Optimal" or referenz["status"] != "Optimal":
        return False
    return (
        abs(ergebnis["maximale_belastung"] - referenz["maximale_belastung"])
        < VERGLEICHS_TOLERANZ
        and abs(ergebnis["gesamtdistanz"] - referenz["gesamtdistanz"])
        < VERGLEICHS_TOLERANZ
    )


def erzeuge_testinstanz(seed):
    instanz = erzeuge_variable_instanz(
        seed=seed,
        artikel_bereich=ARTIKEL_BEREICH,
        standorte_bereich=STANDORTE_BEREICH,
        pods_pro_artikel_bereich=PODS_PRO_ARTIKEL_BEREICH,
        belastung_bereich=BELASTUNG_BEREICH,
        distanz_bereich=DISTANZ_BEREICH,
    )
    spalten = erzeuge_spalten(
        artikel=instanz["artikel"], aufzuege=instanz["aufzuege"],
        pods_pro_artikel=instanz["pods_pro_artikel"],
        belastung=instanz["belastung"], distanz=instanz["distanz"],
    )
    return instanz, spalten


def loese(spalten, instanz):
    return loese_modell(
        spalten=spalten, artikel=instanz["artikel"], aufzuege=instanz["aufzuege"],
        toleranz=TOLERANZ, max_aufzuege=MAX_AUFZUEGE,
    )


def main():
    paket = joblib.load(LR_MODELL_DATEI)
    modell = paket["modell"]
    feature_spalten = paket["feature_spalten"]

    verfahren = ["ohne Filter", "Regelfilter"] + [
        f"LR @ {schwelle:.2f}" for schwelle in SCHWELLENWERTE
    ]
    protokoll = []

    for i in range(ANZAHL_TEST_INSTANZEN):
        seed = TEST_SEED_START + i
        instanz, spalten = erzeuge_testinstanz(seed)

        start = time.perf_counter()
        referenz = loese(spalten, instanz)
        zeit_voll = time.perf_counter() - start

        start = time.perf_counter()
        features = berechne_spaltenfeatures(instanz=instanz, spalten=spalten, max_aufzuege=MAX_AUFZUEGE)
        wahrscheinlichkeiten = modell.predict_proba(features[feature_spalten])[:, 1]
        zeit_features = time.perf_counter() - start

        masken = {
            "Regelfilter": (
                (features["primaer_zulaessig"] == 1)
                & (features["ist_distanzdominiert"] == 0)
            ).to_numpy(),
        }
        for schwelle in SCHWELLENWERTE:
            masken[f"LR @ {schwelle:.2f}"] = wahrscheinlichkeiten >= schwelle

        protokoll.append({
            "verfahren": "ohne Filter",
            "seed": seed,
            "spalten": len(spalten),
            "zeit": zeit_voll,
            "optimal": referenz["status"] == "Optimal",
        })

        for name, maske in masken.items():
            gefiltert = [s for s, ja in zip(spalten, maske, strict=True) if ja]

            start = time.perf_counter()
            ergebnis = loese(gefiltert, instanz) if gefiltert else {"status": "leer"}
            zeit_loesen = time.perf_counter() - start

            protokoll.append({
                "verfahren": name,
                "seed": seed,
                "spalten": len(gefiltert),
                "zeit": zeit_features + zeit_loesen,
                "optimal": ist_gleichwertig(ergebnis, referenz),
            })

        print(
            f"Instanz {i + 1}/{ANZAHL_TEST_INSTANZEN} (Seed {seed}): "
            f"{len(spalten)} Spalten, Referenzzeit {zeit_voll:.4f}s"
        )

    rahmen = pd.DataFrame(protokoll)
    zusammenfassung = rahmen.groupby("verfahren", sort=False).agg(
        spalten_mittel=("spalten", "mean"),
        zeit_gesamt=("zeit", "sum"),
        zeit_mittel=("zeit", "mean"),
        optimal=("optimal", "sum"),
    )
    zusammenfassung = zusammenfassung.reindex(verfahren)
    referenzzeit = zusammenfassung.loc["ohne Filter", "zeit_gesamt"]
    zusammenfassung["speedup"] = referenzzeit / zusammenfassung["zeit_gesamt"]
    zusammenfassung["optimal_anteil"] = zusammenfassung["optimal"] / ANZAHL_TEST_INSTANZEN

    schreibe_bericht(zusammenfassung)

    print()
    print(zusammenfassung.to_string())
    print()
    print(f"Ergebnisse geschrieben nach: {AUSGABE_DATEI}")


def schreibe_bericht(zusammenfassung):
    z = [
        "# End-to-End-Vergleich der Spaltenvorauswahl",
        "",
        "Automatisch erzeugt von `schwellenwert_endtoend.py`.",
        "",
        f"Testmenge: {ANZAHL_TEST_INSTANZEN} Instanzen mit den Seeds "
        f"`{TEST_SEED_START}` bis `{TEST_SEED_START + ANZAHL_TEST_INSTANZEN - 1}`. "
        "Diese Seeds liegen ausserhalb des Trainingsbereichs von "
        "`ml_datensatz.py`.",
        "",
        "## Fragestellung",
        "",
        "`ml_schwellenwert.py` bewertet die Vorhersage **je Spalte** "
        "(Precision/Recall). Fuer die Anwendung zaehlt aber etwas anderes: "
        "Findet der Solver auf der reduzierten Spaltenmenge noch die *echte* "
        "Optimalloesung? Dafuer muessen nicht alle optimalen Spalten "
        "ueberleben, sondern mindestens **eine vollstaendige** Optimalloesung. "
        "Ein hoher Spalten-Recall garantiert das nicht.",
        "",
        "Die Spalte *exakt optimal* zaehlt Instanzen, bei denen die gefilterte "
        "Loesung in maximaler Belastung **und** Gesamtdistanz mit der "
        "ungefilterten Referenzloesung uebereinstimmt.",
        "",
        "Die Zeitmessung der gefilterten Verfahren enthaelt die vollstaendige "
        "Pipeline: Featureberechnung, Modellauswertung und MIP-Loesung.",
        "",
        "## Ergebnisse",
        "",
        "| Verfahren | Spalten (Mittel) | Gesamtzeit (s) | Zeit/Instanz (s) "
        "| Speedup | exakt optimal |",
        "|---|---|---|---|---|---|",
    ]

    for name, reihe in zusammenfassung.iterrows():
        z.append(
            f"| {name} "
            f"| {reihe['spalten_mittel']:.1f} "
            f"| {reihe['zeit_gesamt']:.3f} "
            f"| {reihe['zeit_mittel']:.4f} "
            f"| {reihe['speedup']:.2f}x "
            f"| {int(reihe['optimal'])}/{ANZAHL_TEST_INSTANZEN} "
            f"({reihe['optimal_anteil']:.0%}) |"
        )

    z += [
        "",
        "## Einordnung",
        "",
        "Der **Regelfilter** verwendet ausschliesslich die beiden exakt "
        "hergeleiteten notwendigen Bedingungen "
        "`primaer_zulaessig == 1` und `ist_distanzdominiert == 0` "
        "(Herleitung in `features_dokumentation.md`, Abschnitte 4.5 und 4.7). "
        "Er kann per Konstruktion keine Optimalloesung verlieren und dient "
        "deshalb als verlustfreie Referenz.",
        "",
        "Beim gelernten Modell steuert der Schwellenwert den Kompromiss "
        "zwischen Geschwindigkeit und Loesungsqualitaet. Die Tabelle macht "
        "sichtbar, wie stark die Instanz-Optimalitaet einbricht, lange bevor "
        "der Spalten-Recall aus `schwellenwert_ergebnisse.md` das nahelegt.",
        "",
    ]

    with open(AUSGABE_DATEI, "w", encoding="utf-8") as datei:
        datei.write("\n".join(z).rstrip("\n") + "\n")


if __name__ == "__main__":
    main()
