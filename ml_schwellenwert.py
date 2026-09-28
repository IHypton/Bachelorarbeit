import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

DATENSATZ_DATEI = "ml_datensatz.csv"
TRAIN_ANTEIL = 0.8

SCHWELLENWERT_MIN = 0.05
SCHWELLENWERT_MAX = 1
SCHWELLENWERT_SCHRITT = 0.05

AUSGABE_DATEI = "schwellenwert_ergebnisse.md"

NICHT_ALS_FEATURE_VERWENDEN = {"instanz_seed", "spalten_id", "label"}

MODELL_NAME = "Logistische Regression"


def teile_instanzbasiert_auf(daten, train_anteil):
    instanz_seeds = sorted(daten["instanz_seed"].unique())
    grenze = int(len(instanz_seeds) * train_anteil)

    train_seeds = set(instanz_seeds[:grenze])
    test_seeds = set(instanz_seeds[grenze:])

    train = daten[daten["instanz_seed"].isin(train_seeds)]
    test = daten[daten["instanz_seed"].isin(test_seeds)]

    print(f"Training: {len(train_seeds)} Instanzen, {len(train)} Zeilen")
    print(f"Test:     {len(test_seeds)} Instanzen, {len(test)} Zeilen")

    return train, test


def erstelle_modell():
    return LogisticRegression(class_weight="balanced", max_iter=2000, solver="liblinear", random_state=42)


def erzeuge_schwellenwerte():
    anzahl_schritte = round((SCHWELLENWERT_MAX - SCHWELLENWERT_MIN) / SCHWELLENWERT_SCHRITT)
    return [round(SCHWELLENWERT_MIN + i * SCHWELLENWERT_SCHRITT, 4) for i in range(anzahl_schritte + 1)]


def berechne_metriken(y_test, wahrscheinlichkeiten, schwellenwert):
    vorhersage = (wahrscheinlichkeiten >= schwellenwert).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, vorhersage, labels=[0, 1], zero_division=0
    )
    matrix = confusion_matrix(y_test, vorhersage, labels=[0, 1])

    return {
        "schwellenwert": schwellenwert,
        "precision_0": precision[0],
        "recall_0": recall[0],
        "f1_0": f1[0],
        "precision_1": precision[1],
        "recall_1": recall[1],
        "f1_1": f1[1],
        "tn": matrix[0][0],
        "fp": matrix[0][1],
        "fn": matrix[1][0],
        "tp": matrix[1][1],
    }


def formatiere_markdown_abschnitt(modell_name, ergebnisse):
    zeilen = [
        f"## {modell_name}",
        "",
        "| Schwellenwert | Precision (gewählt) | Recall (gewählt) | F1 (gewählt) "
        "| Precision (nicht gewählt) | Recall (nicht gewählt) | F1 (nicht gewählt) "
        "| TP | FP | FN | TN |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for e in ergebnisse:
        zeilen.append(
            f"| {e['schwellenwert']:.2f} "
            f"| {e['precision_1']:.2%} | {e['recall_1']:.2%} | {e['f1_1']:.2%} "
            f"| {e['precision_0']:.2%} | {e['recall_0']:.2%} | {e['f1_0']:.2%} "
            f"| {e['tp']} | {e['fp']} | {e['fn']} | {e['tn']} |"
        )

    return zeilen


def schreibe_markdown_datei(pfad, abschnitte):
    zeilen = ["# Schwellenwert-Vergleich", ""]
    for abschnitt in abschnitte:
        zeilen.extend(abschnitt)
        zeilen.append("")

    with open(pfad, "w", encoding="utf-8") as datei:
        datei.write("\n".join(zeilen).rstrip("\n") + "\n")


def main():
    daten = pd.read_csv(DATENSATZ_DATEI)

    feature_spalten = [spalte for spalte in daten.columns if spalte not in NICHT_ALS_FEATURE_VERWENDEN]

    train, test = teile_instanzbasiert_auf(daten, TRAIN_ANTEIL)

    X_train = train[feature_spalten]
    y_train = train["label"]
    X_test = test[feature_spalten]
    y_test = test["label"]

    schwellenwerte = erzeuge_schwellenwerte()

    print(f"Trainiere {MODELL_NAME} ...")

    modell = erstelle_modell()
    modell.fit(X_train, y_train)

    wahrscheinlichkeiten = modell.predict_proba(X_test)[:, 1]

    ergebnisse = [
        berechne_metriken(y_test, wahrscheinlichkeiten, schwellenwert)
        for schwellenwert in schwellenwerte
    ]

    abschnitte = [formatiere_markdown_abschnitt(MODELL_NAME, ergebnisse)]

    schreibe_markdown_datei(AUSGABE_DATEI, abschnitte)

    print()
    print(f"Ergebnisse für {len(schwellenwerte)} Schwellenwerte je Modell geschrieben nach: {AUSGABE_DATEI}")


if __name__ == "__main__":
    main()
