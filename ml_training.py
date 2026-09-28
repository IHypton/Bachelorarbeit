import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

DATENSATZ_DATEI = "ml_datensatz.csv"
TRAIN_ANTEIL = 0.8

LR_MODELL_DATEI = "modell_logistische_regression.joblib"

SCHWELLENWERT_LR = 0.80

NICHT_ALS_FEATURE_VERWENDEN = {"instanz_seed", "spalten_id", "label"}


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


def drucke_ergebnis_tabelle(y_test, vorhersage):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, vorhersage, labels=[0, 1], zero_division=0
    )

    klassen_namen = {0: "nicht gewaehlt", 1: "gewaehlt"}

    print()
    print("Ergebnis auf dem Test-Set")
    print("-" * 60)
    print(f"{'Klasse':<16}{'Precision':>12}{'Recall':>12}{'F1-Score':>12}{'Anzahl':>10}")
    print("-" * 60)
    for i, klasse in enumerate([0, 1]):
        print(
            f"{klassen_namen[klasse]:<16}"
            f"{precision[i]:>12.2%}"
            f"{recall[i]:>12.2%}"
            f"{f1[i]:>12.2%}"
            f"{support[i]:>10}"
        )
    print("-" * 60)

    matrix = confusion_matrix(y_test, vorhersage, labels=[0, 1])
    print()
    print("Konfusionsmatrix (Zeile = tatsaechlich, Spalte = vorhergesagt)")
    print("-" * 60)
    print(f"{'':<24}{'vorhergesagt: 0':>18}{'vorhergesagt: 1':>18}")
    print(f"{'tatsaechlich: 0':<24}{matrix[0][0]:>18}{matrix[0][1]:>18}")
    print(f"{'tatsaechlich: 1':<24}{matrix[1][0]:>18}{matrix[1][1]:>18}")
    print("-" * 60)


def main():
    daten = pd.read_csv(DATENSATZ_DATEI)

    feature_spalten = [spalte for spalte in daten.columns if spalte not in NICHT_ALS_FEATURE_VERWENDEN]

    train, test = teile_instanzbasiert_auf(daten, TRAIN_ANTEIL)

    X_train = train[feature_spalten]
    y_train = train["label"]
    X_test = test[feature_spalten]
    y_test = test["label"]

    lr_modell = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="liblinear",
        random_state=42,
    )

    lr_modell.fit(X_train, y_train)

    wahrscheinlichkeiten_lr = lr_modell.predict_proba(X_test)[:, 1]
    vorhersage_lr = (wahrscheinlichkeiten_lr >= SCHWELLENWERT_LR).astype(int)

    print("Ergebnis LR")
    print()
    drucke_ergebnis_tabelle(y_test, vorhersage_lr)

    joblib.dump(
        {
            "modell": lr_modell,
            "feature_spalten": feature_spalten,
            "schwellenwert": SCHWELLENWERT_LR,
        },
        LR_MODELL_DATEI,
    )

    print()
    print("Modell gespeichert:")
    print(f"  {LR_MODELL_DATEI}")


if __name__ == "__main__":
    main()
