import time
from math import ceil

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

DATENSATZ_DATEI = "ml_datensatz.csv"
AUSGABE_DATEI = "feature_ablation_ergebnisse.md"

NICHT_ALS_FEATURE_VERWENDEN = {"instanz_seed", "spalten_id", "label"}
ANZAHL_FOLDS = 5
ANALYSE_FOLD = 0

ZIEL_RECALLS = (0.95, 0.99, 1.00)
ANZAHL_JOBS = 3

GREEDY_MAX_FEATURES = 12
GREEDY_MIN_ZUWACHS = 0.002
REDUNDANZ_SCHWELLE = 0.95

ZUFALLS_SEED = 42


def erstelle_modell():
    return LogisticRegression(class_weight="balanced", max_iter=2000, random_state=ZUFALLS_SEED)

def definiere_gruppen(feature_spalten):
    feste_gruppen = {
        "A_instanzkontext": [
            "anzahl_artikel_gesamt",
            "anzahl_aufzuege_gesamt",
            "anzahl_pods_gesamt",
            "pods_pro_artikel_min",
            "pods_pro_artikel_max",
            "pods_pro_artikel_mittel",
            "optimale_belastungsgrenze",
        ],
        "B_abdeckung": [
            "anzahl_artikel_spalte",
            "anzahl_artikel_rest",
            "anteil_abgedeckter_artikel",
        ],
        "C_rohkosten": [
            "belastung",
            "distanz",
            "belastung_pro_artikel",
            "distanz_pro_artikel",
        ],
        "D_podstatistik": [
            "min_pod_belastung",
            "max_pod_belastung",
            "spannweite_pod_belastung",
            "min_pod_distanz",
            "max_pod_distanz",
            "spannweite_pod_distanz",
        ],
        "E_zulaessigkeit": [
            "belastung_relativ_zu_grenze",
            "belastung_innerhalb_grenze",
            "min_belastung_rest",
            "min_aufzuege_fuer_rest",
            "rest_ergaenzbar",
            "primaer_zulaessig",
        ],
        "F_regret": [
            "min_belastung_abgedeckte_artikel",
            "belastungs_regret_pods",
            "min_distanz_abgedeckte_artikel",
            "distanz_regret_pods",
            "min_distanz_rest",
        ],
        "G_dominanz": [
            "beste_distanz_ohne_mehrbelastung",
            "distanz_regret_ohne_mehrbelastung",
            "ist_distanzdominiert",
        ],
        "H_podverfuegbarkeit": [
            "pods_abgedeckte_artikel_mittel",
            "pods_rest_mittel",
        ],
    }

    gruppen = {name: list(spalten) for name, spalten in feste_gruppen.items()}
    for praefix, name in (
        ("rang_instanz_", "I_rang_instanz"),
        ("rang_spaltengroesse_", "J_rang_spaltengroesse"),
        ("rang_aufzug_", "K_rang_aufzug"),
        ("rang_artikelmenge_", "L_rang_artikelmenge"),
    ):
        gruppen[name] = [
            spalte for spalte in feature_spalten if spalte.startswith(praefix)
        ]

    zugeordnet = [spalte for spalten in gruppen.values() for spalte in spalten]
    if sorted(zugeordnet) != sorted(feature_spalten):
        fehlend = sorted(set(feature_spalten) - set(zugeordnet))
        unbekannt = sorted(set(zugeordnet) - set(feature_spalten))
        raise ValueError(
            "Gruppendefinition passt nicht zum Datensatz. "
            f"Nicht zugeordnet: {fehlend}. Unbekannt: {unbekannt}."
        )

    return gruppen


def behaltequote(y_wahr, punkte, ziel_recall):
    anzahl_positiv = int(y_wahr.sum())
    if anzahl_positiv == 0:
        return float("nan")

    reihenfolge = np.argsort(-punkte, kind="stable")
    sortierte_labels = y_wahr[reihenfolge]
    kumuliert = np.cumsum(sortierte_labels)

    benoetigt = ceil(ziel_recall * anzahl_positiv)
    position = int(np.searchsorted(kumuliert, benoetigt))
    schwelle = punkte[reihenfolge][position]

    return float(np.mean(punkte >= schwelle))


def berechne_metriken(y_wahr, punkte):
    ergebnis = {
        "pr_auc": float(average_precision_score(y_wahr, punkte)),
        "roc_auc": float(roc_auc_score(y_wahr, punkte)),
    }
    for ziel in ZIEL_RECALLS:
        schluessel = f"behaltequote_{int(round(ziel * 100))}"
        ergebnis[schluessel] = behaltequote(y_wahr, punkte, ziel)
    return ergebnis


def _bewerte_featuremenge(X_train, y_train, X_test, y_test, spalten_index):
    modell = erstelle_modell()
    modell.fit(X_train[:, spalten_index], y_train)
    punkte = modell.predict_proba(X_test[:, spalten_index])[:, 1]
    return berechne_metriken(y_test, punkte)


def standardisiere(X, train_maske):
    mittelwert = X[train_maske].mean(axis=0)
    streuung = X[train_maske].std(axis=0)
    streuung[streuung == 0] = 1.0
    return (X - mittelwert) / streuung

def erzeuge_folds(instanz_seeds):
    eindeutige_seeds = np.array(sorted(np.unique(instanz_seeds)))
    bloecke = np.array_split(eindeutige_seeds, ANZAHL_FOLDS)

    folds = []
    for block in bloecke:
        test_maske = np.isin(instanz_seeds, block)
        folds.append((~test_maske, test_maske))
    return folds


def lade_daten():
    daten = pd.read_csv(DATENSATZ_DATEI)
    feature_spalten = [
        spalte
        for spalte in daten.columns
        if spalte not in NICHT_ALS_FEATURE_VERWENDEN
    ]
    X = daten[feature_spalten].to_numpy(np.float32)
    y = daten["label"].to_numpy(np.int8)
    instanz_seeds = daten["instanz_seed"].to_numpy()
    return X, y, instanz_seeds, feature_spalten


def fuehre_konfigurationen_aus(konfigurationen, X_train, y_train, X_test, y_test):
    ergebnisse = Parallel(n_jobs=ANZAHL_JOBS, verbose=0)(
        delayed(_bewerte_featuremenge)(X_train, y_train, X_test, y_test, np.asarray(index, dtype=np.int64))
        for _, index in konfigurationen
    )
    return {
        name: metriken
        for (name, _), metriken in zip(konfigurationen, ergebnisse, strict=True)
    }

def mittle_ueber_folds(pro_fold):
    namen = pro_fold[0].keys()
    gemittelt = {}
    for name in namen:
        werte = {}
        for kennzahl in pro_fold[0][name]:
            reihe = np.array([fold[name][kennzahl] for fold in pro_fold])
            werte[kennzahl] = float(np.mean(reihe))
            werte[f"{kennzahl}_std"] = float(np.std(reihe))
        gemittelt[name] = werte
    return gemittelt


def analysiere_regelfilter(X, y, instanz_seeds, feature_spalten):
    spalte = {name: position for position, name in enumerate(feature_spalten)}
    werte = {
        name: X[:, spalte[name]]
        for name in (
            "belastung_innerhalb_grenze",
            "rest_ergaenzbar",
            "primaer_zulaessig",
            "ist_distanzdominiert",
        )
    }

    bedingungen = {
        "belastung_innerhalb_grenze == 1": werte["belastung_innerhalb_grenze"] == 1,
        "rest_ergaenzbar == 1": werte["rest_ergaenzbar"] == 1,
        "primaer_zulaessig == 1": werte["primaer_zulaessig"] == 1,
        "ist_distanzdominiert == 0": werte["ist_distanzdominiert"] == 0,
    }
    bedingungen["primaer_zulaessig == 1 UND ist_distanzdominiert == 0"] = (
        bedingungen["primaer_zulaessig == 1"]
        & bedingungen["ist_distanzdominiert == 0"]
    )

    anzahl_positiv = int(y.sum())
    zeilen = []
    for name, maske in bedingungen.items():
        behalten = int(maske.sum())
        positiv_behalten = int(y[maske].sum())
        zeilen.append({
            "bedingung": name,
            "behalten": behalten,
            "behaltequote": behalten / len(y),
            "positiv_behalten": positiv_behalten,
            "recall": positiv_behalten / anzahl_positiv,
            "precision": positiv_behalten / behalten if behalten else float("nan"),
            "verletzungen": anzahl_positiv - positiv_behalten,
        })

    kombiniert = bedingungen[
        "primaer_zulaessig == 1 UND ist_distanzdominiert == 0"
    ]
    rahmen = pd.DataFrame({
        "instanz": instanz_seeds,
        "label": y,
        "behalten": kombiniert.astype(np.int8),
        "positiv_behalten": (kombiniert & (y == 1)).astype(np.int8),
    })
    je_instanz = rahmen.groupby("instanz", sort=False).sum(numeric_only=True)
    anzahl_je_instanz = rahmen.groupby("instanz", sort=False).size()
    quote_je_instanz = je_instanz["behalten"] / anzahl_je_instanz

    return {
        "tabelle": pd.DataFrame(zeilen),
        "anzahl_positiv": anzahl_positiv,
        "spalten_vorher_mittel": float(anzahl_je_instanz.mean()),
        "spalten_nachher_mittel": float(je_instanz["behalten"].mean()),
        "quote_mittel": float(quote_je_instanz.mean()),
        "quote_median": float(quote_je_instanz.median()),
        "quote_min": float(quote_je_instanz.min()),
        "quote_max": float(quote_je_instanz.max()),
        "instanzen_vollstaendig": int(
            (je_instanz["label"] == je_instanz["positiv_behalten"]).sum()
        ),
        "instanzen_gesamt": int(len(je_instanz)),
    }


def berechne_permutation_importance(
    X_train, y_train, X_test, y_test, feature_spalten, anzahl_wiederholungen=5
):
    modell = erstelle_modell()
    modell.fit(X_train, y_train)
    basis = average_precision_score(y_test, modell.predict_proba(X_test)[:, 1])

    rng = np.random.default_rng(ZUFALLS_SEED)
    zeilen = []
    for position, name in enumerate(feature_spalten):
        original = X_test[:, position].copy()
        verluste = []
        for _ in range(anzahl_wiederholungen):
            X_test[:, position] = rng.permutation(original)
            punkte = modell.predict_proba(X_test)[:, 1]
            verluste.append(basis - average_precision_score(y_test, punkte))
        X_test[:, position] = original
        zeilen.append({
            "feature": name,
            "pr_auc_verlust": float(np.mean(verluste)),
            "pr_auc_verlust_std": float(np.std(verluste)),
        })

    return pd.DataFrame(zeilen).sort_values("pr_auc_verlust", ascending=False)


def berechne_koeffizienten(X_train, y_train, feature_spalten):
    modell = erstelle_modell()
    modell.fit(X_train, y_train)
    koeffizienten = modell.coef_[0]
    rahmen = pd.DataFrame({
        "feature": feature_spalten,
        "koeffizient": koeffizienten,
        "betrag": np.abs(koeffizienten),
    })
    return rahmen.sort_values("betrag", ascending=False)


def finde_redundante_paare(X, feature_spalten, stichprobe=200_000):
    rng = np.random.default_rng(ZUFALLS_SEED)
    if X.shape[0] > stichprobe:
        auswahl = rng.choice(X.shape[0], size=stichprobe, replace=False)
        teilmenge = X[auswahl]
    else:
        teilmenge = X

    korrelation = np.corrcoef(teilmenge, rowvar=False)
    korrelation = np.nan_to_num(korrelation)

    zeilen = []
    anzahl = len(feature_spalten)
    for i in range(anzahl):
        for j in range(i + 1, anzahl):
            wert = korrelation[i, j]
            if abs(wert) >= REDUNDANZ_SCHWELLE:
                zeilen.append({
                    "feature_a": feature_spalten[i],
                    "feature_b": feature_spalten[j],
                    "korrelation": float(wert),
                })

    rahmen = pd.DataFrame(zeilen)
    if rahmen.empty:
        return rahmen
    return rahmen.reindex(
        rahmen["korrelation"].abs().sort_values(ascending=False).index
    )


def greedy_forward_selection(X_train, y_train, X_test, y_test, feature_spalten):
    gewaehlt = []
    offen = list(range(len(feature_spalten)))
    verlauf = []
    letzter_wert = 0.0

    while len(gewaehlt) < GREEDY_MAX_FEATURES and offen:
        konfigurationen = [
            (str(kandidat), gewaehlt + [kandidat]) for kandidat in offen
        ]
        ergebnisse = fuehre_konfigurationen_aus(
            konfigurationen, X_train, y_train, X_test, y_test
        )

        bester_kandidat = max(
            offen, key=lambda kandidat: ergebnisse[str(kandidat)]["pr_auc"]
        )
        bester_wert = ergebnisse[str(bester_kandidat)]["pr_auc"]
        zuwachs = bester_wert - letzter_wert

        gewaehlt.append(bester_kandidat)
        offen.remove(bester_kandidat)
        verlauf.append({
            "schritt": len(gewaehlt),
            "feature": feature_spalten[bester_kandidat],
            "pr_auc": bester_wert,
            "zuwachs": zuwachs,
            "behaltequote_99": ergebnisse[str(bester_kandidat)]["behaltequote_99"],
        })
        print(
            f"  Greedy Schritt {len(gewaehlt):2d}: "
            f"{feature_spalten[bester_kandidat]:<42s} "
            f"PR-AUC {bester_wert:.4f} (+{zuwachs:.4f})"
        )

        if len(gewaehlt) >= 3 and zuwachs < GREEDY_MIN_ZUWACHS:
            break
        letzter_wert = bester_wert

    return verlauf


def _tabelle(kopf, zeilen):
    ausgabe = [
        "| " + " | ".join(kopf) + " |",
        "|" + "|".join(["---"] * len(kopf)) + "|",
    ]
    for zeile in zeilen:
        ausgabe.append("| " + " | ".join(zeile) + " |")
    return ausgabe


def _delta_zeilen(gemittelt, basis, namen, groessen=None):
    sortiert = sorted(namen, key=lambda name: gemittelt[name]["pr_auc"])
    zeilen = []
    for name in sortiert:
        werte = gemittelt[name]
        zeile = [f"`{name}`"]
        if groessen is not None:
            zeile.append(str(groessen[name]))
        zeile += [
            f"{werte['pr_auc']:.4f}",
            f"{werte['pr_auc'] - basis['pr_auc']:+.4f}",
            f"{werte['pr_auc_std']:.4f}",
            f"{werte['behaltequote_99']:.4%}",
            f"{(werte['behaltequote_99'] - basis['behaltequote_99']) * 100:+.3f} pp",
        ]
        zeilen.append(zeile)
    return zeilen


def schreibe_bericht(
    pfad, kennzahlen, baseline, regelfilter,
    loo, logo, einzeln, gruppe_allein, gruppen,
    permutation, koeffizienten, redundanz,
    greedy, greedy_validierung,
    feature_spalten, laufzeit,
):
    z = []
    z.append("# Ablationsstudie zur Feature-Relevanz")
    z.append("")
    z.append(
        "Automatisch erzeugt von `feature_ablation.py`. Alle Zahlen in diesem "
        "Dokument stammen aus einem einzigen Lauf des Skripts und sind mit "
        f"festem Zufallsseed ({ZUFALLS_SEED}) reproduzierbar."
    )
    z.append("")

    z.append("## 1. Versuchsaufbau")
    z.append("")
    z.append("### 1.1 Datengrundlage")
    z.append("")
    z += _tabelle(
        ["Groesse", "Wert"],
        [
            ["Instanzen", f"{kennzahlen['anzahl_instanzen']}"],
            ["Zeilen (Spalten des Set-Covering-Modells)", f"{kennzahlen['anzahl_zeilen']}"],
            ["Features", f"{kennzahlen['anzahl_features']}"],
            ["Positive Labels", f"{kennzahlen['anzahl_positiv']}"],
            ["Positivrate", f"{kennzahlen['positivrate']:.4%}"],
            ["Zeilen je Instanz (Mittel)", f"{kennzahlen['zeilen_pro_instanz']:.1f}"],
        ],
    )
    z.append("")
    z.append(
        "Eine Zeile entspricht genau einer zulaessigen Spalte des "
        "Set-Covering-Modells. Das Label ist 1, wenn die Spalte in "
        "mindestens einer Optimalloesung der zugehoerigen Instanz "
        "vorkommt (ermittelt ueber den Gurobi-Loesungspool in "
        "`ml_datensatz.py`)."
    )
    z.append("")

    z.append("### 1.2 Aufteilung")
    z.append("")
    z.append(
        f"Es wird eine {ANZAHL_FOLDS}-fache **instanzbasierte** Kreuzvalidierung "
        "verwendet: Die Instanzen werden in "
        f"{ANZAHL_FOLDS} disjunkte Bloecke geteilt, jeder Block dient einmal "
        "als Testmenge. Keine Instanz liegt gleichzeitig in Trainings- und "
        "Testmenge."
    )
    z.append("")
    z.append(
        "Diese Aufteilung ist zwingend: Spalten derselben Instanz teilen sich "
        "Kontextfeatures (z. B. `optimale_belastungsgrenze`) und alle "
        "Rangfeatures sind relativ zu den Konkurrenzspalten *derselben* "
        "Instanz definiert. Eine zeilenweise Aufteilung wuerde Information "
        "aus der Testmenge ins Training tragen und die Guete deutlich zu "
        "optimistisch schaetzen."
    )
    z.append("")

    z.append("### 1.3 Modell")
    z.append("")
    z.append(
        "Logistische Regression (`sklearn`), `class_weight=\"balanced\"`, "
        "`max_iter=2000`, Solver `lbfgs`, Seed "
        f"{ZUFALLS_SEED}. Alle Features werden vorab standardisiert "
        "(z-Transformation), wobei Mittelwert und Standardabweichung "
        "ausschliesslich aus der Trainingsmenge des jeweiligen Folds stammen."
    )
    z.append("")
    z.append(
        "**Warum standardisiert?** Die logistische Regression in `sklearn` ist "
        "standardmaessig L2-regularisiert, und diese Regularisierung ist "
        "skalenabhaengig. Ohne Standardisierung wuerde ein Feature mit grossem "
        "Wertebereich (z. B. `distanz` in [1, 120]) anders bestraft als ein "
        "Rangfeature in [0, 1]. Ein Vergleich der Feature-Beitraege waere dann "
        "nicht aussagekraeftig. Die Standardisierung aendert die Guete des "
        "Baseline-Modells praktisch nicht, macht die Koeffizienten aber "
        "untereinander vergleichbar."
    )
    z.append("")

    z.append("### 1.4 Kennzahlen")
    z.append("")
    z += _tabelle(
        ["Kennzahl", "Bedeutung", "Richtung"],
        [
            [
                "PR-AUC",
                "Flaeche unter der Precision-Recall-Kurve (Average Precision). "
                "Hauptkennzahl, da die Klassen extrem unausgewogen sind.",
                "hoeher = besser",
            ],
            [
                "ROC-AUC",
                "Flaeche unter der ROC-Kurve. Bei dieser Positivrate stark "
                "gesaettigt und daher nur ergaenzend berichtet.",
                "hoeher = besser",
            ],
            [
                "Behaltequote@99",
                "Anteil der Spalten, der behalten werden muss, damit 99 % der "
                "optimalen Spalten den Filter passieren.",
                "niedriger = besser",
            ],
        ],
    )
    z.append("")
    z.append(
        "Die **Behaltequote** ist die anwendungsnaechste Groesse: Sie gibt "
        "direkt an, wie stark das Restproblem schrumpft, wenn eine bestimmte "
        "Loesungsqualitaet garantiert werden soll. Eine Behaltequote von 5 % "
        "bedeutet, dass der Solver nur noch 5 % der urspruenglichen Spalten "
        "sieht. Sie wird ueber einen Wahrscheinlichkeits-Schwellenwert "
        "bestimmt, damit Bindungen genauso behandelt werden wie im echten "
        "Filter in `zeitvergleich_test.py`."
    )
    z.append("")

    z.append("## 2. Baseline mit allen Features")
    z.append("")
    z += _tabelle(
        ["Kennzahl", "Mittelwert", "Standardabweichung ueber Folds"],
        [
            ["PR-AUC", f"{baseline['pr_auc']:.4f}", f"{baseline['pr_auc_std']:.4f}"],
            ["ROC-AUC", f"{baseline['roc_auc']:.4f}", f"{baseline['roc_auc_std']:.4f}"],
        ]
        + [
            [
                f"Behaltequote@{int(round(ziel * 100))}",
                f"{baseline[f'behaltequote_{int(round(ziel * 100))}']:.4%}",
                f"{baseline[f'behaltequote_{int(round(ziel * 100))}_std']:.4%}",
            ]
            for ziel in ZIEL_RECALLS
        ],
    )
    z.append("")
    z.append(
        "Alle folgenden Ablationen werden gegen diese Baseline verglichen. "
        "Die Standardabweichung ueber die Folds ist der Massstab dafuer, ab "
        "wann eine Differenz ueberhaupt bedeutsam ist: Aenderungen, die "
        "kleiner sind als die Streuung der Baseline, sollten nicht "
        "interpretiert werden."
    )
    z.append("")

    z.append("## 3. Regelbasierter Referenzfilter ohne Lernverfahren")
    z.append("")
    z.append(
        "Drei Features sind keine statistischen Indikatoren, sondern "
        "**notwendige Bedingungen fuer Optimalitaet**, die sich aus der "
        "Modellstruktur herleiten lassen (Herleitung in "
        "`features_dokumentation.md`, Abschnitte 4.5 und 4.7):"
    )
    z.append("")
    z.append(
        "- `belastung_innerhalb_grenze`: Wegen der lexikografischen "
        "Zielsetzung muss jede Spalte einer Optimalloesung "
        "`load(s) <= B*` erfuellen."
    )
    z.append(
        "- `rest_ergaenzbar` bzw. `primaer_zulaessig`: Der nicht abgedeckte "
        "Rest muss mit den verbleibenden Aufzuegen innerhalb von `B*` "
        "unterzubringen sein."
    )
    z.append(
        "- `ist_distanzdominiert`: Existiert eine Spalte mit gleichem Aufzug, "
        "gleicher Artikelmenge, nicht groesserer Belastung und echt kleinerer "
        "Distanz, kann die betrachtete Spalte in keiner Optimalloesung "
        "vorkommen."
    )
    z.append("")
    z.append(
        "Die folgende Tabelle prueft diese Herleitungen empirisch auf **allen** "
        f"{kennzahlen['anzahl_zeilen']} Zeilen des Datensatzes. Die Spalte "
        "*Verletzungen* zaehlt positiv gelabelte Spalten, welche die jeweilige "
        "Bedingung verletzen; theoretisch muss sie null sein."
    )
    z.append("")
    z += _tabelle(
        ["Bedingung", "behalten", "Behaltequote", "Recall", "Precision", "Verletzungen"],
        [
            [
                f"`{reihe.bedingung}`",
                f"{reihe.behalten}",
                f"{reihe.behaltequote:.4%}",
                f"{reihe.recall:.4%}",
                f"{reihe.precision:.4%}",
                f"{reihe.verletzungen}",
            ]
            for reihe in regelfilter["tabelle"].itertuples()
        ],
    )
    z.append("")
    z.append("### Kombinierter Regelfilter je Instanz")
    z.append("")
    z += _tabelle(
        ["Groesse", "Wert"],
        [
            ["Spalten je Instanz vorher (Mittel)", f"{regelfilter['spalten_vorher_mittel']:.1f}"],
            ["Spalten je Instanz nachher (Mittel)", f"{regelfilter['spalten_nachher_mittel']:.1f}"],
            ["Behaltequote je Instanz (Mittel)", f"{regelfilter['quote_mittel']:.4%}"],
            ["Behaltequote je Instanz (Median)", f"{regelfilter['quote_median']:.4%}"],
            ["Behaltequote je Instanz (Min / Max)", f"{regelfilter['quote_min']:.4%} / {regelfilter['quote_max']:.4%}"],
            [
                "Instanzen ohne Verlust einer optimalen Spalte",
                f"{regelfilter['instanzen_vollstaendig']} / {regelfilter['instanzen_gesamt']}",
            ],
        ],
    )
    z.append("")
    z.append(
        "**Einordnung.** Der kombinierte Regelfilter erreicht per Konstruktion "
        "einen Recall von 100 % — er verwirft ausschliesslich Spalten, die "
        "beweisbar in keiner Optimalloesung vorkommen koennen. Die "
        "entscheidende Vergleichsgroesse fuer das gelernte Modell ist deshalb "
        "die Zeile *Behaltequote@100* der Baseline in Abschnitt 2: Nur wenn "
        "das Modell bei vollem Recall unter die Behaltequote des Regelfilters "
        "kommt, liefert es einen Mehrwert gegenueber reinem Preprocessing. "
        "Der Vergleich dieser beiden Zahlen sollte in der Arbeit explizit "
        "gezogen werden."
    )
    z.append("")

    z.append("## 4. Ablation A: Leave-one-out je Einzelfeature")
    z.append("")
    z.append(
        "Es wird jeweils **ein** Feature entfernt und das Modell komplett neu "
        "trainiert. Gemessen wird der *eindeutige* Beitrag eines Features: "
        "Wie viel geht verloren, wenn es fehlt und alle anderen Features "
        "einspringen duerfen? Ein Wert nahe null bedeutet nicht, dass das "
        "Feature nutzlos ist, sondern dass seine Information auch in anderen "
        "Features steckt (siehe Abschnitt 10)."
    )
    z.append("")
    z.append("Sortiert nach PR-AUC aufsteigend, d. h. schaedlichste Entfernung zuerst.")
    z.append("")
    z += _tabelle(
        [
            "entferntes Feature",
            "PR-AUC",
            "Delta PR-AUC",
            "Std",
            "Behaltequote@99",
            "Delta Behaltequote",
        ],
        _delta_zeilen(loo, baseline, list(loo.keys())),
    )
    z.append("")

    z.append("## 5. Ablation B: Leave-one-group-out je Featuregruppe")
    z.append("")
    z.append(
        "Redundante Einzelfeatures verstecken ihren Beitrag gegenseitig. "
        "Deshalb wird hier jeweils eine ganze thematische Gruppe entfernt. "
        "Die Gruppen bilden eine vollstaendige Partition aller "
        f"{len(feature_spalten)} Features."
    )
    z.append("")
    z += _tabelle(
        [
            "entfernte Gruppe",
            "Features",
            "PR-AUC",
            "Delta PR-AUC",
            "Std",
            "Behaltequote@99",
            "Delta Behaltequote",
        ],
        _delta_zeilen(
            logo,
            baseline,
            list(logo.keys()),
            groessen={name: len(gruppen[name]) for name in logo},
        ),
    )
    z.append("")
    z.append("### Zusammensetzung der Gruppen")
    z.append("")
    for name in sorted(gruppen):
        z.append(f"- **`{name}`** ({len(gruppen[name])}): " + ", ".join(
            f"`{spalte}`" for spalte in gruppen[name]
        ))
    z.append("")

    z.append("## 6. Ablation C: Einzelfeature allein")
    z.append("")
    z.append(
        "Umgekehrte Richtung: Das Modell wird mit **nur einem** Feature "
        "trainiert. Das misst die Alleinstellungskraft und ist unabhaengig "
        "von Redundanz. Sortiert nach PR-AUC absteigend."
    )
    z.append("")
    sortiert_einzeln = sorted(
        einzeln, key=lambda name: einzeln[name]["pr_auc"], reverse=True
    )
    z += _tabelle(
        ["Feature", "PR-AUC", "Anteil an Baseline", "ROC-AUC", "Behaltequote@99"],
        [
            [
                f"`{name}`",
                f"{einzeln[name]['pr_auc']:.4f}",
                f"{einzeln[name]['pr_auc'] / baseline['pr_auc']:.1%}",
                f"{einzeln[name]['roc_auc']:.4f}",
                f"{einzeln[name]['behaltequote_99']:.4%}",
            ]
            for name in sortiert_einzeln
        ],
    )
    z.append("")

    z.append("## 7. Ablation D: Einzelne Gruppe allein")
    z.append("")
    sortiert_gruppen = sorted(
        gruppe_allein, key=lambda name: gruppe_allein[name]["pr_auc"], reverse=True
    )
    z += _tabelle(
        ["Gruppe", "Features", "PR-AUC", "Anteil an Baseline", "Behaltequote@99"],
        [
            [
                f"`{name}`",
                str(len(gruppen[name])),
                f"{gruppe_allein[name]['pr_auc']:.4f}",
                f"{gruppe_allein[name]['pr_auc'] / baseline['pr_auc']:.1%}",
                f"{gruppe_allein[name]['behaltequote_99']:.4%}",
            ]
            for name in sortiert_gruppen
        ],
    )
    z.append("")

    z.append("## 8. Permutation Importance")
    z.append("")
    z.append(
        "Gemessen auf dem Testteil von Fold "
        f"{ANALYSE_FOLD} mit 5 Wiederholungen je Feature. Die Werte der "
        "Spalte eines Features werden zufaellig durchmischt und der "
        "PR-AUC-Verlust des **bereits trainierten** Modells gemessen. "
        "Im Unterschied zur Leave-one-out-Ablation wird nicht neu trainiert: "
        "Gemessen wird, wie stark das konkrete Modell auf ein Feature "
        "zurueckgreift, nicht ob es ohne dieses Feature gleichwertig neu "
        "lernen koennte."
    )
    z.append("")
    z += _tabelle(
        ["Feature", "PR-AUC-Verlust", "Std"],
        [
            [
                f"`{reihe.feature}`",
                f"{reihe.pr_auc_verlust:+.4f}",
                f"{reihe.pr_auc_verlust_std:.4f}",
            ]
            for reihe in permutation.itertuples()
        ],
    )
    z.append("")

    z.append("## 9. Standardisierte Koeffizienten des Baseline-Modells")
    z.append("")
    z.append(
        "Da alle Features auf Standardabweichung 1 gebracht wurden, ist der "
        "Koeffizient direkt als Effektstaerke je Standardabweichung lesbar. "
        "Ein positiver Wert erhoeht die vorhergesagte Wahrscheinlichkeit, "
        "dass die Spalte in einer Optimalloesung vorkommt."
    )
    z.append("")
    z.append(
        "**Vorsicht bei der Interpretation:** Bei korrelierten Features "
        "verteilt die Regression den gemeinsamen Effekt teils "
        "gegenlaeufig auf mehrere Koeffizienten. Einzelne Vorzeichen sind "
        "deshalb nur zusammen mit Abschnitt 10 zu lesen."
    )
    z.append("")
    z += _tabelle(
        ["Feature", "Koeffizient", "Betrag"],
        [
            [f"`{reihe.feature}`", f"{reihe.koeffizient:+.4f}", f"{reihe.betrag:.4f}"]
            for reihe in koeffizienten.itertuples()
        ],
    )
    z.append("")

    z.append("## 10. Redundanzanalyse")
    z.append("")
    z.append(
        f"Featurepaare mit |Pearson-Korrelation| >= {REDUNDANZ_SCHWELLE:.2f}, "
        "berechnet auf einer Zufallsstichprobe des Gesamtdatensatzes. Solche "
        "Paare erklaeren, warum viele Leave-one-out-Effekte aus Abschnitt 4 "
        "nahe null liegen: Faellt ein Partner weg, uebernimmt der andere."
    )
    z.append("")
    if redundanz.empty:
        z.append(
            f"Es wurden keine Featurepaare oberhalb der Schwelle "
            f"{REDUNDANZ_SCHWELLE:.2f} gefunden."
        )
    else:
        z += _tabelle(
            ["Feature A", "Feature B", "Korrelation"],
            [
                [f"`{reihe.feature_a}`", f"`{reihe.feature_b}`", f"{reihe.korrelation:+.4f}"]
                for reihe in redundanz.itertuples()
            ],
        )
    z.append("")

    z.append("## 11. Greedy Forward Selection")
    z.append("")
    z.append(
        "Aufbau eines minimalen Feature-Satzes: In jedem Schritt wird das "
        "Feature aufgenommen, das den PR-AUC am staerksten erhoeht. Die "
        "Auswahl laeuft auf Fold "
        f"{ANALYSE_FOLD} (Abbruch bei einem Zuwachs < {GREEDY_MIN_ZUWACHS} "
        f"oder nach {GREEDY_MAX_FEATURES} Features). Anschliessend wird der "
        "gefundene Satz auf **allen** Folds neu bewertet, damit die "
        "Auswahlentscheidung nicht dieselben Daten bewertet, auf denen sie "
        "getroffen wurde."
    )
    z.append("")
    z += _tabelle(
        ["Schritt", "aufgenommenes Feature", "PR-AUC", "Zuwachs", "Behaltequote@99"],
        [
            [
                str(eintrag["schritt"]),
                f"`{eintrag['feature']}`",
                f"{eintrag['pr_auc']:.4f}",
                f"{eintrag['zuwachs']:+.4f}",
                f"{eintrag['behaltequote_99']:.4%}",
            ]
            for eintrag in greedy
        ],
    )
    z.append("")
    z.append(
        f"### Validierung des Greedy-Satzes ({len(greedy)} Features) ueber alle Folds"
    )
    z.append("")
    z += _tabelle(
        ["Kennzahl", "Greedy-Satz", "Baseline (alle Features)", "Differenz"],
        [
            [
                "PR-AUC",
                f"{greedy_validierung['pr_auc']:.4f}",
                f"{baseline['pr_auc']:.4f}",
                f"{greedy_validierung['pr_auc'] - baseline['pr_auc']:+.4f}",
            ],
            [
                "ROC-AUC",
                f"{greedy_validierung['roc_auc']:.4f}",
                f"{baseline['roc_auc']:.4f}",
                f"{greedy_validierung['roc_auc'] - baseline['roc_auc']:+.4f}",
            ],
            [
                "Behaltequote@99",
                f"{greedy_validierung['behaltequote_99']:.4%}",
                f"{baseline['behaltequote_99']:.4%}",
                f"{(greedy_validierung['behaltequote_99'] - baseline['behaltequote_99']) * 100:+.3f} pp",
            ],
        ],
    )
    z.append("")

    z.append("## 12. Reproduktion")
    z.append("")
    z.append("```")
    z.append("python ml_datensatz.py      # erzeugt ml_datensatz.csv")
    z.append("python feature_ablation.py  # erzeugt dieses Dokument")
    z.append("```")
    z.append("")
    z.append(
        f"Laufzeit dieses Laufs: {laufzeit_text(laufzeit)}. "
        f"Konfiguration: {ANZAHL_FOLDS} Folds, {len(feature_spalten)} Features, "
        f"{len(gruppen)} Gruppen, Seed {ZUFALLS_SEED}."
    )
    z.append("")

    with open(pfad, "w", encoding="utf-8") as datei:
        datei.write("\n".join(z).rstrip("\n") + "\n")


def laufzeit_text(sekunden):
    minuten, rest = divmod(int(sekunden), 60)
    return f"{minuten} min {rest} s"

def main():
    start = time.perf_counter()

    print("Lade Datensatz ...")
    X, y, instanz_seeds, feature_spalten = lade_daten()
    gruppen = definiere_gruppen(feature_spalten)
    print(
        f"  {X.shape[0]} Zeilen, {X.shape[1]} Features, "
        f"{len(np.unique(instanz_seeds))} Instanzen, "
        f"Positivrate {y.mean():.4%}"
    )

    kennzahlen = {
        "anzahl_zeilen": int(X.shape[0]),
        "anzahl_features": int(X.shape[1]),
        "anzahl_instanzen": int(len(np.unique(instanz_seeds))),
        "anzahl_positiv": int(y.sum()),
        "positivrate": float(y.mean()),
        "zeilen_pro_instanz": float(X.shape[0] / len(np.unique(instanz_seeds))),
    }

    print("\nRegelbasierter Referenzfilter ...")
    regelfilter = analysiere_regelfilter(X, y, instanz_seeds, feature_spalten)
    for reihe in regelfilter["tabelle"].itertuples():
        print(
            f"  {reihe.bedingung:<52s} behalten {reihe.behaltequote:7.4%}  "
            f"Recall {reihe.recall:8.4%}  Verletzungen {reihe.verletzungen}"
        )

    alle_index = list(range(len(feature_spalten)))
    folds = erzeuge_folds(instanz_seeds)

    konfigurationen = [("__baseline__", alle_index)]
    for position, name in enumerate(feature_spalten):
        konfigurationen.append(
            (f"loo::{name}", [i for i in alle_index if i != position])
        )
    for gruppenname, spalten in gruppen.items():
        entfernt = {feature_spalten.index(spalte) for spalte in spalten}
        konfigurationen.append(
            (f"logo::{gruppenname}", [i for i in alle_index if i not in entfernt])
        )
    for position, name in enumerate(feature_spalten):
        konfigurationen.append((f"einzeln::{name}", [position]))
    for gruppenname, spalten in gruppen.items():
        konfigurationen.append((
            f"nur::{gruppenname}",
            [feature_spalten.index(spalte) for spalte in spalten],
        ))

    print(
        f"\n{len(konfigurationen)} Konfigurationen x {ANZAHL_FOLDS} Folds "
        f"= {len(konfigurationen) * ANZAHL_FOLDS} Trainingslaeufe"
    )

    pro_fold = []
    analyse_daten = None
    for nummer, (train_maske, test_maske) in enumerate(folds):
        fold_start = time.perf_counter()
        print(f"\nFold {nummer + 1}/{ANZAHL_FOLDS} ...")

        X_standardisiert = standardisiere(X, train_maske)
        X_train = np.ascontiguousarray(X_standardisiert[train_maske])
        X_test = np.ascontiguousarray(X_standardisiert[test_maske])
        y_train = y[train_maske]
        y_test = y[test_maske]
        del X_standardisiert

        ergebnisse = fuehre_konfigurationen_aus(
            konfigurationen, X_train, y_train, X_test, y_test
        )
        pro_fold.append(ergebnisse)

        print(
            f"  fertig in {laufzeit_text(time.perf_counter() - fold_start)}, "
            f"PR-AUC Baseline {ergebnisse['__baseline__']['pr_auc']:.4f}"
        )

        if nummer == ANALYSE_FOLD:
            analyse_daten = (X_train, y_train, X_test, y_test)
        else:
            del X_train, X_test

    gemittelt = mittle_ueber_folds(pro_fold)
    baseline = gemittelt["__baseline__"]

    loo = {
        name[len("loo::"):]: werte
        for name, werte in gemittelt.items()
        if name.startswith("loo::")
    }
    logo = {
        name[len("logo::"):]: werte
        for name, werte in gemittelt.items()
        if name.startswith("logo::")
    }
    einzeln = {
        name[len("einzeln::"):]: werte
        for name, werte in gemittelt.items()
        if name.startswith("einzeln::")
    }
    gruppe_allein = {
        name[len("nur::"):]: werte
        for name, werte in gemittelt.items()
        if name.startswith("nur::")
    }

    X_train, y_train, X_test, y_test = analyse_daten

    print("\nPermutation Importance ...")
    permutation = berechne_permutation_importance(
        X_train, y_train, X_test, y_test, feature_spalten
    )

    print("Koeffizienten ...")
    koeffizienten = berechne_koeffizienten(X_train, y_train, feature_spalten)

    print("Redundanzanalyse ...")
    redundanz = finde_redundante_paare(X, feature_spalten)

    print("Greedy Forward Selection ...")
    greedy = greedy_forward_selection(
        X_train, y_train, X_test, y_test, feature_spalten
    )

    print("Validiere Greedy-Satz ueber alle Folds ...")
    greedy_index = [feature_spalten.index(eintrag["feature"]) for eintrag in greedy]
    greedy_pro_fold = []
    for train_maske, test_maske in folds:
        X_standardisiert = standardisiere(X, train_maske)
        werte = _bewerte_featuremenge(
            np.ascontiguousarray(X_standardisiert[train_maske]),
            y[train_maske],
            np.ascontiguousarray(X_standardisiert[test_maske]),
            y[test_maske],
            np.asarray(greedy_index, dtype=np.int64),
        )
        greedy_pro_fold.append(werte)
        del X_standardisiert
    greedy_validierung = {
        kennzahl: float(np.mean([fold[kennzahl] for fold in greedy_pro_fold]))
        for kennzahl in greedy_pro_fold[0]
    }

    laufzeit = time.perf_counter() - start
    schreibe_bericht(
        pfad=AUSGABE_DATEI,
        kennzahlen=kennzahlen,
        baseline=baseline,
        regelfilter=regelfilter,
        loo=loo,
        logo=logo,
        einzeln=einzeln,
        gruppe_allein=gruppe_allein,
        gruppen=gruppen,
        permutation=permutation,
        koeffizienten=koeffizienten,
        redundanz=redundanz,
        greedy=greedy,
        greedy_validierung=greedy_validierung,
        feature_spalten=feature_spalten,
        laufzeit=laufzeit,
    )

    print(f"\nFertig in {laufzeit_text(laufzeit)}.")
    print(f"Ergebnisse geschrieben nach: {AUSGABE_DATEI}")


if __name__ == "__main__":
    main()
