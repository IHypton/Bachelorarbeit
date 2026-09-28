from statistics import mean

import pandas as pd


RANG_FEATURES_INSTANZ = (
    "belastung",
    "distanz",
    "belastung_pro_artikel",
    "distanz_pro_artikel",
    "anteil_abgedeckter_artikel",
    "min_pod_belastung",
    "max_pod_belastung",
    "min_pod_distanz",
    "max_pod_distanz",
)

RANG_FEATURES_GRUPPE = tuple(
    feature for feature in RANG_FEATURES_INSTANZ if feature != "anteil_abgedeckter_artikel"
)

RANG_FEATURES_GLEICHE_GROESSE = tuple(
    feature
    for feature in RANG_FEATURES_GRUPPE
    if feature not in {"belastung_pro_artikel", "distanz_pro_artikel"}
)


def _effektive_anzahl_aufzuege(instanz, max_aufzuege):
    anzahl = len(instanz["aufzuege"])
    if max_aufzuege is not None:
        anzahl = min(anzahl, max_aufzuege)
    if anzahl < 1:
        raise ValueError("Es muss mindestens ein Aufzug verfuegbar sein.")
    return anzahl


def _passt_in_kapazitaeten(gewichte, anzahl_aufzuege, kapazitaet):
    if not gewichte:
        return True
    if anzahl_aufzuege < 1 or max(gewichte) > kapazitaet:
        return False

    sortierte_gewichte = sorted(gewichte, reverse=True)
    fuellstaende = [0.0] * anzahl_aufzuege

    def verteile(index):
        if index == len(sortierte_gewichte):
            return True

        gewicht = sortierte_gewichte[index]
        bereits_probierte_fuellstaende = set()
        for aufzug_index, fuellstand in enumerate(fuellstaende):
            if fuellstand in bereits_probierte_fuellstaende:
                continue
            bereits_probierte_fuellstaende.add(fuellstand)

            if fuellstand + gewicht > kapazitaet:
                continue

            fuellstaende[aufzug_index] += gewicht
            if verteile(index + 1):
                return True
            fuellstaende[aufzug_index] -= gewicht

        return False

    return verteile(0)


def berechne_optimale_belastungsgrenze(instanz, max_aufzuege=None):
    min_belastungen = [
        min(instanz["belastung"][pod] for pod in pods)
        for pods in instanz["pods_pro_artikel"].values()
    ]
    anzahl_aufzuege = _effektive_anzahl_aufzuege(instanz, max_aufzuege)

    sortierte_belastungen = sorted(min_belastungen, reverse=True)
    fuellstaende = [0.0] * anzahl_aufzuege
    beste_grenze = float(sum(sortierte_belastungen))

    def verteile(index):
        nonlocal beste_grenze
        if index == len(sortierte_belastungen):
            beste_grenze = min(beste_grenze, max(fuellstaende))
            return

        belastung = sortierte_belastungen[index]
        bereits_probierte_fuellstaende = set()
        for aufzug_index, fuellstand in enumerate(fuellstaende):
            if fuellstand in bereits_probierte_fuellstaende:
                continue
            bereits_probierte_fuellstaende.add(fuellstand)

            neuer_fuellstand = fuellstand + belastung
            if neuer_fuellstand >= beste_grenze:
                continue

            fuellstaende[aufzug_index] = neuer_fuellstand
            verteile(index + 1)
            fuellstaende[aufzug_index] = fuellstand

    verteile(0)
    return beste_grenze


def _minimale_anzahl_aufzuege(gewichte, kapazitaet, maximal_verfuegbar):
    if not gewichte:
        return 0
    for anzahl in range(1, maximal_verfuegbar + 1):
        if _passt_in_kapazitaeten(gewichte, anzahl, kapazitaet):
            return anzahl
    return maximal_verfuegbar + 1


def _berechne_basisfeatures(spalte, instanz, optimale_belastungsgrenze, anzahl_aufzuege, min_belastung_pro_artikel):
    artikel_gesamt = tuple(instanz["artikel"])
    artikel_spalte = tuple(spalte["artikel"])
    artikel_spalte_set = set(artikel_spalte)
    artikel_rest = [artikel for artikel in artikel_gesamt if artikel not in artikel_spalte_set]

    anzahl_artikel_gesamt = len(artikel_gesamt)
    anzahl_artikel_spalte = len(artikel_spalte)
    pod_anzahlen_instanz = [len(pods) for pods in instanz["pods_pro_artikel"].values()]
    pod_belastungen = [instanz["belastung"][pod] for pod in spalte["pods"]]
    pod_distanzen = [instanz["distanz"][pod][spalte["aufzug"]] for pod in spalte["pods"]]

    min_belastung_abgedeckt = sum(min_belastung_pro_artikel[artikel] for artikel in artikel_spalte)
    min_belastungen_rest = [min_belastung_pro_artikel[artikel] for artikel in artikel_rest]
    min_aufzuege_rest = _minimale_anzahl_aufzuege(
        min_belastungen_rest,
        optimale_belastungsgrenze,
        anzahl_aufzuege,
    )
    rest_ergaenzbar = min_aufzuege_rest <= anzahl_aufzuege - 1

    minimale_distanz_abgedeckt = sum(
        min(instanz["distanz"][pod][spalte["aufzug"]] for pod in instanz["pods_pro_artikel"][artikel])
        for artikel in artikel_spalte
    )
    minimale_distanz_rest = sum(
        min(
            instanz["distanz"][pod][aufzug]
            for pod in instanz["pods_pro_artikel"][artikel]
            for aufzug in instanz["aufzuege"]
        )
        for artikel in artikel_rest
    )

    pod_anzahlen_abgedeckt = [len(instanz["pods_pro_artikel"][artikel]) for artikel in artikel_spalte]
    pod_anzahlen_rest = [len(instanz["pods_pro_artikel"][artikel]) for artikel in artikel_rest]

    return {
        "anzahl_artikel_gesamt": anzahl_artikel_gesamt,
        "anzahl_aufzuege_gesamt": len(instanz["aufzuege"]),
        "anzahl_pods_gesamt": sum(pod_anzahlen_instanz),
        "pods_pro_artikel_min": min(pod_anzahlen_instanz),
        "pods_pro_artikel_max": max(pod_anzahlen_instanz),
        "pods_pro_artikel_mittel": mean(pod_anzahlen_instanz),
        "anzahl_artikel_spalte": anzahl_artikel_spalte,
        "anzahl_artikel_rest": len(artikel_rest),
        "belastung": spalte["belastung"],
        "distanz": spalte["distanz"],
        "belastung_pro_artikel": spalte["belastung"] / anzahl_artikel_spalte,
        "distanz_pro_artikel": spalte["distanz"] / anzahl_artikel_spalte,
        "anteil_abgedeckter_artikel": anzahl_artikel_spalte / anzahl_artikel_gesamt,
        "min_pod_belastung": min(pod_belastungen),
        "max_pod_belastung": max(pod_belastungen),
        "spannweite_pod_belastung": max(pod_belastungen) - min(pod_belastungen),
        "min_pod_distanz": min(pod_distanzen),
        "max_pod_distanz": max(pod_distanzen),
        "spannweite_pod_distanz": max(pod_distanzen) - min(pod_distanzen),
        "optimale_belastungsgrenze": optimale_belastungsgrenze,
        "belastung_relativ_zu_grenze": spalte["belastung"] / optimale_belastungsgrenze,
        "belastung_innerhalb_grenze": int(spalte["belastung"] <= optimale_belastungsgrenze),
        "min_belastung_abgedeckte_artikel": min_belastung_abgedeckt,
        "belastungs_regret_pods": spalte["belastung"] - min_belastung_abgedeckt,
        "min_belastung_rest": sum(min_belastungen_rest),
        "min_aufzuege_fuer_rest": min_aufzuege_rest,
        "rest_ergaenzbar": int(rest_ergaenzbar),
        "primaer_zulaessig": int(
            spalte["belastung"] <= optimale_belastungsgrenze and rest_ergaenzbar
        ),
        "min_distanz_abgedeckte_artikel": minimale_distanz_abgedeckt,
        "distanz_regret_pods": spalte["distanz"] - minimale_distanz_abgedeckt,
        "min_distanz_rest": minimale_distanz_rest,
        "pods_abgedeckte_artikel_mittel": mean(pod_anzahlen_abgedeckt),
        "pods_rest_mittel": mean(pod_anzahlen_rest) if pod_anzahlen_rest else 0.0,
    }


def _fuege_gruppenvergleiche_hinzu(features, spalten):
    ergebnis = features.copy()
    ergebnis["_aufzug"] = [spalte["aufzug"] for spalte in spalten]
    ergebnis["_artikelmenge"] = [tuple(spalte["artikel"]) for spalte in spalten]

    sortiert = ergebnis.sort_values(
        ["_aufzug", "_artikelmenge", "belastung", "distanz"],
        kind="stable",
    )
    beste_distanz = sortiert.groupby(["_aufzug", "_artikelmenge"], sort=False)["distanz"].cummin()
    ergebnis["beste_distanz_ohne_mehrbelastung"] = beste_distanz
    ergebnis["distanz_regret_ohne_mehrbelastung"] = (
        ergebnis["distanz"] - ergebnis["beste_distanz_ohne_mehrbelastung"]
    )
    ergebnis["ist_distanzdominiert"] = (ergebnis["distanz_regret_ohne_mehrbelastung"] > 0).astype(int)

    for feature in RANG_FEATURES_INSTANZ:
        ergebnis[f"rang_instanz_{feature}"] = ergebnis[feature].rank(method="average", pct=True)

    gruppen = {
        "spaltengroesse": (["anzahl_artikel_spalte"], RANG_FEATURES_GLEICHE_GROESSE),
        "aufzug": (["_aufzug"], RANG_FEATURES_GRUPPE),
        "artikelmenge": (["_artikelmenge"], RANG_FEATURES_GLEICHE_GROESSE),
    }
    for gruppenname, (gruppenspalten, rang_features) in gruppen.items():
        gruppe = ergebnis.groupby(gruppenspalten, sort=False)
        for feature in rang_features:
            ergebnis[f"rang_{gruppenname}_{feature}"] = gruppe[feature].rank(method="average", pct=True)

    ergebnis = ergebnis.drop(columns=["_aufzug", "_artikelmenge"])
    return ergebnis


def berechne_spaltenfeatures(instanz, spalten, max_aufzuege=None):
    if not spalten:
        return pd.DataFrame()

    anzahl_aufzuege = _effektive_anzahl_aufzuege(instanz, max_aufzuege)
    optimale_belastungsgrenze = berechne_optimale_belastungsgrenze(instanz, max_aufzuege=max_aufzuege)
    min_belastung_pro_artikel = {
        artikel: min(instanz["belastung"][pod] for pod in pods)
        for artikel, pods in instanz["pods_pro_artikel"].items()
    }

    basisfeatures = pd.DataFrame(
        [
            _berechne_basisfeatures(
                spalte=spalte,
                instanz=instanz,
                optimale_belastungsgrenze=optimale_belastungsgrenze,
                anzahl_aufzuege=anzahl_aufzuege,
                min_belastung_pro_artikel=min_belastung_pro_artikel,
            )
            for spalte in spalten
        ]
    )
    return _fuege_gruppenvergleiche_hinzu(basisfeatures, spalten)
