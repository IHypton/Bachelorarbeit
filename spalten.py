from itertools import combinations, product


def erzeuge_spalten(artikel, aufzuege, pods_pro_artikel, belastung, distanz):
    spalten = []
    for aufzug in aufzuege:
        for groesse in range(1, len(artikel) + 1):
            for artikel_auswahl in combinations(artikel, groesse):
                pod_listen = [pods_pro_artikel[i] for i in artikel_auswahl]

                for pod_auswahl in product(*pod_listen):
                    gesamtbelastung = sum(belastung[pod] for pod in pod_auswahl)
                    gesamtdistanz = sum(
                        distanz[pod][aufzug]
                        for pod in pod_auswahl
                    )

                    spalten.append({
                        "aufzug": aufzug,
                        "pods": pod_auswahl,
                        "artikel": artikel_auswahl,
                        "belastung": gesamtbelastung,
                        "distanz": gesamtdistanz,
                    })

    for spalten_id, spalte in enumerate(spalten):
        spalte["id"] = spalten_id

    return spalten
