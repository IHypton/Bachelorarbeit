import random
from statistics import mean


def erzeuge_instanz(
    anzahl_artikel,
    pods_je_artikel,
    anzahl_aufzuege,
    seed,
    min_belastung=1,
    max_belastung=10,
    min_distanz=1,
    max_distanz=20,
):
    if anzahl_artikel < 1:
        raise ValueError("anzahl_artikel muss mindestens 1 sein.")

    if anzahl_aufzuege < 1:
        raise ValueError("anzahl_aufzuege muss mindestens 1 sein.")

    if min_belastung > max_belastung:
        raise ValueError(
            "min_belastung darf nicht größer als "
            "max_belastung sein."
        )

    if min_distanz > max_distanz:
        raise ValueError(
            "min_distanz darf nicht größer als "
            "max_distanz sein."
        )

    min_pods, max_pods = pods_je_artikel

    rng = random.Random(seed)

    artikel = list(range(1, anzahl_artikel + 1))

    aufzuege = [f"A{nummer}" for nummer in range(1, anzahl_aufzuege + 1)]

    pods_pro_artikel = {}
    belastung = {}
    distanz = {}

    pod_zaehler = 1

    for artikel_id in artikel:
        artikel_pods = []

        anzahl_pods_dieses_artikels = rng.randint(min_pods, max_pods)

        for _ in range(anzahl_pods_dieses_artikels):
            pod_id = f"r{pod_zaehler}"
            pod_zaehler += 1

            artikel_pods.append(pod_id)

            belastung[pod_id] = rng.randint(min_belastung, max_belastung)

            distanz[pod_id] = {
                aufzug: rng.randint(min_distanz, max_distanz)
                for aufzug in aufzuege
            }

        pods_pro_artikel[artikel_id] = artikel_pods

    pod_anzahlen = [len(pods) for pods in pods_pro_artikel.values()]

    return {
        "artikel": artikel,
        "aufzuege": aufzuege,
        "pods_pro_artikel": pods_pro_artikel,
        "belastung": belastung,
        "distanz": distanz,
        "seed": seed,
        "anzahl_artikel": len(artikel),
        "anzahl_aufzuege": len(aufzuege),
        "anzahl_pods_gesamt": sum(pod_anzahlen),
        "pods_pro_artikel_min": min(pod_anzahlen),
        "pods_pro_artikel_max": max(pod_anzahlen),
        "pods_pro_artikel_mittel": mean(pod_anzahlen),
    }


def erzeuge_variable_instanz(
    seed,
    artikel_bereich=(4, 6),
    standorte_bereich=(2, 4),
    pods_pro_artikel_bereich=(1, 2),
    belastung_bereich=(1, 10),
    distanz_bereich=(1, 20),
):
    parameter_rng = random.Random(seed)

    anzahl_artikel = parameter_rng.randint(artikel_bereich[0], artikel_bereich[1])
    anzahl_aufzuege = parameter_rng.randint(standorte_bereich[0], standorte_bereich[1])

    daten_seed = parameter_rng.randrange(0, 2**32)

    instanz = erzeuge_instanz(
        anzahl_artikel=anzahl_artikel,
        pods_je_artikel=pods_pro_artikel_bereich,
        anzahl_aufzuege=anzahl_aufzuege,
        seed=daten_seed,
        min_belastung=belastung_bereich[0],
        max_belastung=belastung_bereich[1],
        min_distanz=distanz_bereich[0],
        max_distanz=distanz_bereich[1],
    )

    instanz["variations_seed"] = seed
    instanz["daten_seed"] = daten_seed

    return instanz


def drucke_instanz(instanz):
    print("Instanzparameter:")
    print(f"  Artikel: {len(instanz['artikel'])}")
    print(f"  Aufzugsstandorte: {len(instanz['aufzuege'])}")
    print(
        f"  Pods insgesamt: "
        f"{sum(len(pods) for pods in instanz['pods_pro_artikel'].values())}"
    )
    print()

    print("Artikel:")
    print(instanz["artikel"])
    print()

    print("Aufzüge:")
    print(instanz["aufzuege"])
    print()

    print("Pods pro Artikel:")
    for artikel_id, pods in instanz["pods_pro_artikel"].items():
        print(f"  Artikel {artikel_id}: {len(pods)} Pods {pods}")
    print()

    print("Belastungen:")
    for pod, wert in instanz["belastung"].items():
        print(f"  {pod}: {wert}")
    print()

    print("Distanzen:")
    for pod, werte in instanz["distanz"].items():
        print(f"  {pod}: {werte}")


if __name__ == "__main__":
    for test_seed in range(1, 6):
        print("=" * 60)
        print(f"Variable Testinstanz, Seed {test_seed}")
        print("=" * 60)

        testinstanz = erzeuge_variable_instanz(
            seed=test_seed,
            artikel_bereich=(4, 6),
            standorte_bereich=(2, 4),
            pods_pro_artikel_bereich=(1, 2),
        )

        drucke_instanz(testinstanz)
        print()
