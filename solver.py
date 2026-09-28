import gurobipy as gup
from gurobipy import GRB


ENV = None


def get_environment():
    global ENV

    if ENV is None:
        ENV = gup.Env(empty=True)
        ENV.setParam("OutputFlag", 0)
        ENV.setParam("LogToConsole", 0)
        ENV.start()

    return ENV


def baue_modell(
    spalten,
    artikel,
    aufzuege,
    ziel,
    toleranz=1e-6,
    feste_maximale_belastung=None,
    max_aufzuege=None,
):
    if max_aufzuege is None:
        max_aufzuege = len(aufzuege)

    modell = gup.Model(f"Elevator_Set_Covering_{ziel}", env=get_environment())

    x = {
        k: modell.addVar(vtype=GRB.BINARY,name=f"x_{spalte['id']}",)
        for k, spalte in enumerate(spalten)
    }

    b = modell.addVar(lb=0.0,name="maximale_belastung",)

    modell.Params.DualReductions = 0

    for i in artikel:
        modell.addConstr(
            gup.quicksum(x[k] for k, spalte in enumerate(spalten) if i in spalte["artikel"]) >= 1,
            name=f"Artikel_{i}_mindestens_einmal",
        )

    for aufzug in aufzuege:
        modell.addConstr(
            gup.quicksum(x[k] for k, spalte in enumerate(spalten) if spalte["aufzug"] == aufzug) <= 1,
            name=f"Aufzug_{aufzug}_hoechstens_eine_Spalte",
        )

    modell.addConstr(
        gup.quicksum(x[k] for k in range(len(spalten))) <= max_aufzuege,
        name="Maximal_Q_Aufzuege",
    )

    for k, spalte in enumerate(spalten):
        modell.addConstr(
            b >= spalte["belastung"] * x[k],
            name=f"MaxBelastung_Spalte_{spalte['id']}",
        )

    if feste_maximale_belastung is not None:
        modell.addConstr(
            b <= feste_maximale_belastung + toleranz,
            name="Fixiere_optimale_maximale_Belastung",
        )

    if ziel == "belastung":
        modell.setObjective(b, GRB.MINIMIZE)
    elif ziel == "distanz":
        modell.setObjective(
            gup.quicksum(
                spalten[k]["distanz"] * x[k] for k in range(len(spalten))),
            GRB.MINIMIZE,
        )
    else:
        raise ValueError("ziel muss 'belastung' oder 'distanz' sein")

    modell.update()

    return modell, x, b


def loese_modell(
    spalten,
    artikel,
    aufzuege,
    toleranz=1e-6,
    msg=False,
    max_aufzuege=None,
    alle_optima=False,
    max_pool_loesungen=1000,
):
    modell_last, _, b_last = baue_modell(
        spalten=spalten,
        artikel=artikel,
        aufzuege=aufzuege,
        ziel="belastung",
        toleranz=toleranz,
        max_aufzuege=max_aufzuege,
    )

    modell_last.Params.OutputFlag = 1 if msg else 0
    modell_last.Params.LogToConsole = 1 if msg else 0
    modell_last.optimize()

    status_last = modell_last.Status

    if status_last != GRB.OPTIMAL:
        return {
            "status": status_last,
            "anzahl_spalten": len(spalten),
            "gewaehlte_spalten": [],
            "maximale_belastung": None,
            "gesamtdistanz": None,
        }

    optimale_belastung = b_last.X

    modell_distanz, x, b = baue_modell(
        spalten=spalten,
        artikel=artikel,
        aufzuege=aufzuege,
        ziel="distanz",
        toleranz=toleranz,
        feste_maximale_belastung=optimale_belastung,
        max_aufzuege=max_aufzuege,
    )

    modell_distanz.Params.OutputFlag = 1 if msg else 0
    modell_distanz.Params.LogToConsole = 1 if msg else 0

    if alle_optima:
        modell_distanz.Params.PoolSearchMode = 2
        modell_distanz.Params.PoolSolutions = max_pool_loesungen
        modell_distanz.Params.PoolGapAbs = 0.5

    modell_distanz.optimize()

    status_distanz = modell_distanz.Status

    if status_distanz != GRB.OPTIMAL:
        return {
            "status": status_distanz,
            "anzahl_spalten": len(spalten),
            "gewaehlte_spalten": [],
            "maximale_belastung": None,
            "gesamtdistanz": None,
        }

    gewaehlte_spalten = [spalte for k, spalte in enumerate(spalten) if x[k].X > 0.5]

    ergebnis = {
        "status": "Optimal",
        "anzahl_spalten": len(spalten),
        "gewaehlte_spalten": gewaehlte_spalten,
        "maximale_belastung": b.X,
        "gesamtdistanz": sum(spalte["distanz"] for spalte in gewaehlte_spalten),
    }

    if alle_optima:
        optimale_spalten_ids = set()

        for loesung_nummer in range(modell_distanz.SolCount):
            modell_distanz.Params.SolutionNumber = loesung_nummer
            optimale_spalten_ids.update(
                spalte["id"] for k, spalte in enumerate(spalten) if x[k].Xn > 0.5
            )

        ergebnis["optimale_spalten_ids"] = optimale_spalten_ids
        ergebnis["anzahl_optimalloesungen"] = modell_distanz.SolCount
        ergebnis["pool_limit_erreicht"] = modell_distanz.SolCount >= max_pool_loesungen

    return ergebnis
