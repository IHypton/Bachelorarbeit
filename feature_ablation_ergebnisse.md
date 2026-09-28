# Ablationsstudie zur Feature-Relevanz

Automatisch erzeugt von `feature_ablation.py`. Alle Zahlen in diesem Dokument stammen aus einem einzigen Lauf des Skripts und sind mit festem Zufallsseed (42) reproduzierbar.

## 1. Versuchsaufbau

### 1.1 Datengrundlage

| Groesse | Wert |
|---|---|
| Instanzen | 1000 |
| Zeilen (Spalten des Set-Covering-Modells) | 1778909 |
| Features | 65 |
| Positive Labels | 3466 |
| Positivrate | 0.1948% |
| Zeilen je Instanz (Mittel) | 1778.9 |

Eine Zeile entspricht genau einer zulaessigen Spalte des Set-Covering-Modells. Das Label ist 1, wenn die Spalte in mindestens einer Optimalloesung der zugehoerigen Instanz vorkommt (ermittelt ueber den Gurobi-Loesungspool in `ml_datensatz.py`).

### 1.2 Aufteilung

Es wird eine 5-fache **instanzbasierte** Kreuzvalidierung verwendet: Die Instanzen werden in 5 disjunkte Bloecke geteilt, jeder Block dient einmal als Testmenge. Keine Instanz liegt gleichzeitig in Trainings- und Testmenge.

Diese Aufteilung ist zwingend: Spalten derselben Instanz teilen sich Kontextfeatures (z. B. `optimale_belastungsgrenze`) und alle Rangfeatures sind relativ zu den Konkurrenzspalten *derselben* Instanz definiert. Eine zeilenweise Aufteilung wuerde Information aus der Testmenge ins Training tragen und die Guete deutlich zu optimistisch schaetzen.

### 1.3 Modell

Logistische Regression (`sklearn`), `class_weight="balanced"`, `max_iter=2000`, Solver `lbfgs`, Seed 42. Alle Features werden vorab standardisiert (z-Transformation), wobei Mittelwert und Standardabweichung ausschliesslich aus der Trainingsmenge des jeweiligen Folds stammen.

**Warum standardisiert?** Die logistische Regression in `sklearn` ist standardmaessig L2-regularisiert, und diese Regularisierung ist skalenabhaengig. Ohne Standardisierung wuerde ein Feature mit grossem Wertebereich (z. B. `distanz` in [1, 120]) anders bestraft als ein Rangfeature in [0, 1]. Ein Vergleich der Feature-Beitraege waere dann nicht aussagekraeftig. Die Standardisierung aendert die Guete des Baseline-Modells praktisch nicht, macht die Koeffizienten aber untereinander vergleichbar.

### 1.4 Kennzahlen

| Kennzahl | Bedeutung | Richtung |
|---|---|---|
| PR-AUC | Flaeche unter der Precision-Recall-Kurve (Average Precision). Hauptkennzahl, da die Klassen extrem unausgewogen sind. | hoeher = besser |
| ROC-AUC | Flaeche unter der ROC-Kurve. Bei dieser Positivrate stark gesaettigt und daher nur ergaenzend berichtet. | hoeher = besser |
| Behaltequote@99 | Anteil der Spalten, der behalten werden muss, damit 99 % der optimalen Spalten den Filter passieren. | niedriger = besser |

Die **Behaltequote** ist die anwendungsnaechste Groesse: Sie gibt direkt an, wie stark das Restproblem schrumpft, wenn eine bestimmte Loesungsqualitaet garantiert werden soll. Eine Behaltequote von 5 % bedeutet, dass der Solver nur noch 5 % der urspruenglichen Spalten sieht. Sie wird ueber einen Wahrscheinlichkeits-Schwellenwert bestimmt, damit Bindungen genauso behandelt werden wie im echten Filter in `zeitvergleich_test.py`.

## 2. Baseline mit allen Features

| Kennzahl | Mittelwert | Standardabweichung ueber Folds |
|---|---|---|
| PR-AUC | 0.5231 | 0.0305 |
| ROC-AUC | 0.9982 | 0.0001 |
| Behaltequote@95 | 0.8701% | 0.0463% |
| Behaltequote@99 | 1.2673% | 0.1015% |
| Behaltequote@100 | 1.8248% | 0.2271% |

Alle folgenden Ablationen werden gegen diese Baseline verglichen. Die Standardabweichung ueber die Folds ist der Massstab dafuer, ab wann eine Differenz ueberhaupt bedeutsam ist: Aenderungen, die kleiner sind als die Streuung der Baseline, sollten nicht interpretiert werden.

## 3. Regelbasierter Referenzfilter ohne Lernverfahren

Drei Features sind keine statistischen Indikatoren, sondern **notwendige Bedingungen fuer Optimalitaet**, die sich aus der Modellstruktur herleiten lassen (Herleitung in `features_dokumentation.md`, Abschnitte 4.5 und 4.7):

- `belastung_innerhalb_grenze`: Wegen der lexikografischen Zielsetzung muss jede Spalte einer Optimalloesung `load(s) <= B*` erfuellen.
- `rest_ergaenzbar` bzw. `primaer_zulaessig`: Der nicht abgedeckte Rest muss mit den verbleibenden Aufzuegen innerhalb von `B*` unterzubringen sein.
- `ist_distanzdominiert`: Existiert eine Spalte mit gleichem Aufzug, gleicher Artikelmenge, nicht groesserer Belastung und echt kleinerer Distanz, kann die betrachtete Spalte in keiner Optimalloesung vorkommen.

Die folgende Tabelle prueft diese Herleitungen empirisch auf **allen** 1778909 Zeilen des Datensatzes. Die Spalte *Verletzungen* zaehlt positiv gelabelte Spalten, welche die jeweilige Bedingung verletzen; theoretisch muss sie null sein.

| Bedingung | behalten | Behaltequote | Recall | Precision | Verletzungen |
|---|---|---|---|---|---|
| `belastung_innerhalb_grenze == 1` | 125965 | 7.0810% | 100.0000% | 2.7516% | 0 |
| `rest_ergaenzbar == 1` | 1685852 | 94.7689% | 100.0000% | 0.2056% | 0 |
| `primaer_zulaessig == 1` | 72919 | 4.0991% | 100.0000% | 4.7532% | 0 |
| `ist_distanzdominiert == 0` | 444795 | 25.0038% | 100.0000% | 0.7792% | 0 |
| `primaer_zulaessig == 1 UND ist_distanzdominiert == 0` | 50559 | 2.8421% | 100.0000% | 6.8554% | 0 |

### Kombinierter Regelfilter je Instanz

| Groesse | Wert |
|---|---|
| Spalten je Instanz vorher (Mittel) | 1778.9 |
| Spalten je Instanz nachher (Mittel) | 50.6 |
| Behaltequote je Instanz (Mittel) | 6.3119% |
| Behaltequote je Instanz (Median) | 4.2254% |
| Behaltequote je Instanz (Min / Max) | 0.0977% / 46.6667% |
| Instanzen ohne Verlust einer optimalen Spalte | 1000 / 1000 |

**Einordnung.** Der kombinierte Regelfilter erreicht per Konstruktion einen Recall von 100 % — er verwirft ausschliesslich Spalten, die beweisbar in keiner Optimalloesung vorkommen koennen. Die entscheidende Vergleichsgroesse fuer das gelernte Modell ist deshalb die Zeile *Behaltequote@100* der Baseline in Abschnitt 2: Nur wenn das Modell bei vollem Recall unter die Behaltequote des Regelfilters kommt, liefert es einen Mehrwert gegenueber reinem Preprocessing. Der Vergleich dieser beiden Zahlen sollte in der Arbeit explizit gezogen werden.

## 4. Ablation A: Leave-one-out je Einzelfeature

Es wird jeweils **ein** Feature entfernt und das Modell komplett neu trainiert. Gemessen wird der *eindeutige* Beitrag eines Features: Wie viel geht verloren, wenn es fehlt und alle anderen Features einspringen duerfen? Ein Wert nahe null bedeutet nicht, dass das Feature nutzlos ist, sondern dass seine Information auch in anderen Features steckt (siehe Abschnitt 10).

Sortiert nach PR-AUC aufsteigend, d. h. schaedlichste Entfernung zuerst.

| entferntes Feature | PR-AUC | Delta PR-AUC | Std | Behaltequote@99 | Delta Behaltequote |
|---|---|---|---|---|---|
| `optimale_belastungsgrenze` | 0.4890 | -0.0342 | 0.0616 | 1.3435% | +0.076 pp |
| `pods_abgedeckte_artikel_mittel` | 0.5096 | -0.0136 | 0.0303 | 1.2909% | +0.024 pp |
| `rang_artikelmenge_min_pod_distanz` | 0.5178 | -0.0054 | 0.0278 | 1.2439% | -0.023 pp |
| `rang_instanz_distanz_pro_artikel` | 0.5185 | -0.0047 | 0.0307 | 1.2808% | +0.014 pp |
| `anzahl_aufzuege_gesamt` | 0.5186 | -0.0046 | 0.0280 | 1.3067% | +0.039 pp |
| `ist_distanzdominiert` | 0.5191 | -0.0041 | 0.0286 | 1.3095% | +0.042 pp |
| `rang_aufzug_min_pod_distanz` | 0.5194 | -0.0037 | 0.0285 | 1.2735% | +0.006 pp |
| `rang_aufzug_distanz_pro_artikel` | 0.5203 | -0.0028 | 0.0291 | 1.2515% | -0.016 pp |
| `min_belastung_abgedeckte_artikel` | 0.5205 | -0.0026 | 0.0304 | 1.2609% | -0.006 pp |
| `rang_artikelmenge_max_pod_belastung` | 0.5207 | -0.0024 | 0.0302 | 1.2640% | -0.003 pp |
| `rang_artikelmenge_max_pod_distanz` | 0.5211 | -0.0020 | 0.0307 | 1.2650% | -0.002 pp |
| `max_pod_belastung` | 0.5215 | -0.0017 | 0.0300 | 1.2494% | -0.018 pp |
| `belastung` | 0.5216 | -0.0016 | 0.0305 | 1.2663% | -0.001 pp |
| `anzahl_artikel_gesamt` | 0.5216 | -0.0016 | 0.0297 | 1.2610% | -0.006 pp |
| `rang_spaltengroesse_max_pod_belastung` | 0.5220 | -0.0011 | 0.0299 | 1.2630% | -0.004 pp |
| `rang_aufzug_belastung` | 0.5221 | -0.0010 | 0.0290 | 1.2605% | -0.007 pp |
| `rang_spaltengroesse_min_pod_belastung` | 0.5221 | -0.0010 | 0.0306 | 1.2692% | +0.002 pp |
| `rang_artikelmenge_belastung` | 0.5222 | -0.0010 | 0.0299 | 1.2660% | -0.001 pp |
| `belastung_relativ_zu_grenze` | 0.5222 | -0.0009 | 0.0316 | 1.2739% | +0.007 pp |
| `min_belastung_rest` | 0.5223 | -0.0008 | 0.0277 | 1.2862% | +0.019 pp |
| `distanz_regret_ohne_mehrbelastung` | 0.5224 | -0.0008 | 0.0303 | 1.2628% | -0.005 pp |
| `rang_instanz_max_pod_belastung` | 0.5224 | -0.0007 | 0.0301 | 1.2691% | +0.002 pp |
| `rang_aufzug_distanz` | 0.5226 | -0.0006 | 0.0296 | 1.2647% | -0.003 pp |
| `rang_instanz_max_pod_distanz` | 0.5226 | -0.0005 | 0.0306 | 1.2699% | +0.003 pp |
| `rang_artikelmenge_distanz` | 0.5227 | -0.0004 | 0.0299 | 1.2439% | -0.023 pp |
| `anteil_abgedeckter_artikel` | 0.5228 | -0.0003 | 0.0299 | 1.2630% | -0.004 pp |
| `pods_rest_mittel` | 0.5228 | -0.0003 | 0.0293 | 1.2686% | +0.001 pp |
| `pods_pro_artikel_max` | 0.5229 | -0.0003 | 0.0290 | 1.2608% | -0.007 pp |
| `beste_distanz_ohne_mehrbelastung` | 0.5229 | -0.0002 | 0.0294 | 1.2635% | -0.004 pp |
| `belastung_innerhalb_grenze` | 0.5229 | -0.0002 | 0.0301 | 1.2667% | -0.001 pp |
| `anzahl_artikel_spalte` | 0.5229 | -0.0002 | 0.0290 | 1.2575% | -0.010 pp |
| `min_aufzuege_fuer_rest` | 0.5230 | -0.0001 | 0.0292 | 1.2715% | +0.004 pp |
| `min_pod_distanz` | 0.5230 | -0.0001 | 0.0300 | 1.2655% | -0.002 pp |
| `belastungs_regret_pods` | 0.5231 | -0.0001 | 0.0299 | 1.2667% | -0.001 pp |
| `anzahl_pods_gesamt` | 0.5231 | -0.0000 | 0.0287 | 1.2747% | +0.007 pp |
| `pods_pro_artikel_mittel` | 0.5231 | +0.0000 | 0.0290 | 1.2628% | -0.005 pp |
| `rang_aufzug_min_pod_belastung` | 0.5231 | +0.0000 | 0.0303 | 1.2656% | -0.002 pp |
| `min_pod_belastung` | 0.5232 | +0.0000 | 0.0293 | 1.2542% | -0.013 pp |
| `rang_aufzug_max_pod_belastung` | 0.5232 | +0.0000 | 0.0302 | 1.2561% | -0.011 pp |
| `spannweite_pod_belastung` | 0.5232 | +0.0001 | 0.0305 | 1.2674% | +0.000 pp |
| `rang_instanz_min_pod_distanz` | 0.5232 | +0.0001 | 0.0301 | 1.2549% | -0.012 pp |
| `rang_instanz_anteil_abgedeckter_artikel` | 0.5232 | +0.0001 | 0.0292 | 1.2648% | -0.003 pp |
| `primaer_zulaessig` | 0.5232 | +0.0001 | 0.0299 | 1.2657% | -0.002 pp |
| `distanz_regret_pods` | 0.5233 | +0.0002 | 0.0298 | 1.2773% | +0.010 pp |
| `rang_spaltengroesse_min_pod_distanz` | 0.5233 | +0.0002 | 0.0302 | 1.2641% | -0.003 pp |
| `belastung_pro_artikel` | 0.5234 | +0.0002 | 0.0297 | 1.2637% | -0.004 pp |
| `rang_instanz_min_pod_belastung` | 0.5234 | +0.0002 | 0.0297 | 1.2592% | -0.008 pp |
| `distanz_pro_artikel` | 0.5234 | +0.0003 | 0.0308 | 1.2539% | -0.013 pp |
| `rang_spaltengroesse_distanz` | 0.5234 | +0.0003 | 0.0297 | 1.2663% | -0.001 pp |
| `rang_aufzug_max_pod_distanz` | 0.5235 | +0.0003 | 0.0295 | 1.2667% | -0.001 pp |
| `rang_instanz_belastung` | 0.5236 | +0.0005 | 0.0293 | 1.2603% | -0.007 pp |
| `rang_instanz_belastung_pro_artikel` | 0.5237 | +0.0006 | 0.0301 | 1.2662% | -0.001 pp |
| `rang_spaltengroesse_belastung` | 0.5238 | +0.0006 | 0.0298 | 1.2642% | -0.003 pp |
| `rang_spaltengroesse_max_pod_distanz` | 0.5238 | +0.0006 | 0.0301 | 1.2754% | +0.008 pp |
| `min_distanz_rest` | 0.5238 | +0.0007 | 0.0303 | 1.2750% | +0.008 pp |
| `rang_instanz_distanz` | 0.5239 | +0.0008 | 0.0295 | 1.2702% | +0.003 pp |
| `max_pod_distanz` | 0.5240 | +0.0009 | 0.0310 | 1.2685% | +0.001 pp |
| `min_distanz_abgedeckte_artikel` | 0.5240 | +0.0009 | 0.0305 | 1.2672% | -0.000 pp |
| `rang_artikelmenge_min_pod_belastung` | 0.5241 | +0.0009 | 0.0310 | 1.2635% | -0.004 pp |
| `rang_aufzug_belastung_pro_artikel` | 0.5241 | +0.0010 | 0.0302 | 1.2593% | -0.008 pp |
| `rest_ergaenzbar` | 0.5244 | +0.0012 | 0.0313 | 1.2759% | +0.009 pp |
| `distanz` | 0.5245 | +0.0014 | 0.0292 | 1.2556% | -0.012 pp |
| `anzahl_artikel_rest` | 0.5246 | +0.0015 | 0.0301 | 1.2716% | +0.004 pp |
| `spannweite_pod_distanz` | 0.5247 | +0.0016 | 0.0293 | 1.2670% | -0.000 pp |
| `pods_pro_artikel_min` | 0.5248 | +0.0017 | 0.0305 | 1.2739% | +0.007 pp |

## 5. Ablation B: Leave-one-group-out je Featuregruppe

Redundante Einzelfeatures verstecken ihren Beitrag gegenseitig. Deshalb wird hier jeweils eine ganze thematische Gruppe entfernt. Die Gruppen bilden eine vollstaendige Partition aller 65 Features.

| entfernte Gruppe | Features | PR-AUC | Delta PR-AUC | Std | Behaltequote@99 | Delta Behaltequote |
|---|---|---|---|---|---|---|
| `E_zulaessigkeit` | 6 | 0.2377 | -0.2855 | 0.0122 | 4.2296% | +2.962 pp |
| `L_rang_artikelmenge` | 6 | 0.4560 | -0.0671 | 0.0231 | 1.4083% | +0.141 pp |
| `H_podverfuegbarkeit` | 2 | 0.4888 | -0.0343 | 0.0266 | 1.2737% | +0.006 pp |
| `A_instanzkontext` | 7 | 0.4932 | -0.0300 | 0.0650 | 1.4053% | +0.138 pp |
| `K_rang_aufzug` | 8 | 0.5075 | -0.0156 | 0.0264 | 1.2455% | -0.022 pp |
| `F_regret` | 5 | 0.5125 | -0.0106 | 0.0287 | 1.3736% | +0.106 pp |
| `G_dominanz` | 3 | 0.5127 | -0.0104 | 0.0267 | 1.4572% | +0.190 pp |
| `C_rohkosten` | 4 | 0.5199 | -0.0032 | 0.0314 | 1.2659% | -0.001 pp |
| `J_rang_spaltengroesse` | 6 | 0.5207 | -0.0024 | 0.0289 | 1.2587% | -0.009 pp |
| `I_rang_instanz` | 9 | 0.5209 | -0.0022 | 0.0303 | 1.2546% | -0.013 pp |
| `D_podstatistik` | 6 | 0.5230 | -0.0001 | 0.0305 | 1.2563% | -0.011 pp |
| `B_abdeckung` | 3 | 0.5234 | +0.0002 | 0.0300 | 1.2656% | -0.002 pp |

### Zusammensetzung der Gruppen

- **`A_instanzkontext`** (7): `anzahl_artikel_gesamt`, `anzahl_aufzuege_gesamt`, `anzahl_pods_gesamt`, `pods_pro_artikel_min`, `pods_pro_artikel_max`, `pods_pro_artikel_mittel`, `optimale_belastungsgrenze`
- **`B_abdeckung`** (3): `anzahl_artikel_spalte`, `anzahl_artikel_rest`, `anteil_abgedeckter_artikel`
- **`C_rohkosten`** (4): `belastung`, `distanz`, `belastung_pro_artikel`, `distanz_pro_artikel`
- **`D_podstatistik`** (6): `min_pod_belastung`, `max_pod_belastung`, `spannweite_pod_belastung`, `min_pod_distanz`, `max_pod_distanz`, `spannweite_pod_distanz`
- **`E_zulaessigkeit`** (6): `belastung_relativ_zu_grenze`, `belastung_innerhalb_grenze`, `min_belastung_rest`, `min_aufzuege_fuer_rest`, `rest_ergaenzbar`, `primaer_zulaessig`
- **`F_regret`** (5): `min_belastung_abgedeckte_artikel`, `belastungs_regret_pods`, `min_distanz_abgedeckte_artikel`, `distanz_regret_pods`, `min_distanz_rest`
- **`G_dominanz`** (3): `beste_distanz_ohne_mehrbelastung`, `distanz_regret_ohne_mehrbelastung`, `ist_distanzdominiert`
- **`H_podverfuegbarkeit`** (2): `pods_abgedeckte_artikel_mittel`, `pods_rest_mittel`
- **`I_rang_instanz`** (9): `rang_instanz_belastung`, `rang_instanz_distanz`, `rang_instanz_belastung_pro_artikel`, `rang_instanz_distanz_pro_artikel`, `rang_instanz_anteil_abgedeckter_artikel`, `rang_instanz_min_pod_belastung`, `rang_instanz_max_pod_belastung`, `rang_instanz_min_pod_distanz`, `rang_instanz_max_pod_distanz`
- **`J_rang_spaltengroesse`** (6): `rang_spaltengroesse_belastung`, `rang_spaltengroesse_distanz`, `rang_spaltengroesse_min_pod_belastung`, `rang_spaltengroesse_max_pod_belastung`, `rang_spaltengroesse_min_pod_distanz`, `rang_spaltengroesse_max_pod_distanz`
- **`K_rang_aufzug`** (8): `rang_aufzug_belastung`, `rang_aufzug_distanz`, `rang_aufzug_belastung_pro_artikel`, `rang_aufzug_distanz_pro_artikel`, `rang_aufzug_min_pod_belastung`, `rang_aufzug_max_pod_belastung`, `rang_aufzug_min_pod_distanz`, `rang_aufzug_max_pod_distanz`
- **`L_rang_artikelmenge`** (6): `rang_artikelmenge_belastung`, `rang_artikelmenge_distanz`, `rang_artikelmenge_min_pod_belastung`, `rang_artikelmenge_max_pod_belastung`, `rang_artikelmenge_min_pod_distanz`, `rang_artikelmenge_max_pod_distanz`

## 6. Ablation C: Einzelfeature allein

Umgekehrte Richtung: Das Modell wird mit **nur einem** Feature trainiert. Das misst die Alleinstellungskraft und ist unabhaengig von Redundanz. Sortiert nach PR-AUC absteigend.

| Feature | PR-AUC | Anteil an Baseline | ROC-AUC | Behaltequote@99 |
|---|---|---|---|---|
| `distanz` | 0.0747 | 14.3% | 0.9716 | 23.7226% |
| `max_pod_distanz` | 0.0624 | 11.9% | 0.9508 | 52.3527% |
| `distanz_pro_artikel` | 0.0548 | 10.5% | 0.9184 | 74.1164% |
| `primaer_zulaessig` | 0.0478 | 9.1% | 0.9803 | 4.1370% |
| `rang_instanz_distanz` | 0.0440 | 8.4% | 0.9575 | 28.7415% |
| `rang_instanz_max_pod_distanz` | 0.0411 | 7.9% | 0.9398 | 51.1216% |
| `beste_distanz_ohne_mehrbelastung` | 0.0388 | 7.4% | 0.9373 | 46.9700% |
| `rang_instanz_distanz_pro_artikel` | 0.0369 | 7.1% | 0.9172 | 66.1525% |
| `rang_aufzug_distanz` | 0.0346 | 6.6% | 0.9483 | 36.8014% |
| `rang_aufzug_max_pod_distanz` | 0.0340 | 6.5% | 0.9282 | 66.4783% |
| `min_distanz_abgedeckte_artikel` | 0.0328 | 6.3% | 0.9343 | 45.2239% |
| `rang_aufzug_distanz_pro_artikel` | 0.0312 | 6.0% | 0.9073 | 79.2168% |
| `belastung_innerhalb_grenze` | 0.0276 | 5.3% | 0.9652 | 7.1349% |
| `belastung_relativ_zu_grenze` | 0.0208 | 4.0% | 0.9512 | 7.1349% |
| `spannweite_pod_distanz` | 0.0198 | 3.8% | 0.9035 | 79.3338% |
| `anzahl_artikel_spalte` | 0.0191 | 3.6% | 0.9237 | 47.0314% |
| `anteil_abgedeckter_artikel` | 0.0164 | 3.1% | 0.9135 | 42.6913% |
| `belastung` | 0.0163 | 3.1% | 0.9388 | 23.5723% |
| `pods_abgedeckte_artikel_mittel` | 0.0158 | 3.0% | 0.6681 | 100.0000% |
| `spannweite_pod_belastung` | 0.0151 | 2.9% | 0.8567 | 90.3642% |
| `distanz_regret_pods` | 0.0126 | 2.4% | 0.9084 | 51.9241% |
| `rang_instanz_anteil_abgedeckter_artikel` | 0.0121 | 2.3% | 0.8890 | 51.3295% |
| `rang_instanz_belastung` | 0.0121 | 2.3% | 0.9178 | 28.2530% |
| `rang_aufzug_belastung` | 0.0118 | 2.3% | 0.9157 | 29.7557% |
| `belastungs_regret_pods` | 0.0116 | 2.2% | 0.9051 | 50.7697% |
| `rang_spaltengroesse_distanz` | 0.0101 | 1.9% | 0.8630 | 61.9336% |
| `rang_spaltengroesse_min_pod_distanz` | 0.0099 | 1.9% | 0.8178 | 66.8070% |
| `rang_spaltengroesse_max_pod_distanz` | 0.0098 | 1.9% | 0.8541 | 66.3634% |
| `distanz_regret_ohne_mehrbelastung` | 0.0078 | 1.5% | 0.8751 | 25.1228% |
| `ist_distanzdominiert` | 0.0078 | 1.5% | 0.8751 | 25.1228% |
| `anzahl_pods_gesamt` | 0.0077 | 1.5% | 0.7662 | 95.9440% |
| `max_pod_belastung` | 0.0076 | 1.5% | 0.7860 | 100.0000% |
| `anzahl_artikel_rest` | 0.0075 | 1.4% | 0.8413 | 60.4861% |
| `rang_instanz_max_pod_belastung` | 0.0071 | 1.4% | 0.7990 | 76.9299% |
| `rang_aufzug_max_pod_belastung` | 0.0069 | 1.3% | 0.7934 | 83.4502% |
| `rang_artikelmenge_distanz` | 0.0069 | 1.3% | 0.8378 | 62.1497% |
| `min_belastung_abgedeckte_artikel` | 0.0066 | 1.3% | 0.8423 | 51.2511% |
| `rang_artikelmenge_max_pod_distanz` | 0.0066 | 1.3% | 0.8277 | 71.7398% |
| `rang_aufzug_min_pod_belastung` | 0.0065 | 1.2% | 0.5927 | 97.0333% |
| `min_pod_belastung` | 0.0058 | 1.1% | 0.6142 | 100.0000% |
| `rang_artikelmenge_min_pod_distanz` | 0.0056 | 1.1% | 0.7944 | 72.9583% |
| `rang_instanz_min_pod_belastung` | 0.0054 | 1.0% | 0.5913 | 96.8098% |
| `min_belastung_rest` | 0.0053 | 1.0% | 0.7912 | 72.1498% |
| `belastung_pro_artikel` | 0.0052 | 1.0% | 0.6107 | 100.0000% |
| `rang_instanz_belastung_pro_artikel` | 0.0051 | 1.0% | 0.6320 | 99.8688% |
| `anzahl_artikel_gesamt` | 0.0050 | 1.0% | 0.7024 | 100.0000% |
| `rang_aufzug_belastung_pro_artikel` | 0.0050 | 1.0% | 0.6298 | 100.0000% |
| `min_distanz_rest` | 0.0049 | 0.9% | 0.7519 | 78.1043% |
| `min_aufzuege_fuer_rest` | 0.0046 | 0.9% | 0.7568 | 89.3516% |
| `pods_pro_artikel_mittel` | 0.0045 | 0.9% | 0.6671 | 97.3778% |
| `rang_instanz_min_pod_distanz` | 0.0037 | 0.7% | 0.6862 | 90.6532% |
| `rang_spaltengroesse_max_pod_belastung` | 0.0037 | 0.7% | 0.6114 | 99.9336% |
| `rang_spaltengroesse_belastung` | 0.0035 | 0.7% | 0.6301 | 95.9157% |
| `rang_spaltengroesse_min_pod_belastung` | 0.0034 | 0.6% | 0.5833 | 97.0192% |
| `min_pod_distanz` | 0.0031 | 0.6% | 0.6504 | 94.9975% |
| `rang_artikelmenge_max_pod_belastung` | 0.0030 | 0.6% | 0.6392 | 96.4331% |
| `rang_artikelmenge_belastung` | 0.0029 | 0.6% | 0.6516 | 85.3092% |
| `pods_pro_artikel_max` | 0.0027 | 0.5% | 0.5516 | 100.0000% |
| `rang_aufzug_min_pod_distanz` | 0.0026 | 0.5% | 0.6238 | 94.8357% |
| `rang_artikelmenge_min_pod_belastung` | 0.0024 | 0.5% | 0.5842 | 88.7197% |
| `pods_pro_artikel_min` | 0.0022 | 0.4% | 0.5569 | 99.0502% |
| `anzahl_aufzuege_gesamt` | 0.0022 | 0.4% | 0.5377 | 100.0000% |
| `optimale_belastungsgrenze` | 0.0022 | 0.4% | 0.5101 | 99.8845% |
| `rest_ergaenzbar` | 0.0021 | 0.4% | 0.5263 | 94.7562% |
| `pods_rest_mittel` | 0.0020 | 0.4% | 0.5455 | 89.3516% |

## 7. Ablation D: Einzelne Gruppe allein

| Gruppe | Features | PR-AUC | Anteil an Baseline | Behaltequote@99 |
|---|---|---|---|---|
| `C_rohkosten` | 4 | 0.0823 | 15.7% | 11.0766% |
| `F_regret` | 5 | 0.0809 | 15.5% | 8.5408% |
| `G_dominanz` | 3 | 0.0808 | 15.4% | 12.5653% |
| `E_zulaessigkeit` | 6 | 0.0722 | 13.8% | 4.0230% |
| `D_podstatistik` | 6 | 0.0631 | 12.1% | 28.5958% |
| `I_rang_instanz` | 9 | 0.0628 | 12.0% | 15.1445% |
| `K_rang_aufzug` | 8 | 0.0493 | 9.4% | 19.1438% |
| `B_abdeckung` | 3 | 0.0287 | 5.5% | 43.0474% |
| `J_rang_spaltengroesse` | 6 | 0.0229 | 4.4% | 66.6398% |
| `L_rang_artikelmenge` | 6 | 0.0193 | 3.7% | 68.1690% |
| `H_podverfuegbarkeit` | 2 | 0.0148 | 2.8% | 96.1072% |
| `A_instanzkontext` | 7 | 0.0094 | 1.8% | 93.2194% |

## 8. Permutation Importance

Gemessen auf dem Testteil von Fold 0 mit 5 Wiederholungen je Feature. Die Werte der Spalte eines Features werden zufaellig durchmischt und der PR-AUC-Verlust des **bereits trainierten** Modells gemessen. Im Unterschied zur Leave-one-out-Ablation wird nicht neu trainiert: Gemessen wird, wie stark das konkrete Modell auf ein Feature zurueckgreift, nicht ob es ohne dieses Feature gleichwertig neu lernen koennte.

| Feature | PR-AUC-Verlust | Std |
|---|---|---|
| `primaer_zulaessig` | +0.4999 | 0.0003 |
| `ist_distanzdominiert` | +0.3184 | 0.0085 |
| `rang_artikelmenge_distanz` | +0.2698 | 0.0074 |
| `min_belastung_abgedeckte_artikel` | +0.2405 | 0.0123 |
| `optimale_belastungsgrenze` | +0.2177 | 0.0054 |
| `belastung_relativ_zu_grenze` | +0.1409 | 0.0081 |
| `min_belastung_rest` | +0.1299 | 0.0073 |
| `belastung` | +0.1265 | 0.0129 |
| `anzahl_pods_gesamt` | +0.0972 | 0.0118 |
| `distanz_regret_ohne_mehrbelastung` | +0.0936 | 0.0168 |
| `belastung_innerhalb_grenze` | +0.0931 | 0.0131 |
| `distanz_regret_pods` | +0.0851 | 0.0102 |
| `pods_abgedeckte_artikel_mittel` | +0.0556 | 0.0061 |
| `rang_spaltengroesse_distanz` | +0.0539 | 0.0066 |
| `rang_instanz_distanz` | +0.0460 | 0.0067 |
| `anteil_abgedeckter_artikel` | +0.0419 | 0.0072 |
| `spannweite_pod_distanz` | +0.0417 | 0.0061 |
| `distanz_pro_artikel` | +0.0414 | 0.0069 |
| `min_pod_distanz` | +0.0381 | 0.0080 |
| `belastung_pro_artikel` | +0.0367 | 0.0024 |
| `rang_artikelmenge_min_pod_distanz` | +0.0355 | 0.0030 |
| `min_aufzuege_fuer_rest` | +0.0342 | 0.0051 |
| `rang_aufzug_distanz` | +0.0340 | 0.0045 |
| `rang_instanz_distanz_pro_artikel` | +0.0305 | 0.0043 |
| `anzahl_aufzuege_gesamt` | +0.0303 | 0.0079 |
| `distanz` | +0.0265 | 0.0066 |
| `rang_instanz_max_pod_distanz` | +0.0237 | 0.0051 |
| `rang_spaltengroesse_max_pod_belastung` | +0.0213 | 0.0028 |
| `rang_aufzug_max_pod_belastung` | +0.0189 | 0.0040 |
| `rang_aufzug_min_pod_distanz` | +0.0188 | 0.0046 |
| `rang_aufzug_max_pod_distanz` | +0.0182 | 0.0049 |
| `rest_ergaenzbar` | +0.0180 | 0.0058 |
| `rang_instanz_max_pod_belastung` | +0.0169 | 0.0042 |
| `rang_instanz_belastung_pro_artikel` | +0.0145 | 0.0033 |
| `rang_aufzug_belastung_pro_artikel` | +0.0135 | 0.0027 |
| `min_pod_belastung` | +0.0102 | 0.0021 |
| `max_pod_belastung` | +0.0096 | 0.0034 |
| `rang_spaltengroesse_belastung` | +0.0095 | 0.0042 |
| `rang_artikelmenge_belastung` | +0.0083 | 0.0012 |
| `rang_aufzug_distanz_pro_artikel` | +0.0081 | 0.0038 |
| `pods_pro_artikel_mittel` | +0.0080 | 0.0023 |
| `anzahl_artikel_gesamt` | +0.0078 | 0.0040 |
| `min_distanz_rest` | +0.0064 | 0.0024 |
| `pods_pro_artikel_min` | +0.0060 | 0.0023 |
| `belastungs_regret_pods` | +0.0057 | 0.0044 |
| `rang_instanz_belastung` | +0.0056 | 0.0042 |
| `rang_instanz_anteil_abgedeckter_artikel` | +0.0042 | 0.0022 |
| `rang_aufzug_belastung` | +0.0036 | 0.0065 |
| `rang_instanz_min_pod_belastung` | +0.0029 | 0.0006 |
| `min_distanz_abgedeckte_artikel` | +0.0028 | 0.0007 |
| `rang_spaltengroesse_max_pod_distanz` | +0.0018 | 0.0017 |
| `rang_aufzug_min_pod_belastung` | +0.0017 | 0.0008 |
| `rang_instanz_min_pod_distanz` | +0.0013 | 0.0007 |
| `anzahl_artikel_rest` | +0.0011 | 0.0004 |
| `rang_spaltengroesse_min_pod_distanz` | +0.0007 | 0.0009 |
| `rang_artikelmenge_min_pod_belastung` | +0.0006 | 0.0013 |
| `pods_rest_mittel` | +0.0006 | 0.0011 |
| `rang_spaltengroesse_min_pod_belastung` | +0.0006 | 0.0031 |
| `beste_distanz_ohne_mehrbelastung` | +0.0002 | 0.0019 |
| `pods_pro_artikel_max` | +0.0001 | 0.0002 |
| `spannweite_pod_belastung` | -0.0000 | 0.0003 |
| `rang_artikelmenge_max_pod_belastung` | -0.0001 | 0.0001 |
| `max_pod_distanz` | -0.0002 | 0.0003 |
| `anzahl_artikel_spalte` | -0.0004 | 0.0003 |
| `rang_artikelmenge_max_pod_distanz` | -0.0006 | 0.0020 |

## 9. Standardisierte Koeffizienten des Baseline-Modells

Da alle Features auf Standardabweichung 1 gebracht wurden, ist der Koeffizient direkt als Effektstaerke je Standardabweichung lesbar. Ein positiver Wert erhoeht die vorhergesagte Wahrscheinlichkeit, dass die Spalte in einer Optimalloesung vorkommt.

**Vorsicht bei der Interpretation:** Bei korrelierten Features verteilt die Regression den gemeinsamen Effekt teils gegenlaeufig auf mehrere Koeffizienten. Einzelne Vorzeichen sind deshalb nur zusammen mit Abschnitt 10 zu lesen.

| Feature | Koeffizient | Betrag |
|---|---|---|
| `ist_distanzdominiert` | -4.2006 | 4.2006 |
| `primaer_zulaessig` | +2.9702 | 2.9702 |
| `min_belastung_abgedeckte_artikel` | +2.2075 | 2.2075 |
| `rang_artikelmenge_distanz` | -2.1957 | 2.1957 |
| `optimale_belastungsgrenze` | -1.8844 | 1.8844 |
| `belastung_relativ_zu_grenze` | +1.5637 | 1.5637 |
| `belastung` | +1.5142 | 1.5142 |
| `min_belastung_rest` | +1.3707 | 1.3707 |
| `distanz_regret_pods` | +1.2021 | 1.2021 |
| `distanz_regret_ohne_mehrbelastung` | +1.1890 | 1.1890 |
| `belastung_innerhalb_grenze` | +0.9168 | 0.9168 |
| `rest_ergaenzbar` | +0.8885 | 0.8885 |
| `rang_spaltengroesse_distanz` | -0.8447 | 0.8447 |
| `rang_instanz_distanz` | -0.8373 | 0.8373 |
| `anzahl_pods_gesamt` | -0.8273 | 0.8273 |
| `rang_aufzug_distanz` | -0.7440 | 0.7440 |
| `min_aufzuege_fuer_rest` | -0.7218 | 0.7218 |
| `rang_instanz_distanz_pro_artikel` | +0.7211 | 0.7211 |
| `anteil_abgedeckter_artikel` | -0.6866 | 0.6866 |
| `anzahl_aufzuege_gesamt` | -0.6838 | 0.6838 |
| `rang_artikelmenge_min_pod_distanz` | -0.6537 | 0.6537 |
| `min_pod_distanz` | +0.6394 | 0.6394 |
| `distanz` | +0.6172 | 0.6172 |
| `rang_instanz_max_pod_distanz` | -0.5854 | 0.5854 |
| `spannweite_pod_distanz` | -0.5833 | 0.5833 |
| `distanz_pro_artikel` | -0.5784 | 0.5784 |
| `rang_aufzug_min_pod_distanz` | -0.5015 | 0.5015 |
| `rang_aufzug_max_pod_distanz` | +0.4808 | 0.4808 |
| `rang_spaltengroesse_max_pod_belastung` | -0.4728 | 0.4728 |
| `pods_abgedeckte_artikel_mittel` | -0.4555 | 0.4555 |
| `rang_aufzug_belastung` | -0.4019 | 0.4019 |
| `rang_instanz_belastung` | -0.3902 | 0.3902 |
| `rang_aufzug_distanz_pro_artikel` | -0.3757 | 0.3757 |
| `rang_aufzug_max_pod_belastung` | +0.3558 | 0.3558 |
| `rang_instanz_max_pod_belastung` | +0.3468 | 0.3468 |
| `belastung_pro_artikel` | +0.3439 | 0.3439 |
| `max_pod_belastung` | -0.3261 | 0.3261 |
| `rang_spaltengroesse_belastung` | -0.3218 | 0.3218 |
| `pods_pro_artikel_mittel` | +0.3218 | 0.3218 |
| `min_pod_belastung` | -0.2840 | 0.2840 |
| `rang_artikelmenge_belastung` | -0.2732 | 0.2732 |
| `belastungs_regret_pods` | -0.2719 | 0.2719 |
| `rang_spaltengroesse_min_pod_belastung` | -0.2380 | 0.2380 |
| `rang_instanz_belastung_pro_artikel` | +0.2342 | 0.2342 |
| `rang_aufzug_belastung_pro_artikel` | +0.2240 | 0.2240 |
| `min_distanz_abgedeckte_artikel` | -0.2099 | 0.2099 |
| `pods_pro_artikel_min` | -0.2069 | 0.2069 |
| `rang_artikelmenge_max_pod_distanz` | +0.1867 | 0.1867 |
| `rang_spaltengroesse_max_pod_distanz` | +0.1693 | 0.1693 |
| `rang_instanz_anteil_abgedeckter_artikel` | +0.1689 | 0.1689 |
| `anzahl_artikel_gesamt` | -0.1670 | 0.1670 |
| `pods_rest_mittel` | +0.1612 | 0.1612 |
| `min_distanz_rest` | -0.1559 | 0.1559 |
| `beste_distanz_ohne_mehrbelastung` | -0.1533 | 0.1533 |
| `rang_instanz_min_pod_belastung` | +0.1221 | 0.1221 |
| `rang_artikelmenge_min_pod_belastung` | -0.1096 | 0.1096 |
| `rang_instanz_min_pod_distanz` | +0.1005 | 0.1005 |
| `rang_aufzug_min_pod_belastung` | +0.0820 | 0.0820 |
| `anzahl_artikel_rest` | -0.0559 | 0.0559 |
| `rang_spaltengroesse_min_pod_distanz` | +0.0498 | 0.0498 |
| `spannweite_pod_belastung` | -0.0309 | 0.0309 |
| `anzahl_artikel_spalte` | -0.0303 | 0.0303 |
| `max_pod_distanz` | -0.0248 | 0.0248 |
| `pods_pro_artikel_max` | -0.0184 | 0.0184 |
| `rang_artikelmenge_max_pod_belastung` | +0.0075 | 0.0075 |

## 10. Redundanzanalyse

Featurepaare mit |Pearson-Korrelation| >= 0.95, berechnet auf einer Zufallsstichprobe des Gesamtdatensatzes. Solche Paare erklaeren, warum viele Leave-one-out-Effekte aus Abschnitt 4 nahe null liegen: Faellt ein Partner weg, uebernimmt der andere.

| Feature A | Feature B | Korrelation |
|---|---|---|
| `rang_instanz_belastung_pro_artikel` | `rang_aufzug_belastung_pro_artikel` | +1.0000 |
| `rang_instanz_belastung` | `rang_aufzug_belastung` | +1.0000 |
| `rang_instanz_min_pod_belastung` | `rang_aufzug_min_pod_belastung` | +1.0000 |
| `rang_instanz_max_pod_belastung` | `rang_aufzug_max_pod_belastung` | +1.0000 |
| `rang_instanz_distanz_pro_artikel` | `rang_spaltengroesse_distanz` | +0.9909 |
| `rang_spaltengroesse_belastung` | `rang_aufzug_belastung_pro_artikel` | +0.9821 |
| `rang_instanz_belastung_pro_artikel` | `rang_spaltengroesse_belastung` | +0.9821 |
| `anzahl_artikel_rest` | `anteil_abgedeckter_artikel` | -0.9771 |
| `rang_spaltengroesse_min_pod_distanz` | `rang_artikelmenge_min_pod_distanz` | +0.9697 |
| `rang_spaltengroesse_max_pod_distanz` | `rang_artikelmenge_max_pod_distanz` | +0.9697 |
| `rang_spaltengroesse_distanz` | `rang_artikelmenge_distanz` | +0.9689 |
| `anteil_abgedeckter_artikel` | `rang_instanz_anteil_abgedeckter_artikel` | +0.9640 |
| `rang_instanz_distanz_pro_artikel` | `rang_artikelmenge_distanz` | +0.9581 |
| `rang_instanz_min_pod_distanz` | `rang_spaltengroesse_min_pod_distanz` | +0.9507 |

## 11. Greedy Forward Selection

Aufbau eines minimalen Feature-Satzes: In jedem Schritt wird das Feature aufgenommen, das den PR-AUC am staerksten erhoeht. Die Auswahl laeuft auf Fold 0 (Abbruch bei einem Zuwachs < 0.002 oder nach 12 Features). Anschliessend wird der gefundene Satz auf **allen** Folds neu bewertet, damit die Auswahlentscheidung nicht dieselben Daten bewertet, auf denen sie getroffen wurde.

| Schritt | aufgenommenes Feature | PR-AUC | Zuwachs | Behaltequote@99 |
|---|---|---|---|---|
| 1 | `distanz` | 0.0789 | +0.0789 | 24.6493% |
| 2 | `primaer_zulaessig` | 0.2593 | +0.1803 | 2.9431% |
| 3 | `anteil_abgedeckter_artikel` | 0.3457 | +0.0865 | 2.7800% |
| 4 | `min_belastung_abgedeckte_artikel` | 0.3864 | +0.0407 | 2.5817% |
| 5 | `optimale_belastungsgrenze` | 0.4360 | +0.0496 | 2.3547% |
| 6 | `rang_artikelmenge_distanz` | 0.4837 | +0.0478 | 1.9492% |
| 7 | `anzahl_pods_gesamt` | 0.5134 | +0.0296 | 1.9282% |
| 8 | `pods_abgedeckte_artikel_mittel` | 0.5290 | +0.0156 | 1.7893% |
| 9 | `ist_distanzdominiert` | 0.5392 | +0.0102 | 1.6210% |
| 10 | `anzahl_artikel_gesamt` | 0.5424 | +0.0032 | 1.5820% |
| 11 | `min_distanz_rest` | 0.5443 | +0.0019 | 1.5981% |

### Validierung des Greedy-Satzes (11 Features) ueber alle Folds

| Kennzahl | Greedy-Satz | Baseline (alle Features) | Differenz |
|---|---|---|---|
| PR-AUC | 0.5214 | 0.5231 | -0.0018 |
| ROC-AUC | 0.9980 | 0.9982 | -0.0003 |
| Behaltequote@99 | 1.4383% | 1.2673% | +0.171 pp |

## 12. Reproduktion

```
python ml_datensatz.py      # erzeugt ml_datensatz.csv
python feature_ablation.py  # erzeugt dieses Dokument
```

Laufzeit dieses Laufs: 59 min 39 s. Konfiguration: 5 Folds, 65 Features, 12 Gruppen, Seed 42.
