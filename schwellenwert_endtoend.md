# End-to-End-Vergleich der Spaltenvorauswahl

Automatisch erzeugt von `schwellenwert_endtoend.py`.

Testmenge: 100 Instanzen mit den Seeds `20360630` bis `20360729`. Diese Seeds liegen ausserhalb des Trainingsbereichs von `ml_datensatz.py`.

## Fragestellung

`ml_schwellenwert.py` bewertet die Vorhersage **je Spalte** (Precision/Recall). Fuer die Anwendung zaehlt aber etwas anderes: Findet der Solver auf der reduzierten Spaltenmenge noch die *echte* Optimalloesung? Dafuer muessen nicht alle optimalen Spalten ueberleben, sondern mindestens **eine vollstaendige** Optimalloesung. Ein hoher Spalten-Recall garantiert das nicht.

Die Spalte *exakt optimal* zaehlt Instanzen, bei denen die gefilterte Loesung in maximaler Belastung **und** Gesamtdistanz mit der ungefilterten Referenzloesung uebereinstimmt.

Die Zeitmessung der gefilterten Verfahren enthaelt die vollstaendige Pipeline: Featureberechnung, Modellauswertung und MIP-Loesung.

## Ergebnisse

| Verfahren | Spalten (Mittel) | Gesamtzeit (s) | Zeit/Instanz (s) | Speedup | exakt optimal |
|---|---|---|---|---|---|
| ohne Filter | 1642.9 | 30.423 | 0.3042 | 1.00x | 100/100 (100%) |
| Regelfilter | 55.1 | 8.482 | 0.0848 | 3.59x | 100/100 (100%) |
| LR @ 0.30 | 34.0 | 8.194 | 0.0819 | 3.71x | 99/100 (99%) |
| LR @ 0.50 | 29.6 | 8.142 | 0.0814 | 3.74x | 98/100 (98%) |
| LR @ 0.70 | 25.9 | 8.074 | 0.0807 | 3.77x | 98/100 (98%) |
| LR @ 0.80 | 22.9 | 8.014 | 0.0801 | 3.80x | 96/100 (96%) |
| LR @ 0.85 | 21.1 | 7.981 | 0.0798 | 3.81x | 96/100 (96%) |
| LR @ 0.90 | 18.7 | 7.903 | 0.0790 | 3.85x | 91/100 (91%) |
| LR @ 0.95 | 15.2 | 7.810 | 0.0781 | 3.90x | 83/100 (83%) |

## Einordnung

Der **Regelfilter** verwendet ausschliesslich die beiden exakt hergeleiteten notwendigen Bedingungen `primaer_zulaessig == 1` und `ist_distanzdominiert == 0` (Herleitung in `features_dokumentation.md`, Abschnitte 4.5 und 4.7). Er kann per Konstruktion keine Optimalloesung verlieren und dient deshalb als verlustfreie Referenz.

Beim gelernten Modell steuert der Schwellenwert den Kompromiss zwischen Geschwindigkeit und Loesungsqualitaet. Die Tabelle macht sichtbar, wie stark die Instanz-Optimalitaet einbricht, lange bevor der Spalten-Recall aus `schwellenwert_ergebnisse.md` das nahelegt.
