import unittest

from ml_features import (
    berechne_optimale_belastungsgrenze,
    berechne_spaltenfeatures,
)
from spalten import erzeuge_spalten


class FeatureBerechnungTest(unittest.TestCase):
    def setUp(self):
        self.instanz = {
            "artikel": [1, 2, 3],
            "aufzuege": ["L1", "L2"],
            "pods_pro_artikel": {
                1: ["A", "B"],
                2: ["C"],
                3: ["D"],
            },
            "belastung": {
                "A": 3,
                "B": 4,
                "C": 5,
                "D": 2,
            },
            "distanz": {
                "A": {"L1": 8, "L2": 6},
                "B": {"L1": 10, "L2": 5},
                "C": {"L1": 12, "L2": 9},
                "D": {"L1": 7, "L2": 11},
            },
        }
        self.spalten = erzeuge_spalten(
            artikel=self.instanz["artikel"],
            aufzuege=self.instanz["aufzuege"],
            pods_pro_artikel=self.instanz["pods_pro_artikel"],
            belastung=self.instanz["belastung"],
            distanz=self.instanz["distanz"],
        )

    def test_optimale_belastungsgrenze(self):
        self.assertEqual(
            berechne_optimale_belastungsgrenze(self.instanz, max_aufzuege=2),
            5,
        )

    def test_feature_schema_ohne_exakte_duplikate(self):
        features = berechne_spaltenfeatures(self.instanz, self.spalten, max_aufzuege=2)

        self.assertEqual(len(features), len(self.spalten))
        self.assertFalse(features.isna().any().any())
        for entferntes_feature in (
            "anzahl_pods",
            "mittel_pod_belastung",
            "mittel_pod_distanz",
            "min_belastung_gleiche_artikel",
            "belastungs_regret_gleiche_artikel",
            "min_distanz_gleiche_artikel",
            "distanz_regret_gleiche_artikel",
            "rang_spaltengroesse_belastung_pro_artikel",
            "rang_spaltengroesse_distanz_pro_artikel",
            "rang_artikelmenge_belastung_pro_artikel",
            "rang_artikelmenge_distanz_pro_artikel",
        ):
            self.assertNotIn(entferntes_feature, features.columns)

    def test_kontextfeatures(self):
        features = berechne_spaltenfeatures(self.instanz, self.spalten, max_aufzuege=2)

        self.assertTrue(features["rang_instanz_belastung"].between(0, 1).all())
        self.assertTrue(features["rang_aufzug_distanz"].between(0, 1).all())
        self.assertTrue(features["primaer_zulaessig"].isin([0, 1]).all())
        self.assertGreater(int(features["ist_distanzdominiert"].sum()), 0)


if __name__ == "__main__":
    unittest.main()
