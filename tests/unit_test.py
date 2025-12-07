import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


class UnitTest(unittest.TestCase):

    # Test 1: Ensure dataset loads correctly.
    def test_load_data(self):

        df = pd.read_csv("apachejit.csv")

        self.assertGreater(len(df), 0, "Dataset must not be empty")
        self.assertIn("buggy", df.columns, "Target column 'buggy' missing")

    #Test 2: Verify correct train/test splitting.
    def test_train_test_split(self):
        df = pd.read_csv("apachejit.csv")
        X = df.drop(columns=["buggy"]).values
        Y = df["buggy"].values

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42, stratify=Y
        )

        self.assertEqual(len(X_train) + len(X_test), len(X))
        self.assertAlmostEqual(len(X_test) / len(X), 0.2, delta=0.02)

    #Test 3: Check RandomForest trains without errors.
    def test_randomforest_training(self):
        df = pd.read_csv("apachejit.csv")
        X = df.drop(columns=["buggy"]).values
        Y = df["buggy"].values

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42, stratify=Y
        )

        model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
        model.fit(X_train, Y_train)

        preds = model.predict(X_test)

        self.assertEqual(len(preds), len(X_test))


if __name__ == "__main__":
    unittest.main()