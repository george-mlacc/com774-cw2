import unittest
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class IntegrationTest(unittest.TestCase):

    def test_pipeline(self):

        #1 Load dataset
        df = pd.read_csv("apachejit.csv")
        X = df.drop(columns=["buggy"]).values
        Y = df["buggy"].values

        #2 Split dataset
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42, stratify=Y
        )

        #3 Train model
        model = RandomForestClassifier(n_estimators=200, max_depth=12,random_state=42,n_jobs=-1)
        model.fit(X_train, Y_train)

        preds_before = model.predict(X_test)

        #4 Save model
        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)

        #5 Load model back
        with open("model.pkl", "rb") as f:
            loaded_model = pickle.load(f)
        
        #6 Check predictions after loading
        preds_after = loaded_model.predict(X_test)
        np.testing.assert_array_equal(
            preds_before,
            preds_after,
            "Saved and loaded model predictions do not match"
        )

        #7 Accuracy check
        acc = accuracy_score(Y_test, preds_after)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)
        self.assertGreaterEqual(
            acc, 0.6,
            f"Accuracy too low (acc={acc})"
        )


if __name__ == "__main__":
    unittest.main()
