import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

FEATURE_COLS = [
    "num_tokens", "avg_token_length", "unique_token_ratio",
    "num_chars", "digit_count", "punct_count", "uppercase_ratio",
    "whitespace_ratio", "zero_width_count", "non_ascii_count",
    "homoglyph_count",
]

def evaluate_model_per_class(model, scaler, df, name):
    results = {}
    for atk in df["AttackType"].unique():
        subset = df[df["AttackType"] == atk]
        X = subset[FEATURE_COLS]

        if scaler:
            X = scaler.transform(X)

        y = subset["is_adversarial"]
        preds = model.predict(X)

        acc = accuracy_score(y, preds)
        results[atk] = float(acc)
        print(f"{name} — {atk}: {acc:.4f}")
    return results

if __name__ == "__main__":
    df = pd.read_csv("data/features/resume_features.csv")

    df["AttackType"] = df["AttackType"].fillna("clean")
    df["is_adversarial"] = (df["AttackType"] != "clean").astype(int)

    logreg = joblib.load("models/logreg_adversarial.pkl")
    rf = joblib.load("models/rf_adversarial.pkl")
    scaler = joblib.load("models/feature_scaler.pkl")

    results = {
        "logistic_regression": evaluate_model_per_class(logreg, scaler, df, "LogReg"),
        "random_forest": evaluate_model_per_class(rf, None, df, "RandomForest"),
    }

    with open("models/per_attack_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved --> models/per_attack_evaluation.json")
