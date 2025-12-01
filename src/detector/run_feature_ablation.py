import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

RANDOM_STATE = 42

FEATURE_GROUPS = {
    "all_features": [
        "num_tokens", "avg_token_length", "unique_token_ratio",
        "num_chars", "digit_count", "punct_count", "uppercase_ratio",
        "whitespace_ratio", "zero_width_count", "non_ascii_count",
        "homoglyph_count",
    ],
    "no_unicode": [
        "num_tokens", "avg_token_length", "unique_token_ratio",
        "num_chars", "digit_count", "punct_count", "uppercase_ratio",
        "whitespace_ratio",
    ],
    "no_counts": [
        "avg_token_length", "unique_token_ratio",
        "uppercase_ratio", "whitespace_ratio",
        "zero_width_count", "non_ascii_count", "homoglyph_count",
    ],
    "only_unicode": [
        "zero_width_count", "non_ascii_count", "homoglyph_count",
    ],
    "only_structure": [
        "num_tokens", "avg_token_length", "unique_token_ratio",
        "num_chars", "digit_count", "punct_count",
        "uppercase_ratio", "whitespace_ratio",
    ],
}

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

def run_experiment(feature_cols, df):
    X = df[feature_cols]
    y = df["is_adversarial"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=df["AttackType"], random_state=RANDOM_STATE
    )

    # LR requires scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(class_weight="balanced", max_iter=5000)
    lr.fit(X_train_scaled, y_train)

    rf = RandomForestClassifier(
        class_weight={0: 3.0, 1: 1.0},
        n_estimators=300,
        random_state=RANDOM_STATE
    )
    rf.fit(X_train, y_train)

    return {
        "logistic_regression": evaluate(lr, X_test_scaled, y_test),
        "random_forest": evaluate(rf, X_test, y_test)
    }

if __name__ == "__main__":
    df = pd.read_csv("data/features/resume_features.csv")

    # Fix missing AttackType
    df["AttackType"] = df["AttackType"].fillna("clean")
    df["is_adversarial"] = (df["AttackType"] != "clean").astype(int)

    results = {}

    for name, cols in FEATURE_GROUPS.items():
        print(f"\n==== Running ablation: {name} ====")
        results[name] = run_experiment(cols, df)
        print(json.dumps(results[name], indent=2))

    with open("models/feature_ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved feature ablation results → models/feature_ablation_results.json")
