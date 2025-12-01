import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# CONFIG
RANDOM_STATE = 42
FEATURE_CSV_PATH = "data/features/resume_features.csv"
MODEL_DIR = "models"

LR_MODEL_PATH = os.path.join(MODEL_DIR, "logreg_adversarial.pkl")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_adversarial.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.pkl")
RESULTS_JSON_PATH = os.path.join(MODEL_DIR, "training_results.json")

FEATURE_COLS = [
    "num_tokens",
    "avg_token_length",
    "unique_token_ratio",
    "num_chars",
    "digit_count",
    "punct_count",
    "uppercase_ratio",
    "whitespace_ratio",
    "zero_width_count",
    "non_ascii_count",
    "homoglyph_count",
]


# Evaluation helper
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    report = classification_report(y_test, y_pred, digits=4)

    print(f"\n===== {name} =====")
    print(f"Accuracy: {acc}")
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    print(f"F1: {f1}")
    print(report)

    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, y_pred))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "report": report,
    }


def main():
    # 1. Load CSV
    print(f"Loading feature dataframe: {FEATURE_CSV_PATH}")
    df = pd.read_csv(FEATURE_CSV_PATH)
    print(f"Loaded feature dataframe: {df.shape}")

    missing_features = set(FEATURE_COLS) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing expected feature columns: {missing_features}")

    if "AttackType" not in df.columns:
        raise ValueError("CSV missing required column 'AttackType'")

    # 2. Fix missing AttackType (treat NaN as clean)
    print("\nMissing AttackType BEFORE fix:", df["AttackType"].isna().sum())

    df["AttackType"] = df["AttackType"].fillna("clean")

    # 3. Create binary label
    df["is_adversarial"] = (df["AttackType"] != "clean").astype(int)

    print("Missing AttackType AFTER fix:", df["AttackType"].isna().sum())

    print("\nLabel distribution:")
    print(df["is_adversarial"].value_counts())

    print("\nAttackType distribution:")
    print(df["AttackType"].value_counts())

    # 4. Build dataset
    X = df[FEATURE_COLS]
    y = df["is_adversarial"]

    # 5. Check for remaining NaNs in features
    nan_counts = X.isna().sum()
    if nan_counts.sum() > 0:
        print("\nWARNING — NaNs found in features:")
        print(nan_counts[nan_counts > 0])
        print("\nFilling NaNs with 0.")
        X = X.fillna(0)

    # 6. Train/test split stratified by AttackType
    X_train, X_test, y_train, y_test, atk_train, atk_test = train_test_split(
        X,
        y,
        df["AttackType"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["AttackType"],
    )

    print(f"\nTrain size: {X_train.shape}")
    print(f"Test size:  {X_test.shape}")

    # 7. Scale features for LR
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 8. Logistic Regression (class_weight balanced)
    log_reg = LogisticRegression(
        class_weight="balanced",
        max_iter=5000,
        n_jobs=-1,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )

    log_reg.fit(X_train_scaled, y_train)
    lr_results = evaluate_model(
        "Logistic Regression (Balanced, No SMOTE)",
        log_reg,
        X_test_scaled,
        y_test,
    )

    # 9. Random Forest (class-weight boosted for class 0)
    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight={0: 3.0, 1: 1.0},
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    rf.fit(X_train, y_train)
    rf_results = evaluate_model(
        "Random Forest (Class-Weighted, No SMOTE)",
        rf,
        X_test,
        y_test,
    )

    rf_results["feature_importances"] = dict(
        zip(FEATURE_COLS, rf.feature_importances_)
    )

    # 10. Save models + scaler + results
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(log_reg, LR_MODEL_PATH)
    joblib.dump(rf, RF_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(
            {"logistic_regression": lr_results, "random_forest": rf_results},
            f,
            indent=2,
        )

    print("\nModels saved:")
    print("  Logistic Regression:", LR_MODEL_PATH)
    print("  Random Forest:      ", RF_MODEL_PATH)
    print("  Scaler:             ", SCALER_PATH)
    print("  Results JSON:       ", RESULTS_JSON_PATH)

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
