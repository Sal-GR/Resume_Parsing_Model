import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


def load_features(path="data/features/resume_features.csv"):
    print(f"Loading feature dataframe: {path}")
    df = pd.read_csv(path)
    print("Loaded feature dataframe:", df.shape)
    return df


def build_hybrid_sampler():
    """
    Hybrid approach:
      1. Undersample majority class (adversarial)
      2. SMOTE oversample minority class (clean)
    """
    return Pipeline(steps=[
        ("undersample", RandomUnderSampler(sampling_strategy=0.7)),  # reduce 1:5 imbalance → 1:0.7
        ("oversample", SMOTE(sampling_strategy=1.0)),                # finish by balancing 1:1
    ])


def train_and_evaluate(X_train, X_test, y_train, y_test, scaler, hybrid_sampler):
    # Apply sampling
    print("\nApplying hybrid resampling...")
    X_resampled, y_resampled = hybrid_sampler.fit_resample(X_train, y_train)
    print("Resampled dataset:", X_resampled.shape, " | Positive:", sum(y_resampled), " Negative:", len(y_resampled) - sum(y_resampled))

    # Scale AFTER resampling
    X_resampled = scaler.fit_transform(X_resampled)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # Logistic Regression (balanced)
    print("\n===== Logistic Regression (Balanced) =====")

    logreg = LogisticRegression(
        max_iter=500,
        class_weight="balanced",          # give equal importance to minority class
        n_jobs=-1,
    )
    logreg.fit(X_resampled, y_resampled)

    y_pred_lr = logreg.predict(X_test_scaled)

    results["logistic_regression"] = {
        "accuracy": accuracy_score(y_test, y_pred_lr),
        "precision": precision_score(y_test, y_pred_lr),
        "recall": recall_score(y_test, y_pred_lr),
        "f1": f1_score(y_test, y_pred_lr),
        "report": classification_report(y_test, y_pred_lr)
    }

    print(f"Accuracy: {results['logistic_regression']['accuracy']}")
    print(f"Precision: {results['logistic_regression']['precision']}")
    print(f"Recall: {results['logistic_regression']['recall']}")
    print(f"F1: {results['logistic_regression']['f1']}")
    print(results['logistic_regression']['report'])


    # Random Forest (class-balanced)
    print("\n===== Random Forest (Balanced) =====")

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        n_jobs=-1
    )
    rf.fit(X_resampled, y_resampled)

    y_pred_rf = rf.predict(X_test_scaled)

    results["random_forest"] = {
        "accuracy": accuracy_score(y_test, y_pred_rf),
        "precision": precision_score(y_test, y_pred_rf),
        "recall": recall_score(y_test, y_pred_rf),
        "f1": f1_score(y_test, y_pred_rf),
        "report": classification_report(y_test, y_pred_rf),
        "feature_importances": rf.feature_importances_
    }

    print(f"Accuracy: {results['random_forest']['accuracy']}")
    print(f"Precision: {results['random_forest']['precision']}")
    print(f"Recall: {results['random_forest']['recall']}")
    print(f"F1: {results['random_forest']['f1']}")
    print(results['random_forest']['report'])

    return logreg, rf, scaler, results


def main():
    df = load_features()

    # Feature columns from earlier extraction
    feature_cols = [
        'num_tokens', 'avg_token_length', 'unique_token_ratio',
        'num_chars', 'digit_count', 'punct_count',
        'uppercase_ratio', 'whitespace_ratio',
        'zero_width_count', 'non_ascii_count', 'homoglyph_count'
    ]

    print("Using feature columns:", feature_cols)

    X = df[feature_cols]
    y = df["is_adversarial"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Train size: {X_train.shape}")
    print(f"Test size: {X_test.shape}")

    # Hybrid sampler
    hybrid_sampler = build_hybrid_sampler()

    # Scaler
    scaler = StandardScaler()

    # Train models
    logreg, rf, scaler, results = train_and_evaluate(
        X_train, X_test, y_train, y_test, scaler, hybrid_sampler
    )

    # Save models
    os.makedirs("models", exist_ok=True)
    joblib.dump(logreg, "models/logreg_balanced_detector.pkl")
    joblib.dump(rf, "models/rf_balanced_detector.pkl")
    joblib.dump(scaler, "models/scaler_balanced.pkl")

    # Save importances
    pd.DataFrame({
        "feature": feature_cols,
        "importance": results["random_forest"]["feature_importances"]
    }).to_csv("models/feature_importances_balanced.csv", index=False)

    print("\nModels + scaler saved successfully!")
    print("Results:")
    print(results)


if __name__ == "__main__":
    main()
