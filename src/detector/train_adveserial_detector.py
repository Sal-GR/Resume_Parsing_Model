import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#install scikit-learn instead of sklearn, but still import from sklearn
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
import joblib

# Load extracted features
df = pd.read_csv("data/features/resume_features.csv")

print("Loaded feature dataframe:", df.shape)

# Create binary label: 1 = adversarial, 0 = clean
df["is_adversarial"] = df["AttackType"].notna().astype(int)

print(df["is_adversarial"].value_counts())

# Select feature columns
# Remove columns that aren't numeric features
drop_cols = ["Label", "AttackType", "SourceIndex"]
feature_cols = [c for c in df.columns if c not in drop_cols + ["is_adversarial"]]

print("Using feature columns:", feature_cols)

X = df[feature_cols]
y = df["is_adversarial"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# Scale numeric features
# (Helps Logistic Regression; RandomForest doesn't need it)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
logreg = LogisticRegression(max_iter=2000, class_weight="balanced")
logreg.fit(X_train_scaled, y_train)

logreg_preds = logreg.predict(X_test_scaled)

print("\n===== Logistic Regression Results =====")
print("Accuracy:", accuracy_score(y_test, logreg_preds))
print("Precision:", precision_score(y_test, logreg_preds))
print("Recall:", recall_score(y_test, logreg_preds))
print("F1:", f1_score(y_test, logreg_preds))
print("\nClassification Report:\n", classification_report(y_test, logreg_preds))

# Train Random Forest (often performs better)
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

rf_preds = rf.predict(X_test)

print("\n===== Random Forest Results =====")
print("Accuracy:", accuracy_score(y_test, rf_preds))
print("Precision:", precision_score(y_test, rf_preds))
print("Recall:", recall_score(y_test, rf_preds))
print("F1:", f1_score(y_test, rf_preds))
print("\nClassification Report:\n", classification_report(y_test, rf_preds))

# Show feature importances (RandomForest)
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\n===== Feature Importances =====")
print(importance_df.head(20))

importance_df.to_csv("models/feature_importances.csv", index=False)

# Save models + scaler
joblib.dump(logreg, "models/logreg_adversarial_detector.pkl")
joblib.dump(rf, "models/rf_adversarial_detector.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nModels saved successfully!")
