import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib

FEATURE_COLS = [
    "num_tokens", "avg_token_length", "unique_token_ratio",
    "num_chars", "digit_count", "punct_count", "uppercase_ratio",
    "whitespace_ratio", "zero_width_count", "non_ascii_count",
    "homoglyph_count",
]

def plot_feature_importances():
    rf = joblib.load("models/rf_adversarial.pkl")
    importances = rf.feature_importances_

    plt.figure(figsize=(10,6))
    plt.barh(FEATURE_COLS, importances)
    plt.title("Random Forest Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("models/plot_feature_importances.png")
    print("Saved --> models/plot_feature_importances.png")

def plot_attack_type_results():
    data = json.load(open("models/per_attack_evaluation.json"))

    labels = list(data["random_forest"].keys())
    rf_vals = list(data["random_forest"].values())
    lr_vals = list(data["logistic_regression"].values())

    x = np.arange(len(labels))
    w = 0.35

    plt.figure(figsize=(10,6))
    plt.bar(x - w/2, lr_vals, w, label="LogReg")
    plt.bar(x + w/2, rf_vals, w, label="RandomForest")

    plt.xticks(x, labels, rotation=45)
    plt.ylabel("Accuracy")
    plt.title("Accuracy Per AttackType")
    plt.legend()
    plt.tight_layout()
    plt.savefig("models/plot_per_attack_accuracy.png")
    print("Saved --> models/plot_per_attack_accuracy.png")

def plot_ablation_results():
    data = json.load(open("models/feature_ablation_results.json"))

    groups = list(data.keys())
    rf_scores = [data[g]["random_forest"]["accuracy"] for g in groups]
    lr_scores = [data[g]["logistic_regression"]["accuracy"] for g in groups]

    x = np.arange(len(groups))
    w = 0.35

    plt.figure(figsize=(12,6))
    plt.bar(x - w/2, lr_scores, w, label="LogReg")
    plt.bar(x + w/2, rf_scores, w, label="RandomForest")

    plt.xticks(x, groups, rotation=45)
    plt.ylabel("Accuracy")
    plt.title("Feature Ablation Accuracy Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("models/plot_ablation_accuracy.png")
    print("Saved --> models/plot_ablation_accuracy.png")

if __name__ == "__main__":
    plot_feature_importances()
    plot_attack_type_results()
    plot_ablation_results()
