import os
import random
import pandas as pd

from attacks import insert_zero_width, homoglyph_substitute, keyword_stuffing, combo_attack

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

INPUT_PATH = "data/clean/combined_resumes.csv"
OUTPUT_PATH = "data/adversarial/adversarial_resumes.csv"

def generate_adversarial(df: pd.DataFrame,
                         max_samples: int | None = 2000) -> pd.DataFrame:
    """
    For each clean resume, generate several adversarial variants.
    Returns a new DataFrame with columns:
        - Text
        - Label
        - AttackType
        - SourceIndex
    """
    # Optionally subsample for faster experimentation
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(max_samples, random_state=RANDOM_SEED).reset_index(drop=True)

    rows = []

    for idx, row in df.iterrows():
        text = row["Text"]
        label = row["Label"]

        # Always store the original (clean) version
        rows.append({
            "Text": text,
            "Label": label,
            "AttackType": "clean",
            "SourceIndex": idx,
        })

        # Zero-width attack
        zw = insert_zero_width(text, p=0.15)
        rows.append({
            "Text": zw,
            "Label": label,
            "AttackType": "zero_width",
            "SourceIndex": idx,
        })

        # Homoglyph attack
        hg = homoglyph_substitute(text, p=0.2)
        rows.append({
            "Text": hg,
            "Label": label,
            "AttackType": "homoglyph",
            "SourceIndex": idx,
        })

        # Keyword stuffing attack
        ks = keyword_stuffing(text, label=label, repeat=5)
        rows.append({
            "Text": ks,
            "Label": label,
            "AttackType": "keyword_stuffing",
            "SourceIndex": idx,
        })

        # Combo attack
        cb = combo_attack(text, label=label)
        rows.append({
            "Text": cb,
            "Label": label,
            "AttackType": "combo",
            "SourceIndex": idx,
        })

    adv_df = pd.DataFrame(rows)
    return adv_df


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Could not find {INPUT_PATH}. Make sure combined_resumes.csv exists.")

    df = pd.read_csv(INPUT_PATH)
    if "Text" not in df.columns or "Label" not in df.columns:
        raise ValueError("combined_resumes.csv must have 'Text' and 'Label' columns.")

    print(f"Loaded {len(df)} clean resumes from {INPUT_PATH}")
    adv_df = generate_adversarial(df, max_samples=None)  # set to None to use all 15,873

    print(f"Generated {len(adv_df)} total rows (clean + adversarial).")
    adv_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved adversarial dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
