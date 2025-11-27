import os
import re
import pandas as pd
import numpy as np
import unicodedata
from tqdm import tqdm

# Paths (you can adjust these)
COMBINED_DATA = "data/clean/combined_resumes.csv"
ADVERSARIAL_DATA = "data/adversarial/adversarial_resumes.csv"
OUTPUT_FEATURES = "data/features/resume_features.csv"

os.makedirs("data/features", exist_ok=True)

# ------------------------------
# Unicode helpers
# ------------------------------

ZERO_WIDTH_CHARS = {
    "\u200b", "\u200c", "\u200d", "\ufeff",
}

def count_zero_width(text):
    return sum(text.count(c) for c in ZERO_WIDTH_CHARS)

def count_non_ascii(text):
    return sum(1 for ch in text if ord(ch) > 127)

def count_homoglyphs(text):
    """Counts characters that are Unicode letters but not ASCII A-Z or a-z."""
    count = 0
    for ch in text:
        if ch.isalpha() and not ("A" <= ch <= "Z" or "a" <= ch <= "z"):
            count += 1
    return count


# ------------------------------
# Token-level features
# ------------------------------

def extract_token_features(text):
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    num_tokens = len(tokens)
    avg_token_len = np.mean([len(t) for t in tokens]) if tokens else 0

    return {
        "num_tokens": num_tokens,
        "avg_token_length": avg_token_len,
        "unique_token_ratio": len(set(tokens)) / num_tokens if num_tokens > 0 else 0,
    }


# ------------------------------
# Character-level features
# ------------------------------

def extract_character_features(text):
    return {
        "num_chars": len(text),
        "digit_count": sum(c.isdigit() for c in text),
        "punct_count": sum(1 for c in text if re.match(r"[^\w\s]", c)),
        "uppercase_ratio": sum(c.isupper() for c in text) / len(text) if len(text) > 0 else 0,
        "whitespace_ratio": sum(c.isspace() for c in text) / len(text) if len(text) > 0 else 0,
    }


# ------------------------------
# Unicode-level features
# ------------------------------

def extract_unicode_features(text):
    return {
        "zero_width_count": count_zero_width(text),
        "non_ascii_count": count_non_ascii(text),
        "homoglyph_count": count_homoglyphs(text),
    }


# ------------------------------
# Main feature extraction
# ------------------------------

def extract_all_features(df, text_col="Text"):
    feature_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        text = str(row[text_col])

        token_feats = extract_token_features(text)
        char_feats = extract_character_features(text)
        unicode_feats = extract_unicode_features(text)

        feature_row = {**token_feats, **char_feats, **unicode_feats}

        # Add merge fields (attack type + label + source index)
        for col in ["Label", "AttackType", "SourceIndex"]:
            if col in row:
                feature_row[col] = row[col]

        feature_rows.append(feature_row)

    return pd.DataFrame(feature_rows)


def main():
    print("Loading datasets...")

    df_clean = pd.read_csv(COMBINED_DATA)
    df_adv = pd.read_csv(ADVERSARIAL_DATA)

    df_all = pd.concat([df_clean, df_adv], ignore_index=True)

    print(f"Total resumes (clean + adversarial): {len(df_all)}")

    features = extract_all_features(df_all)

    print("Saving feature matrix to:", OUTPUT_FEATURES)
    features.to_csv(OUTPUT_FEATURES, index=False)

    print("\nFeature extraction complete!")
    print("Feature shape:", features.shape)
    print("Preview:")
    print(features.head())


if __name__ == "__main__":
    main()
