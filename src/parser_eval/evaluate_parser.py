import os
import pandas as pd
from pyresparser import ResumeParser
from tqdm import tqdm

INPUT_PATH = "data/adversarial_resumes.csv"
OUTPUT_PATH = "data/parser_eval_results.csv"

# Fields to compare between clean vs attacked resumes
FIELDS = [
    "name",
    "email",
    "mobile_number",
    "skills",
    "total_experience",
]


def safe_parse(text: str) -> dict:
    """
    Save the resume text to a temporary file and parse using pyresparser.
    Pyresparser only accepts file paths, not raw text.
    """
    try:
        with open("temp_resume.txt", "w", encoding="utf-8") as f:
            f.write(text)

        parsed = ResumeParser("temp_resume.txt").get_extracted_data()
        return parsed if parsed else {}
    except Exception:
        return {}


def compare_fields(clean_data: dict, adv_data: dict) -> dict:
    """
    Measure whether each field was extracted correctly, changed, or lost.
    """
    result = {}
    for field in FIELDS:
        clean_value = clean_data.get(field)
        adv_value = adv_data.get(field)

        # Normalize for comparison
        if isinstance(clean_value, list):
            clean_value = set(clean_value)
        if isinstance(adv_value, list):
            adv_value = set(adv_value)

        # Boolean: did the attack break this field?
        broken = clean_value != adv_value

        result[field + "_broken"] = int(broken)
    return result


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Missing {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows from {INPUT_PATH}")

    # Only evaluate rows that include a clean version
    clean_df = df[df["AttackType"] == "clean"].reset_index(drop=True)

    results = []

    print("\nRunning parser evaluation on clean + adversarial resumes...\n")

    for idx, clean_row in tqdm(clean_df.iterrows(), total=len(clean_df)):

        clean_text = clean_row["Text"]
        label = clean_row["Label"]
        source_index = clean_row["SourceIndex"]

        # Parse clean
        clean_parsed = safe_parse(clean_text)

        # Get all adversarial variants for this resume
        adv_rows = df[df["SourceIndex"] == source_index]

        for _, adv_row in adv_rows.iterrows():

            adv_text = adv_row["Text"]
            attack_type = adv_row["AttackType"]

            # Parse adversarial version
            adv_parsed = safe_parse(adv_text)

            # Compare clean vs adversarial extraction
            broken_fields = compare_fields(clean_parsed, adv_parsed)

            row = {
                "SourceIndex": source_index,
                "Label": label,
                "AttackType": attack_type,
            }
            row.update(broken_fields)

            results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved parser evaluation results to: {OUTPUT_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()
