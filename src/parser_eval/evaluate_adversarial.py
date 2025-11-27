import os
import pandas as pd
from tqdm import tqdm
import traceback
from datetime import datetime

from evaluate_parser import parse_resume

ADV_CSV = "data/adversarial/adversarial_resumes.csv"
CLEAN_BASELINE_CSV = "data/clean_parser_output.csv"

ADV_OUTPUT_CSV = "data/adversarial/adversarial_parser_output.csv"
ATTACK_SUMMARY_CSV = "data/adversarial/attack_summary.csv"
ERROR_LOG = "logs/parser_adv_errors.log"

os.makedirs("logs", exist_ok=True)
os.makedirs("data/adversarial", exist_ok=True)

def log_error(idx, text, attack_type, err):
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n[{datetime.now()}] Error at adversarial row {idx}\n")
        f.write(f"Attack type: {attack_type}\n")
        f.write("Resume text (first 300 chars):\n")
        f.write(str(text)[:300] + "...\n\n")
        f.write("Error:\n")
        f.write("".join(traceback.format_exception(type(err), err, err.__traceback__)))
        f.write("\n" + "-"*80 + "\n")

def parse_adversarial_resumes():
    print(f"\nLoading adversarial resumes from: {ADV_CSV}")
    df_adv = pd.read_csv(ADV_CSV)

    required = ["Text", "AttackType", "SourceIndex"]
    for col in required:
        assert col in df_adv.columns, f"Missing required column '{col}'"

    print(f"Total adversarial resumes: {len(df_adv)}")

    results = []

    for idx, row in tqdm(df_adv.iterrows(), total=len(df_adv), desc="Processing adversarial resumes"):
        text = str(row["Text"])
        attack_type = row["AttackType"]
        source_index = int(row["SourceIndex"])
        label = row.get("Label", None)

        try:
            temp_path = "temp_adv.txt"
            with open(temp_path, "w", encoding="utf-8", errors="ignore") as f:
                f.write(text)

            parsed = parse_resume(temp_path)

            results.append({
                "adv_row": idx,
                "source_index": source_index,
                "attack_type": attack_type,
                "label": label,
                "name_adv": parsed.get("name"),
                "email_adv": parsed.get("email"),
                "phone_adv": parsed.get("phone"),
                "skills_adv": ", ".join(parsed.get("skills", [])),
                "sections_adv": str(parsed.get("sections")),
                "raw_text_adv": parsed.get("raw_text"),
            })

        except Exception as e:
            log_error(idx, text, attack_type, e)
            results.append({
                "adv_row": idx,
                "source_index": source_index,
                "attack_type": attack_type,
                "label": label,
                "name_adv": None,
                "email_adv": None,
                "phone_adv": None,
                "skills_adv": None,
                "sections_adv": None,
                "raw_text_adv": None,
            })

    df_adv_out = pd.DataFrame(results)
    df_adv_out.to_csv(ADV_OUTPUT_CSV, index=False, encoding="utf-8")

    print(f"\nSaved parsed adversarial results to: {ADV_OUTPUT_CSV}")
    print(f"Errors logged to: {ERROR_LOG}\n")

    return df_adv_out


def summarize_attack_effects(adv_parsed):
    print("Loading clean baseline:", CLEAN_BASELINE_CSV)
    df_clean = pd.read_csv(CLEAN_BASELINE_CSV)

    assert "index" in df_clean.columns, "clean_parser_output.csv must contain an 'index' column."

    merged = adv_parsed.merge(
        df_clean,
        left_on="source_index",
        right_on="index",
        how="left",
        suffixes=("_adv", "_clean")
    )

    attack_summary = []

    for attack, group in merged.groupby("attack_type"):
        entry = {"attack_type": attack, "count": len(group)}

        for field in ["name", "email", "phone"]:
            clean_present = group[f"{field}"].notnull()
            adv_missing = group[f"{field}_adv"].isnull()

            n_clean = clean_present.sum()
            if n_clean == 0:
                failure_rate = None
            else:
                failure_rate = (adv_missing & clean_present).sum() / n_clean

            entry[f"{field}_failure_rate"] = failure_rate
            entry[f"{field}_clean_present"] = int(n_clean)

        attack_summary.append(entry)

    df_summary = pd.DataFrame(attack_summary)
    df_summary.to_csv(ATTACK_SUMMARY_CSV, index=False, encoding="utf-8")

    print(f"\nSaved attack summary to: {ATTACK_SUMMARY_CSV}\n")
    print(df_summary)

    return df_summary


def main():
    adv_parsed = parse_adversarial_resumes()
    summarize_attack_effects(adv_parsed)


if __name__ == "__main__":
    main()
