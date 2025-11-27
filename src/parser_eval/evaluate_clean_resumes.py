import os
import pandas as pd
from tqdm import tqdm
import traceback
from datetime import datetime

from evaluate_parser import parse_resume

# ------------------------------------------------------
# CONFIG
# ------------------------------------------------------

CSV_PATH = "data/clean/combined_resumes.csv"  
OUTPUT_CSV = "data/clean_parser_output.csv"
ERROR_LOG = "logs/parser_errors.log"

os.makedirs("logs", exist_ok=True)

# ------------------------------------------------------
# LOGGING
# ------------------------------------------------------

def log_error(index, text, error):
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n[{datetime.now()}] Error at row {index}\n")
        f.write("Resume text:\n")
        f.write(text[:500] + "...\n\n")
        f.write("Error:\n")
        f.write(traceback.format_exc())
        f.write("\n" + "-"*80 + "\n")

# ------------------------------------------------------
# MAIN EVALUATION
# ------------------------------------------------------

def evaluate_csv_resumes():
    print(f"\nLoading resumes from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    # Expecting a 'Text' column
    assert "Text" in df.columns, "CSV must contain a 'Text' column with resume content."

    results = []

    print(f"Total resumes: {len(df)}\n")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing resumes"):
        resume_text = str(row["Text"])

        try:
            # Save temporary file for parser (parser expects a file path)
            temp_path = "temp_resume.txt"
            with open(temp_path, "w", encoding="utf-8", errors="ignore") as f:
                f.write(resume_text)

            parsed = parse_resume(temp_path)

            results.append({
                "index": idx,
                "label": row.get("Label", None),
                "name": parsed.get("name"),
                "email": parsed.get("email"),
                "phone": parsed.get("phone"),
                "skills": ", ".join(parsed.get("skills", [])),
                "sections": str(parsed.get("sections")),
                "raw_text": parsed.get("raw_text")
            })

        except Exception:
            log_error(idx, resume_text, error=True)
            results.append({
                "index": idx,
                "label": row.get("Label", None),
                "name": None,
                "email": None,
                "phone": None,
                "skills": None,
                "sections": None,
                "raw_text": None
            })

    # Save results
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print(f"\nSaved results to: {OUTPUT_CSV}")
    print(f"Error log saved to: {ERROR_LOG}\n")

    print("Summary of missing fields:")
    print(out_df[["name", "email", "phone"]].isnull().sum())

    return out_df


# ------------------------------------------------------
# RUN
# ------------------------------------------------------

if __name__ == "__main__":
    evaluate_csv_resumes()
