import pandas as pd
import re

def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = s.replace("\ufeff", "")
    s = re.sub(r"[\u2028\u2029]", " ", s)
    s = s.replace("\r", "\n")
    s = re.sub(r"\n+", "\n", s)
    return s.strip()

def load_dataset2(path="data/clean/dataset2.csv"):
    df = pd.read_csv(path, encoding="utf-8", engine="python")

    # Clean the text column
    df["Text"] = df["Text"].apply(clean_text)

    # Rename the label column
    df = df.rename(columns={"Category": "Label"})

    # Clean labels (remove numbers or garbage)
    df["Label"] = df["Label"].astype(str)
    df["Label"] = df["Label"].str.replace(r"[^A-Za-z ]+", "", regex=True)
    df["Label"] = df["Label"].str.strip()

    return df[["Text", "Label"]]

if __name__ == "__main__":
    df = load_dataset2()
    print(df.head())
    print("Loaded rows:", len(df))
