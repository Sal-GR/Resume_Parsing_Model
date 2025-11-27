import pandas as pd
import re
from bs4 import BeautifulSoup

def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = s.replace("\ufeff", "")
    s = re.sub(r"[\u2028\u2029]", " ", s)
    s = s.replace("\r", "\n")
    s = re.sub(r"\n+", "\n", s)
    return s.strip()

def html_to_text(html):
    if not isinstance(html, str):
        return ""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n").strip()

def load_dataset1(path="data/clean/dataset1.csv"):
    df = pd.read_csv(
        path,
        encoding="utf-8",
        engine="python",
        quoting=1,
        on_bad_lines="skip"
    )

    # Rename Category → Label
    df = df.rename(columns={"Category": "Label"})

    # Clean Resume_str
    df["Resume_str"] = df["Resume_str"].apply(clean_text)

    # Clean HTML if needed
    df["Resume_html"] = df["Resume_html"].apply(clean_text)
    df["Text_html"] = df["Resume_html"].apply(html_to_text)

    # Use plain text as the main content
    df["Text"] = df["Resume_str"]

    # 🔽 FIX LABELS HERE 🔽
    df["Label"] = df["Label"].astype(str)
    df["Label"] = df["Label"].str.replace(r"[^A-Za-z ]+", "", regex=True)
    df["Label"] = df["Label"].str.strip()
    # 🔼 END FIX 🔼

    return df[["Text", "Label"]]

if __name__ == "__main__":
    df = load_dataset1()
    print(df.head())
    print("Loaded rows:", len(df))
