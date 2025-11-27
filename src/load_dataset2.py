import pandas as pd
import re

def clean_text(s):
    import re

def clean_text(s):
    if not isinstance(s, str):
        return ""

    # Remove BOM, weird unicode line separators
    s = s.replace("\ufeff", "")
    s = re.sub(r"[\u2028\u2029]", " ", s)

    # Normalize all carriage returns
    s = s.replace("\r", "\n")

    # Replace multiple blank lines with a single newline
    s = re.sub(r"\n+", "\n", s)

    # Strip leading/trailing whitespace
    s = s.strip()

    return s


def load_dataset1(path="data/clean/dataset2.csv"):
    df = pd.read_csv(path, encoding="utf-8", engine="python")
    df["Text"] = df["Text"].apply(clean_text)
    df = df.rename(columns={"Category": "Label"})
    return df[["Text", "Label"]]

if __name__ == "__main__":
    df = load_dataset1()
    print(df.head(), len(df))

