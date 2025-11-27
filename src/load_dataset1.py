import pandas as pd
import re
from bs4 import BeautifulSoup

def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = s.replace("\ufeff", "")
    s = re.sub(r"[\u2028\u2029]", " ", s)
    return s.strip()

def html_to_text(html):
    if not isinstance(html, str):
        return ""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n").strip()

def load_dataset2(path="data/clean/dataset1.csv"):
    df = pd.read_csv(
        path,
        encoding="utf-8",
        engine="python",
        quoting=1,
        on_bad_lines="skip"
    )
    df["Resume_str"] = df["Resume_str"].apply(clean_text)
    df["Resume_html"] = df["Resume_html"].apply(clean_text)
    df["Text"] = df["Resume_str"]  # use text version
    
    # Also extract text from HTML in case it's useful
    df["Text_html"] = df["Resume_html"].apply(html_to_text)

    df = df.rename(columns={"Category": "Label"})
    return df[["Text", "Label"]]

if __name__ == "__main__":
    df = load_dataset2()
    print(df.head(), len(df))
