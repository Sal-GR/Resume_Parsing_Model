from load_dataset1 import load_dataset1
from load_dataset2 import load_dataset2
import pandas as pd

def combine():
    df1 = load_dataset1()
    df2 = load_dataset2()

    df = pd.concat([df1, df2], ignore_index=True)
    return df

if __name__ == "__main__":
    df = combine()
    print(df.head())
    print("Total resumes:", len(df))
    df.to_csv("data/clean/combined_resumes.csv", index=False)
