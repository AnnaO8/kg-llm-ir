import re
import pandas as pd

def clean_relation(text):
    match = re.search(r"\[:([\w\d_]+)\]", text)
    if match:
        return match.group(1).replace("_", " ")
    return text

def clean_text_preserve_case(text):
    match = re.search(r"\(:([\w\d\s_]+)", text)
    if match:
        return match.group(1).replace("_", " ").strip() 
    return text.strip()

def clean_kg(df):
    df["n"] = df["n"].apply(clean_text_preserve_case)
    df["m"] = df["m"].apply(clean_text_preserve_case)
    df["r"] = df["r"].apply(clean_relation)
    return df

if __name__ == "__main__":

    kg = pd.read_csv("data/Economic-KG-with-blank.csv")
    kg_clean = clean_kg(kg)
    #print(kg_clean.head())
    kg_clean.to_csv("data/Economic_KG.csv", index=False)