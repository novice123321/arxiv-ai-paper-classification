"""Clean arXiv metadata and build tabular features for analysis/modeling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_INPUT = DATA_DIR / "raw_arxiv.csv"
CLEAN_OUTPUT = DATA_DIR / "clean_arxiv.csv"
PROFILE_OUTPUT = PROJECT_ROOT / "results" / "data_quality_summary.txt"


NUMERIC_FEATURES = [
    "title_length",
    "summary_length",
    "author_count",
    "category_count",
]


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def count_separated_items(value: object) -> int:
    text = normalize_text(value)
    if not text:
        return 0
    return len([item for item in text.split(";") if item.strip()])


def add_iqr_flag(df: pd.DataFrame, column: str) -> pd.Series:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (df[column] < lower) | (df[column] > upper)


def main() -> None:
    if not RAW_INPUT.exists():
        raise FileNotFoundError(f"Missing raw data file: {RAW_INPUT}")

    df = pd.read_csv(RAW_INPUT, encoding="utf-8-sig")
    original_rows = len(df)
    missing_before = df.isna().sum()

    required_columns = ["arxiv_id", "title", "summary", "primary_category"]
    df = df.dropna(subset=required_columns).copy()
    df = df.drop_duplicates(subset=["arxiv_id"]).copy()

    for column in ["title", "summary", "authors", "categories", "primary_category"]:
        df[column] = df[column].map(normalize_text)

    df["published"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
    df["updated"] = pd.to_datetime(df["updated"], errors="coerce", utc=True)
    df = df.dropna(subset=["published"]).copy()
    df["published_year"] = df["published"].dt.year.astype(int)
    df["published_month"] = df["published"].dt.month.astype(int)
    df["published_ym"] = df["published"].dt.strftime("%Y-%m")

    df["author_count"] = df["authors"].map(count_separated_items)
    df["category_count"] = df["categories"].map(count_separated_items)
    df["title_length"] = df["title"].str.len()
    df["summary_length"] = df["summary"].str.len()
    df["text"] = df["title"] + " " + df["summary"]

    df["summary_length_iqr_outlier"] = add_iqr_flag(df, "summary_length")
    df["author_count_iqr_outlier"] = add_iqr_flag(df, "author_count")

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[NUMERIC_FEATURES])
    for idx, column in enumerate(NUMERIC_FEATURES):
        df[f"{column}_std"] = scaled_values[:, idx]

    df = df.sort_values(["source_category", "published"], ascending=[True, False])
    CLEAN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_OUTPUT, index=False, encoding="utf-8-sig")

    PROFILE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with PROFILE_OUTPUT.open("w", encoding="utf-8") as file:
        file.write("Data quality summary\n")
        file.write(f"Original rows: {original_rows}\n")
        file.write(f"Clean rows: {len(df)}\n")
        file.write(f"Duplicated arxiv_id after cleaning: {df['arxiv_id'].duplicated().sum()}\n")
        file.write("\nMissing values before cleaning:\n")
        file.write(missing_before.to_string())
        file.write("\n\nMissing values after cleaning:\n")
        file.write(df.isna().sum().to_string())
        file.write("\n\nIQR outlier counts:\n")
        file.write(
            f"\nsummary_length: {int(df['summary_length_iqr_outlier'].sum())}"
            f"\nauthor_count: {int(df['author_count_iqr_outlier'].sum())}\n"
        )
        file.write("\nNumeric feature description:\n")
        file.write(df[NUMERIC_FEATURES].describe().to_string())
        file.write("\n\nPrimary category counts:\n")
        file.write(df["primary_category"].value_counts().to_string())

    print(f"Saved clean data: {CLEAN_OUTPUT}")
    print(f"Rows: {original_rows} -> {len(df)}")
    print(f"Outliers summary_length={int(df['summary_length_iqr_outlier'].sum())}")
    print(f"Outliers author_count={int(df['author_count_iqr_outlier'].sum())}")


if __name__ == "__main__":
    main()
