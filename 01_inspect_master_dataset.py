import os
import pandas as pd
import numpy as np

# ============================================================
# 1. FIND PROJECT FOLDER
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_FILE = os.path.join(
    PROJECT_DIR,
    "master_MITBIH_48_records_features.csv"
)

print("=" * 70)
print("ECG MASTER DATASET INSPECTION")
print("=" * 70)

print("\nProject folder:")
print(PROJECT_DIR)

print("\nCSV file:")
print(CSV_FILE)


# ============================================================
# 2. CHECK THAT FILE EXISTS
# ============================================================

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(
        "\nERROR: master_MITBIH_48_records_features.csv "
        "was not found in the project folder."
    )

print("\nFile found successfully.")


# ============================================================
# 3. LOAD DATA
# ============================================================

df = pd.read_csv(CSV_FILE)

print("\nDataset loaded successfully.")


# ============================================================
# 4. BASIC DATASET SIZE
# ============================================================

print("\n" + "=" * 70)
print("1. DATASET SIZE")
print("=" * 70)

print("Number of rows    :", df.shape[0])
print("Number of columns :", df.shape[1])


# ============================================================
# 5. COLUMN NAMES
# ============================================================

print("\n" + "=" * 70)
print("2. COLUMN NAMES")
print("=" * 70)

for i, col in enumerate(df.columns, start=1):
    print(f"{i:2d}. {col}")


# ============================================================
# 6. DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("3. DATA TYPES")
print("=" * 70)

print(df.dtypes)


# ============================================================
# 7. FIRST 10 ROWS
# ============================================================

print("\n" + "=" * 70)
print("4. FIRST 10 ROWS")
print("=" * 70)

print(df.head(10).to_string())


# ============================================================
# 8. MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("5. MISSING VALUES")
print("=" * 70)

missing_count = df.isna().sum()
missing_percent = (missing_count / len(df)) * 100

missing_table = pd.DataFrame({
    "Missing_Count": missing_count,
    "Missing_Percent": missing_percent
})

print(missing_table.to_string())

print("\nTotal missing values in entire dataset:",
      df.isna().sum().sum())


# ============================================================
# 9. INFINITE VALUES
# ============================================================

print("\n" + "=" * 70)
print("6. INFINITE VALUES")
print("=" * 70)

numeric_columns = df.select_dtypes(include=np.number).columns

infinite_count = np.isinf(df[numeric_columns]).sum()

print(infinite_count.to_string())

print("\nTotal infinite numeric values:",
      infinite_count.sum())


# ============================================================
# 10. RECORD STRUCTURE
# ============================================================

print("\n" + "=" * 70)
print("7. RECORD STRUCTURE")
print("=" * 70)

print("\nNumber of unique records:",
      df["record_id"].nunique())

print("\nRecords:")
print(sorted(df["record_id"].unique()))


# ============================================================
# 11. BEATS PER RECORD
# ============================================================

print("\n" + "=" * 70)
print("8. BEATS PER RECORD")
print("=" * 70)

beats_per_record = (
    df.groupby("record_id")
      .size()
      .sort_index()
)

print(beats_per_record.to_string())


# ============================================================
# 12. CHANNEL STRUCTURE
# ============================================================

print("\n" + "=" * 70)
print("9. CHANNEL STRUCTURE")
print("=" * 70)

print("\nUnique channels:")
print(df["channel"].unique())

print("\nNumber of unique channels:",
      df["channel"].nunique())


print("\nRows per channel:")
print(df["channel"].value_counts().to_string())


# ============================================================
# 13. RECORD × CHANNEL STRUCTURE
# ============================================================

print("\n" + "=" * 70)
print("10. RECORD × CHANNEL STRUCTURE")
print("=" * 70)

record_channel = pd.crosstab(
    df["record_id"],
    df["channel"]
)

print(record_channel.to_string())


# ============================================================
# 14. BEAT LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("11. BEAT LABEL DISTRIBUTION")
print("=" * 70)

label_counts = df["beat_label"].value_counts()

print(label_counts.to_string())


# ============================================================
# 15. BEAT LABEL PERCENTAGES
# ============================================================

print("\n" + "=" * 70)
print("12. BEAT LABEL PERCENTAGES")
print("=" * 70)

label_percent = (
    df["beat_label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(3)
)

label_distribution = pd.DataFrame({
    "Count": label_counts,
    "Percentage": label_percent
})

print(label_distribution.to_string())


# ============================================================
# 16. LABEL DISTRIBUTION BY RECORD
# ============================================================

print("\n" + "=" * 70)
print("13. LABEL DISTRIBUTION BY RECORD")
print("=" * 70)

record_label_table = pd.crosstab(
    df["record_id"],
    df["beat_label"]
)

print(record_label_table.to_string())


# ============================================================
# 17. LABEL DISTRIBUTION BY CHANNEL
# ============================================================

print("\n" + "=" * 70)
print("14. LABEL DISTRIBUTION BY CHANNEL")
print("=" * 70)

channel_label_table = pd.crosstab(
    df["channel"],
    df["beat_label"]
)

print(channel_label_table.to_string())


# ============================================================
# 18. DUPLICATE ROWS
# ============================================================

print("\n" + "=" * 70)
print("15. DUPLICATE ROWS")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print("Duplicate rows:", duplicate_count)


# ============================================================
# 19. DUPLICATE BEAT IDENTIFIERS
# ============================================================

print("\n" + "=" * 70)
print("16. DUPLICATE BEAT IDENTIFIERS")
print("=" * 70)

identifier_columns = [
    "record_id",
    "channel",
    "beat_index"
]

duplicate_beats = df.duplicated(
    subset=identifier_columns
).sum()

print(
    "Duplicate record/channel/beat combinations:",
    duplicate_beats
)


# ============================================================
# 20. NUMERIC FEATURE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("17. NUMERIC FEATURE SUMMARY")
print("=" * 70)

print(
    df[numeric_columns]
    .describe()
    .T
    .to_string()
)


# ============================================================
# 21. EXTREME / INVALID VALUES
# ============================================================

print("\n" + "=" * 70)
print("18. EXTREME / INVALID VALUES")
print("=" * 70)

for col in numeric_columns:

    series = df[col]

    if series.notna().sum() == 0:
        continue

    print(
        f"{col:30s} "
        f"min={series.min():.4f} "
        f"max={series.max():.4f}"
    )


# ============================================================
# 22. RECORD 201 / 202 CHECK
# ============================================================

print("\n" + "=" * 70)
print("19. RECORD 201 / 202 CHECK")
print("=" * 70)

for record in ["201", "202"]:

    if record in df["record_id"].astype(str).values:

        subset = df[
            df["record_id"].astype(str) == record
        ]

        print(
            f"\nRecord {record}: "
            f"{len(subset)} rows"
        )

        print(
            "Channels:",
            subset["channel"].unique()
        )

    else:

        print(
            f"Record {record} not found."
        )


# ============================================================
# 23. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("20. FINAL INSPECTION SUMMARY")
print("=" * 70)

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Records:", df["record_id"].nunique())
print("Channels:", df["channel"].nunique())
print("Beat labels:", df["beat_label"].nunique())
print("Missing values:", df.isna().sum().sum())
print("Infinite values:", infinite_count.sum())
print("Duplicate rows:", duplicate_count)
print("Duplicate beat IDs:", duplicate_beats)

print("\nInspection completed.")
print("=" * 70)