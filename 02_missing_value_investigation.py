# ============================================================
# STEP 2: MISSING VALUE INVESTIGATION
# MIT-BIH ARRHYTHMIA DATABASE
# ============================================================
#
# PURPOSE:
# Investigate missing values in the completed master feature
# dataset before making any decision about:
#
#   - deleting rows
#   - deleting features
#   - imputing values
#   - selecting ML features
#
# IMPORTANT:
# This script DOES NOT modify the original dataset.
#
# ============================================================

import os
import pandas as pd
import numpy as np


# ============================================================
# 1. PROJECT LOCATION
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(
    PROJECT_DIR,
    "master_MITBIH_48_records_features.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "missing_value_investigation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 70)
print("STEP 2: MISSING VALUE INVESTIGATION")
print("=" * 70)

print("\nProject folder:")
print(PROJECT_DIR)

print("\nInput file:")
print(INPUT_FILE)

if not os.path.exists(INPUT_FILE):
    print("\nERROR: Master CSV file was not found.")
    print("Expected file:")
    print(INPUT_FILE)
    raise FileNotFoundError(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully.")

print("\nDataset shape:")
print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")


# ============================================================
# 3. BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("3. BASIC DATASET INFORMATION")
print("=" * 70)

print("\nColumns:")
for i, col in enumerate(df.columns, start=1):
    print(f"{i:2d}. {col}")

print("\nData types:")
print(df.dtypes)


# ============================================================
# 4. OVERALL MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("4. OVERALL MISSING VALUES BY FEATURE")
print("=" * 70)

missing_count = df.isna().sum()

missing_percent = (
    df.isna().mean() * 100
)

missing_report = pd.DataFrame({
    "feature": df.columns,
    "missing_count": missing_count.values,
    "total_rows": len(df),
    "missing_percent": missing_percent.values,
    "non_missing_count": (len(df) - missing_count).values,
    "non_missing_percent": (100 - missing_percent).values
})

missing_report = missing_report.sort_values(
    by="missing_count",
    ascending=False
).reset_index(drop=True)

print("\nMissing-value summary:")
print(missing_report.to_string(index=False))

missing_report.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "01_overall_missing_values.csv"
    ),
    index=False
)


# ============================================================
# 5. TOTAL MISSING VALUES
# ============================================================

total_missing = int(df.isna().sum().sum())

total_cells = df.shape[0] * df.shape[1]

overall_missing_percent = (
    total_missing / total_cells * 100
)

print("\n" + "=" * 70)
print("5. TOTAL MISSING VALUES")
print("=" * 70)

print(f"\nTotal cells       : {total_cells:,}")
print(f"Total missing    : {total_missing:,}")
print(f"Overall missing %: {overall_missing_percent:.4f}%")


# ============================================================
# 6. FEATURES WITH NO MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("6. FEATURES WITH NO MISSING VALUES")
print("=" * 70)

complete_features = missing_report[
    missing_report["missing_count"] == 0
]["feature"].tolist()

if len(complete_features) == 0:
    print("\nNo feature is completely free of missing values.")
else:
    for feature in complete_features:
        print(feature)


# ============================================================
# 7. FEATURES WITH MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("7. FEATURES CONTAINING MISSING VALUES")
print("=" * 70)

incomplete_features = missing_report[
    missing_report["missing_count"] > 0
]["feature"].tolist()

print(
    f"\nNumber of features containing missing values: "
    f"{len(incomplete_features)}"
)

for feature in incomplete_features:
    row = missing_report[
        missing_report["feature"] == feature
    ].iloc[0]

    print(
        f"{feature:<25} "
        f"{int(row['missing_count']):>8,} "
        f"({row['missing_percent']:>7.3f}%)"
    )


# ============================================================
# 8. DEFINE ECG FEATURE COLUMNS
# ============================================================

# These are the physiological / morphological features
# extracted from the ECG.
#
# Absolute sample/time positions are intentionally separated
# because they should generally NOT be used as ML predictors.

ecg_feature_columns = [
    "RR_interval_ms",
    "Heart_rate_bpm",
    "R_amplitude_mV",

    "P_amplitude_mV",
    "P_duration_ms",

    "QRS_duration_ms",

    "PR_interval_ms",

    "QT_interval_ms",

    "T_amplitude_mV",
    "T_duration_ms"
]

# Keep only columns that actually exist
ecg_feature_columns = [
    col for col in ecg_feature_columns
    if col in df.columns
]

print("\n" + "=" * 70)
print("8. SELECTED PHYSIOLOGICAL FEATURES FOR MISSING-VALUE ANALYSIS")
print("=" * 70)

for feature in ecg_feature_columns:
    print(feature)


# ============================================================
# 9. MISSINGNESS PER BEAT
# ============================================================

print("\n" + "=" * 70)
print("9. MISSING VALUES PER BEAT")
print("=" * 70)

# Count missing ECG feature values for every row/beat
df["_missing_feature_count"] = (
    df[ecg_feature_columns]
    .isna()
    .sum(axis=1)
)

df["_missing_feature_percent"] = (
    df["_missing_feature_count"]
    / len(ecg_feature_columns)
    * 100
)

missing_per_beat_distribution = (
    df["_missing_feature_count"]
    .value_counts()
    .sort_index()
    .reset_index()
)

missing_per_beat_distribution.columns = [
    "number_of_missing_features",
    "number_of_beats"
]

missing_per_beat_distribution["percentage_of_beats"] = (
    missing_per_beat_distribution["number_of_beats"]
    / len(df)
    * 100
)

print(
    "\nDistribution of missing features per row:"
)

print(
    missing_per_beat_distribution.to_string(
        index=False
    )
)

missing_per_beat_distribution.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "02_missing_features_per_beat.csv"
    ),
    index=False
)


# ============================================================
# 10. COMPLETELY COMPLETE BEATS
# ============================================================

print("\n" + "=" * 70)
print("10. COMPLETELY COMPLETE BEATS")
print("=" * 70)

complete_beats = (
    df["_missing_feature_count"] == 0
).sum()

one_missing = (
    df["_missing_feature_count"] == 1
).sum()

two_missing = (
    df["_missing_feature_count"] == 2
).sum()

three_or_more_missing = (
    df["_missing_feature_count"] >= 3
).sum()

print(
    f"\nTotal rows/beats                  : {len(df):,}"
)

print(
    f"Complete ECG feature rows        : "
    f"{complete_beats:,} "
    f"({complete_beats / len(df) * 100:.3f}%)"
)

print(
    f"Rows with exactly 1 missing       : "
    f"{one_missing:,} "
    f"({one_missing / len(df) * 100:.3f}%)"
)

print(
    f"Rows with exactly 2 missing       : "
    f"{two_missing:,} "
    f"({two_missing / len(df) * 100:.3f}%)"
)

print(
    f"Rows with 3+ missing              : "
    f"{three_or_more_missing:,} "
    f"({three_or_more_missing / len(df) * 100:.3f}%)"
)


# ============================================================
# 11. MISSING VALUES BY BEAT LABEL
# ============================================================

print("\n" + "=" * 70)
print("11. MISSING VALUES BY BEAT LABEL")
print("=" * 70)

label_missing = (
    df.groupby("beat_label")[ecg_feature_columns]
    .apply(lambda x: x.isna().sum())
)

label_total = (
    df.groupby("beat_label")
    .size()
    .rename("total_beats")
)

label_missing_percent = (
    label_missing
    .div(label_total, axis=0)
    * 100
)

# Add total number of missing values across ECG features
label_summary = label_missing.copy()

label_summary["total_beats"] = label_total

label_summary["total_missing_feature_values"] = (
    label_missing.sum(axis=1)
)

label_summary["missing_feature_values_percent"] = (
    label_missing.sum(axis=1)
    /
    (label_total * len(ecg_feature_columns))
    * 100
)

label_summary = label_summary.reset_index()

print(
    "\nMissing values by beat label:"
)

print(
    label_summary.to_string(index=False)
)

label_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "03_missing_values_by_beat_label.csv"
    ),
    index=False
)


# ============================================================
# 12. MISSING PERCENTAGE BY LABEL
# ============================================================

print("\n" + "=" * 70)
print("12. MISSING PERCENTAGE BY BEAT LABEL")
print("=" * 70)

label_percent_summary = (
    label_missing_percent
    .copy()
)

label_percent_summary["total_beats"] = label_total

label_percent_summary[
    "overall_missing_percentage"
] = (
    label_missing.sum(axis=1)
    /
    (label_total * len(ecg_feature_columns))
    * 100
)

label_percent_summary = (
    label_percent_summary
    .sort_values(
        "overall_missing_percentage",
        ascending=False
    )
    .reset_index()
)

print(
    label_percent_summary.to_string(index=False)
)

label_percent_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "04_missing_percentage_by_label.csv"
    ),
    index=False
)


# ============================================================
# 13. MISSING VALUES BY RECORD
# ============================================================

print("\n" + "=" * 70)
print("13. MISSING VALUES BY RECORD")
print("=" * 70)

record_missing = (
    df.groupby("record_id")[ecg_feature_columns]
    .apply(lambda x: x.isna().sum())
)

record_total = (
    df.groupby("record_id")
    .size()
    .rename("total_rows")
)

record_summary = record_missing.copy()

record_summary["total_rows"] = record_total

record_summary["total_missing_feature_values"] = (
    record_missing.sum(axis=1)
)

record_summary["missing_feature_values_percent"] = (
    record_missing.sum(axis=1)
    /
    (record_total * len(ecg_feature_columns))
    * 100
)

record_summary = (
    record_summary
    .reset_index()
    .sort_values(
        "missing_feature_values_percent",
        ascending=False
    )
)

print(
    "\nRecords with highest missing-value percentage:"
)

print(
    record_summary.to_string(index=False)
)

record_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "05_missing_values_by_record.csv"
    ),
    index=False
)


# ============================================================
# 14. MISSING VALUES BY CHANNEL
# ============================================================

print("\n" + "=" * 70)
print("14. MISSING VALUES BY CHANNEL")
print("=" * 70)

channel_missing = (
    df.groupby("channel")[ecg_feature_columns]
    .apply(lambda x: x.isna().sum())
)

channel_total = (
    df.groupby("channel")
    .size()
    .rename("total_rows")
)

channel_summary = channel_missing.copy()

channel_summary["total_rows"] = channel_total

channel_summary["total_missing_feature_values"] = (
    channel_missing.sum(axis=1)
)

channel_summary["missing_feature_values_percent"] = (
    channel_missing.sum(axis=1)
    /
    (channel_total * len(ecg_feature_columns))
    * 100
)

channel_summary = (
    channel_summary
    .reset_index()
    .sort_values(
        "missing_feature_values_percent",
        ascending=False
    )
)

print(
    "\nMissing values by channel:"
)

print(
    channel_summary.to_string(index=False)
)

channel_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "06_missing_values_by_channel.csv"
    ),
    index=False
)


# ============================================================
# 15. MISSING VALUE PATTERN
# ============================================================

print("\n" + "=" * 70)
print("15. MOST COMMON MISSING-VALUE PATTERNS")
print("=" * 70)

# Convert each row to a missing/non-missing pattern
#
# Example:
# 0000000000 = no missing values
# 0010000000 = one specific feature missing
#
# This helps identify whether missing values occur
# independently or in groups.

pattern_matrix = (
    df[ecg_feature_columns]
    .isna()
    .astype(int)
)

df["_missing_pattern"] = (
    pattern_matrix.astype(str)
    .agg("".join, axis=1)
)

pattern_summary = (
    df["_missing_pattern"]
    .value_counts()
    .reset_index()
)

pattern_summary.columns = [
    "missing_pattern",
    "number_of_rows"
]

pattern_summary["percentage_of_rows"] = (
    pattern_summary["number_of_rows"]
    / len(df)
    * 100
)

pattern_summary["number_of_missing_features"] = (
    pattern_summary["missing_pattern"]
    .str.count("1")
)

print(
    "\nTop 30 missing-value patterns:"
)

print(
    pattern_summary.head(30).to_string(index=False)
)

pattern_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "07_missing_value_patterns.csv"
    ),
    index=False
)


# ============================================================
# 16. FEATURE MISSINGNESS CO-OCCURRENCE
# ============================================================

print("\n" + "=" * 70)
print("16. MISSING-VALUE CO-OCCURRENCE")
print("=" * 70)

# Number of rows where two features are BOTH missing

cooccurrence = pd.DataFrame(
    0,
    index=ecg_feature_columns,
    columns=ecg_feature_columns
)

for feature_a in ecg_feature_columns:
    for feature_b in ecg_feature_columns:

        both_missing = (
            df[feature_a].isna()
            &
            df[feature_b].isna()
        ).sum()

        cooccurrence.loc[
            feature_a,
            feature_b
        ] = both_missing

print(
    "\nNumber of rows where two features are simultaneously missing:"
)

print(cooccurrence)

cooccurrence.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "08_missing_value_cooccurrence.csv"
    )
)


# ============================================================
# 17. MISSINGNESS RATE FOR EACH FEATURE
# ============================================================

print("\n" + "=" * 70)
print("17. MISSINGNESS RATE FOR PHYSIOLOGICAL FEATURES")
print("=" * 70)

feature_missingness = []

for feature in ecg_feature_columns:

    count = int(df[feature].isna().sum())

    percent = (
        count / len(df) * 100
    )

    feature_missingness.append({
        "feature": feature,
        "missing_count": count,
        "missing_percentage": percent
    })

feature_missingness = pd.DataFrame(
    feature_missingness
).sort_values(
    "missing_percentage",
    ascending=False
)

print(
    feature_missingness.to_string(index=False)
)

feature_missingness.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "09_ecg_feature_missingness.csv"
    ),
    index=False
)


# ============================================================
# 18. ROWS WITH HIGH MISSINGNESS
# ============================================================

print("\n" + "=" * 70)
print("18. ROWS WITH HIGH MISSINGNESS")
print("=" * 70)

for threshold in [3, 5, 7, 9]:

    count = (
        df["_missing_feature_count"] >= threshold
    ).sum()

    percent = (
        count / len(df) * 100
    )

    print(
        f"Rows with >= {threshold} missing features: "
        f"{count:,} "
        f"({percent:.3f}%)"
    )


# ============================================================
# 19. MISSINGNESS BY RECORD AND LABEL
# ============================================================

print("\n" + "=" * 70)
print("19. MISSINGNESS BY RECORD + BEAT LABEL")
print("=" * 70)

record_label_missing = (
    df.groupby(
        ["record_id", "beat_label"]
    )[ecg_feature_columns]
    .apply(lambda x: x.isna().sum())
)

record_label_total = (
    df.groupby(
        ["record_id", "beat_label"]
    )
    .size()
    .rename("total_beats")
)

record_label_summary = (
    record_label_missing
    .copy()
)

record_label_summary["total_beats"] = (
    record_label_total
)

record_label_summary[
    "total_missing_feature_values"
] = (
    record_label_missing.sum(axis=1)
)

record_label_summary[
    "missing_feature_values_percent"
] = (
    record_label_missing.sum(axis=1)
    /
    (
        record_label_total
        * len(ecg_feature_columns)
    )
    * 100
)

record_label_summary = (
    record_label_summary
    .reset_index()
    .sort_values(
        "missing_feature_values_percent",
        ascending=False
    )
)

print(
    "\nTop 30 record/label combinations by missingness:"
)

print(
    record_label_summary
    .head(30)
    .to_string(index=False)
)

record_label_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "10_missing_values_record_label.csv"
    ),
    index=False
)


# ============================================================
# 20. COMPLETE BEATS BY LABEL
# ============================================================

print("\n" + "=" * 70)
print("20. COMPLETE BEATS BY LABEL")
print("=" * 70)

complete_by_label = (
    df.groupby("beat_label")
    .apply(
        lambda x: (
            x["_missing_feature_count"] == 0
        ).sum()
    )
    .rename("complete_beats")
)

total_by_label = (
    df.groupby("beat_label")
    .size()
    .rename("total_beats")
)

complete_label_summary = pd.concat(
    [
        total_by_label,
        complete_by_label
    ],
    axis=1
)

complete_label_summary[
    "complete_percentage"
] = (
    complete_label_summary["complete_beats"]
    /
    complete_label_summary["total_beats"]
    * 100
)

complete_label_summary[
    "incomplete_beats"
] = (
    complete_label_summary["total_beats"]
    -
    complete_label_summary["complete_beats"]
)

complete_label_summary = (
    complete_label_summary
    .sort_values(
        "complete_percentage"
    )
    .reset_index()
)

print(
    complete_label_summary.to_string(index=False)
)

complete_label_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "11_complete_beats_by_label.csv"
    ),
    index=False
)


# ============================================================
# 21. COMPLETE BEATS BY RECORD
# ============================================================

print("\n" + "=" * 70)
print("21. COMPLETE BEATS BY RECORD")
print("=" * 70)

complete_by_record = (
    df.groupby("record_id")
    .apply(
        lambda x: (
            x["_missing_feature_count"] == 0
        ).sum()
    )
    .rename("complete_beats")
)

total_by_record = (
    df.groupby("record_id")
    .size()
    .rename("total_rows")
)

complete_record_summary = pd.concat(
    [
        total_by_record,
        complete_by_record
    ],
    axis=1
)

complete_record_summary[
    "complete_percentage"
] = (
    complete_record_summary["complete_beats"]
    /
    complete_record_summary["total_rows"]
    * 100
)

complete_record_summary[
    "incomplete_rows"
] = (
    complete_record_summary["total_rows"]
    -
    complete_record_summary["complete_beats"]
)

complete_record_summary = (
    complete_record_summary
    .sort_values(
        "complete_percentage"
    )
    .reset_index()
)

print(
    complete_record_summary.to_string(index=False)
)

complete_record_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "12_complete_beats_by_record.csv"
    ),
    index=False
)


# ============================================================
# 22. SUMMARY OF MISSINGNESS
# ============================================================

print("\n" + "=" * 70)
print("22. FINAL MISSING-VALUE SUMMARY")
print("=" * 70)

print(
    f"\nDataset rows                    : {len(df):,}"
)

print(
    f"Dataset columns                : {len(df.columns) - 3}"
)

print(
    f"Total ECG feature columns      : "
    f"{len(ecg_feature_columns)}"
)

print(
    f"Total missing cells             : "
    f"{total_missing:,}"
)

print(
    f"Overall missing percentage      : "
    f"{overall_missing_percent:.4f}%"
)

print(
    f"Completely complete rows        : "
    f"{complete_beats:,} "
    f"({complete_beats / len(df) * 100:.3f}%)"
)

print(
    f"Rows with >=3 missing features  : "
    f"{three_or_more_missing:,} "
    f"({three_or_more_missing / len(df) * 100:.3f}%)"
)


# ============================================================
# 23. SAVE A CLEAN COPY OF THE INVESTIGATION DATA
# ============================================================

# Remove temporary investigation columns before saving.
# The original master CSV is NOT modified.

investigation_df = df.drop(
    columns=[
        "_missing_feature_count",
        "_missing_feature_percent",
        "_missing_pattern"
    ],
    errors="ignore"
)

investigation_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "master_features_copy_for_investigation.csv"
    ),
    index=False
)


# ============================================================
# 24. SAVE TEXT SUMMARY
# ============================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "missing_value_summary.txt"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "MIT-BIH MISSING VALUE INVESTIGATION\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        f"Dataset rows: {len(df):,}\n"
    )

    f.write(
        f"Dataset columns: {len(df.columns) - 3}\n"
    )

    f.write(
        f"ECG feature columns analyzed: "
        f"{len(ecg_feature_columns)}\n"
    )

    f.write(
        f"Total missing cells: "
        f"{total_missing:,}\n"
    )

    f.write(
        f"Overall missing percentage: "
        f"{overall_missing_percent:.4f}%\n"
    )

    f.write(
        f"Complete rows: "
        f"{complete_beats:,} "
        f"({complete_beats / len(df) * 100:.3f}%)\n"
    )

    f.write(
        f"Rows with exactly 1 missing: "
        f"{one_missing:,}\n"
    )

    f.write(
        f"Rows with exactly 2 missing: "
        f"{two_missing:,}\n"
    )

    f.write(
        f"Rows with 3+ missing: "
        f"{three_or_more_missing:,}\n"
    )

    f.write(
        "\n\nFeature missingness:\n"
    )

    f.write(
        feature_missingness.to_string(
            index=False
        )
    )


# ============================================================
# 25. FINISHED
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 COMPLETE")
print("=" * 70)

print("\nNo changes were made to the original master CSV.")

print("\nInvestigation results saved in:")

print(OUTPUT_DIR)

print("\nGenerated files:")

for filename in sorted(
    os.listdir(OUTPUT_DIR)
):

    filepath = os.path.join(
        OUTPUT_DIR,
        filename
    )

    if os.path.isfile(filepath):

        size_kb = (
            os.path.getsize(filepath)
            / 1024
        )

        print(
            f"  {filename:<50} "
            f"{size_kb:>10.1f} KB"
        )

print("\n" + "=" * 70)
print("IMPORTANT")
print("=" * 70)

print(
    "\nDo NOT delete rows or fill missing values yet."
)

print(
    "We will use this investigation to decide the "
    "correct ML preprocessing strategy."
)