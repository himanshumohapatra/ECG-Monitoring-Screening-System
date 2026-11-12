# ============================================================
# 04_feature_selection_analysis.py
# MIT-BIH Arrhythmia Database
#
# Purpose:
#   Systematically evaluate the 10 candidate ECG features
#   before training the AI models.
#
# Checks:
#   1. Redundancy / correlation
#   2. Missingness
#   3. Class-wise usefulness
#   4. Feature distributions
#   5. Suspicious / extreme values
#   6. Core vs Extended feature sets
#
# IMPORTANT:
#   - Does NOT modify the original master CSV.
#   - Does NOT train the final AI models.
#   - Does NOT use beat_label as an input feature.
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# 1. AUTOMATIC PROJECT / FILE DISCOVERY
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(
    PROJECT_DIR,
    "master_MITBIH_48_records_features.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "feature_selection_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("STEP 4 - FEATURE SELECTION ANALYSIS")
print("=" * 80)

print("\nProject folder:")
print(PROJECT_DIR)

print("\nInput file:")
print(INPUT_FILE)

print("\nOutput folder:")
print(OUTPUT_DIR)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        "\nMaster CSV was not found.\n"
        "Expected:\n"
        + INPUT_FILE
    )

# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading master dataset...")

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully.")
print("Shape:", df.shape)

# ============================================================
# 3. DEFINE THE 10 CANDIDATE FEATURES
# ============================================================

FEATURES = [
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

TARGET = "beat_label"

# Verify columns exist
missing_columns = [
    col for col in FEATURES + [TARGET]
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "The following required columns are missing:\n"
        + "\n".join(missing_columns)
    )

# Convert candidate features to numeric
for col in FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("\nCandidate features:")
for i, feature in enumerate(FEATURES, start=1):
    print(f"{i:2d}. {feature}")

# ============================================================
# 4. FEATURE INFORMATION / BASIC SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("4. BASIC FEATURE INFORMATION")
print("=" * 80)

basic_rows = []

for feature in FEATURES:

    series = df[feature]

    total = len(series)
    missing = series.isna().sum()
    valid = series.notna().sum()

    basic_rows.append({
        "feature": feature,
        "total_rows": total,
        "valid_values": valid,
        "missing_values": missing,
        "missing_percent": (missing / total) * 100,
        "unique_values": series.nunique(dropna=True),
        "mean": series.mean(),
        "std": series.std(),
        "median": series.median(),
        "min": series.min(),
        "max": series.max(),
        "q01": series.quantile(0.01),
        "q05": series.quantile(0.05),
        "q25": series.quantile(0.25),
        "q75": series.quantile(0.75),
        "q95": series.quantile(0.95),
        "q99": series.quantile(0.99)
    })

basic_df = pd.DataFrame(basic_rows)

basic_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "01_feature_basic_statistics.csv"
    ),
    index=False
)

print(
    basic_df[
        [
            "feature",
            "missing_values",
            "missing_percent",
            "mean",
            "median",
            "min",
            "max"
        ]
    ].to_string(index=False)
)

# ============================================================
# 5. MISSINGNESS ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("5. MISSINGNESS ANALYSIS")
print("=" * 80)

missing_rows = []

for feature in FEATURES:

    missing_count = df[feature].isna().sum()

    missing_rows.append({
        "feature": feature,
        "missing_count": missing_count,
        "missing_percent": (
            missing_count / len(df)
        ) * 100,
        "available_count": df[feature].notna().sum(),
        "available_percent": (
            df[feature].notna().sum() / len(df)
        ) * 100
    })

missing_df = pd.DataFrame(missing_rows)

missing_df = missing_df.sort_values(
    "missing_percent",
    ascending=False
)

missing_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "02_feature_missingness.csv"
    ),
    index=False
)

print(
    missing_df.to_string(index=False)
)

# ============================================================
# 6. MISSINGNESS BY BEAT CLASS
# ============================================================

print("\n" + "=" * 80)
print("6. MISSINGNESS BY BEAT CLASS")
print("=" * 80)

class_missing_rows = []

labels = sorted(
    df[TARGET].dropna().unique()
)

for label in labels:

    class_df = df[df[TARGET] == label]

    for feature in FEATURES:

        missing_count = class_df[feature].isna().sum()

        class_missing_rows.append({
            "beat_label": label,
            "feature": feature,
            "class_count": len(class_df),
            "missing_count": missing_count,
            "missing_percent": (
                missing_count / len(class_df)
            ) * 100
        })

class_missing_df = pd.DataFrame(
    class_missing_rows
)

class_missing_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "03_missingness_by_beat_class.csv"
    ),
    index=False
)

# ============================================================
# 7. CORRELATION / REDUNDANCY ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("7. CORRELATION / REDUNDANCY ANALYSIS")
print("=" * 80)

feature_data = df[FEATURES]

# Pearson correlation
pearson_corr = feature_data.corr(
    method="pearson"
)

# Spearman correlation
spearman_corr = feature_data.corr(
    method="spearman"
)

pearson_corr.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "04_pearson_correlation_matrix.csv"
    )
)

spearman_corr.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "05_spearman_correlation_matrix.csv"
    )
)

print("\nPearson correlation matrix:")
print(pearson_corr.round(3).to_string())

# Find highly correlated feature pairs
correlation_pairs = []

for i in range(len(FEATURES)):

    for j in range(i + 1, len(FEATURES)):

        f1 = FEATURES[i]
        f2 = FEATURES[j]

        r = pearson_corr.loc[f1, f2]

        if pd.notna(r):

            correlation_pairs.append({
                "feature_1": f1,
                "feature_2": f2,
                "pearson_correlation": r,
                "absolute_correlation": abs(r)
            })

correlation_pairs_df = pd.DataFrame(
    correlation_pairs
).sort_values(
    "absolute_correlation",
    ascending=False
)

correlation_pairs_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "06_feature_correlation_pairs.csv"
    ),
    index=False
)

print("\nMost correlated feature pairs:")

print(
    correlation_pairs_df.head(15).to_string(
        index=False
    )
)

# ============================================================
# 8. SPECIFIC RR vs HR REDUNDANCY CHECK
# ============================================================

print("\n" + "=" * 80)
print("8. RR vs HEART RATE REDUNDANCY CHECK")
print("=" * 80)

rr_hr_df = df[
    ["RR_interval_ms", "Heart_rate_bpm"]
].dropna()

if len(rr_hr_df) > 0:

    rr_hr_corr = rr_hr_df[
        "RR_interval_ms"
    ].corr(
        rr_hr_df["Heart_rate_bpm"]
    )

    expected_hr = (
        60000 /
        rr_hr_df["RR_interval_ms"]
    )

    hr_error = (
        rr_hr_df["Heart_rate_bpm"]
        - expected_hr
    )

    print(
        f"Valid RR/HR pairs: {len(rr_hr_df):,}"
    )

    print(
        f"Correlation: {rr_hr_corr:.6f}"
    )

    print(
        f"Mean absolute HR conversion error: "
        f"{hr_error.abs().mean():.6f} bpm"
    )

    print(
        f"Maximum absolute HR conversion error: "
        f"{hr_error.abs().max():.6f} bpm"
    )

    rr_hr_summary = pd.DataFrame({
        "metric": [
            "valid_pairs",
            "correlation",
            "mean_absolute_conversion_error_bpm",
            "max_absolute_conversion_error_bpm"
        ],
        "value": [
            len(rr_hr_df),
            rr_hr_corr,
            hr_error.abs().mean(),
            hr_error.abs().max()
        ]
    })

else:

    rr_hr_summary = pd.DataFrame()

rr_hr_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "07_RR_vs_HR_redundancy.csv"
    ),
    index=False
)

# ============================================================
# 9. CLASS-WISE FEATURE STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("9. CLASS-WISE FEATURE STATISTICS")
print("=" * 80)

class_stats_rows = []

for label in labels:

    class_df = df[
        df[TARGET] == label
    ]

    for feature in FEATURES:

        values = class_df[feature].dropna()

        if len(values) == 0:
            continue

        class_stats_rows.append({
            "beat_label": label,
            "feature": feature,
            "n_valid": len(values),
            "mean": values.mean(),
            "std": values.std(),
            "median": values.median(),
            "min": values.min(),
            "max": values.max(),
            "q25": values.quantile(0.25),
            "q75": values.quantile(0.75)
        })

class_stats_df = pd.DataFrame(
    class_stats_rows
)

class_stats_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "08_classwise_feature_statistics.csv"
    ),
    index=False
)

# ============================================================
# 10. CLASS-WISE USEFULNESS USING STANDARDIZED MEAN DIFFERENCE
# ============================================================
#
# We calculate:
#
#   Cohen-like effect size between each class and all other
#   classes.
#
# This is NOT the final feature importance.
# It is only a screening method.
#
# Larger absolute values suggest stronger separation.
# ============================================================

print("\n" + "=" * 80)
print("10. CLASS-WISE FEATURE SEPARATION")
print("=" * 80)

separation_rows = []

for feature in FEATURES:

    overall_valid = df[feature].dropna()

    if len(overall_valid) == 0:
        continue

    overall_mean = overall_valid.mean()
    overall_std = overall_valid.std()

    if overall_std == 0 or pd.isna(overall_std):
        continue

    for label in labels:

        class_values = df.loc[
            df[TARGET] == label,
            feature
        ].dropna()

        if len(class_values) < 2:
            continue

        class_mean = class_values.mean()

        standardized_difference = (
            class_mean - overall_mean
        ) / overall_std

        separation_rows.append({
            "feature": feature,
            "beat_label": label,
            "class_n": len(class_values),
            "class_mean": class_mean,
            "overall_mean": overall_mean,
            "standardized_mean_difference":
                standardized_difference,
            "absolute_separation":
                abs(standardized_difference)
        })

separation_df = pd.DataFrame(
    separation_rows
)

separation_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "09_classwise_feature_separation.csv"
    ),
    index=False
)

# Overall class-separation score
feature_separation_summary = []

for feature in FEATURES:

    temp = separation_df[
        separation_df["feature"] == feature
    ]

    if len(temp) == 0:
        continue

    feature_separation_summary.append({
        "feature": feature,
        "mean_absolute_class_separation":
            temp["absolute_separation"].mean(),
        "max_absolute_class_separation":
            temp["absolute_separation"].max(),
        "median_absolute_class_separation":
            temp["absolute_separation"].median()
    })

feature_separation_summary_df = pd.DataFrame(
    feature_separation_summary
).sort_values(
    "mean_absolute_class_separation",
    ascending=False
)

feature_separation_summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "10_feature_class_separation_summary.csv"
    ),
    index=False
)

print(
    feature_separation_summary_df.to_string(
        index=False
    )
)

# ============================================================
# 11. CLASS-WISE MEDIAN DIFFERENCE
# ============================================================

print("\n" + "=" * 80)
print("11. CLASS-WISE MEDIAN TABLE")
print("=" * 80)

median_table = df.groupby(
    TARGET
)[FEATURES].median()

median_table.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "11_classwise_feature_medians.csv"
    )
)

print(
    median_table.round(3).to_string()
)

# ============================================================
# 12. SUSPICIOUS / EXTREME VALUE SCREENING
# ============================================================

print("\n" + "=" * 80)
print("12. SUSPICIOUS / EXTREME VALUE SCREENING")
print("=" * 80)

# These are screening thresholds.
# They are NOT automatic deletion rules.
#
# They are intentionally broad so that unusual values
# can be investigated rather than blindly removed.

SCREENING_LIMITS = {

    "RR_interval_ms": {
        "low": 250,
        "high": 3000
    },

    "Heart_rate_bpm": {
        "low": 20,
        "high": 220
    },

    "R_amplitude_mV": {
        "low": -5.5,
        "high": 5.5
    },

    "P_amplitude_mV": {
        "low": -2.0,
        "high": 2.0
    },

    "P_duration_ms": {
        "low": 20,
        "high": 300
    },

    "QRS_duration_ms": {
        "low": 20,
        "high": 500
    },

    "PR_interval_ms": {
        "low": 50,
        "high": 600
    },

    "QT_interval_ms": {
        "low": 100,
        "high": 850
    },

    "T_amplitude_mV": {
        "low": -5.5,
        "high": 5.5
    },

    "T_duration_ms": {
        "low": 20,
        "high": 550
    }
}

extreme_summary_rows = []

extreme_rows_all = []

for feature in FEATURES:

    low = SCREENING_LIMITS[
        feature
    ]["low"]

    high = SCREENING_LIMITS[
        feature
    ]["high"]

    valid_mask = df[feature].notna()

    low_mask = (
        valid_mask &
        (df[feature] < low)
    )

    high_mask = (
        valid_mask &
        (df[feature] > high)
    )

    flagged_mask = (
        low_mask |
        high_mask
    )

    flagged_count = flagged_mask.sum()

    extreme_summary_rows.append({
        "feature": feature,
        "screening_low": low,
        "screening_high": high,
        "valid_values": valid_mask.sum(),
        "below_low": low_mask.sum(),
        "above_high": high_mask.sum(),
        "flagged_total": flagged_count,
        "flagged_percent_of_all_rows":
            (flagged_count / len(df)) * 100
    })

    if flagged_count > 0:

        selected = df.loc[
            flagged_mask,
            [
                "record_id",
                "channel",
                "beat_index",
                "beat_label",
                feature
            ]
        ].copy()

        selected["feature"] = feature
        selected["screening_low"] = low
        selected["screening_high"] = high

        extreme_rows_all.append(
            selected
        )

extreme_summary_df = pd.DataFrame(
    extreme_summary_rows
)

extreme_summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "12_suspicious_extreme_value_summary.csv"
    ),
    index=False
)

if extreme_rows_all:

    extreme_rows_df = pd.concat(
        extreme_rows_all,
        ignore_index=True
    )

else:

    extreme_rows_df = pd.DataFrame()

extreme_rows_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "13_suspicious_extreme_value_rows.csv"
    ),
    index=False
)

print(
    extreme_summary_df.to_string(
        index=False
    )
)

# ============================================================
# 13. IQR OUTLIER ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("13. IQR OUTLIER ANALYSIS")
print("=" * 80)

iqr_rows = []

for feature in FEATURES:

    values = df[feature].dropna()

    if len(values) == 0:
        continue

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outlier_mask = (
        df[feature].notna() &
        (
            (df[feature] < lower) |
            (df[feature] > upper)
        )
    )

    outlier_count = outlier_mask.sum()

    iqr_rows.append({
        "feature": feature,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "lower_bound": lower,
        "upper_bound": upper,
        "outlier_count": outlier_count,
        "outlier_percent":
            (outlier_count / len(df)) * 100
    })

iqr_df = pd.DataFrame(
    iqr_rows
)

iqr_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "14_IQR_outlier_summary.csv"
    ),
    index=False
)

print(
    iqr_df.to_string(index=False)
)

# ============================================================
# 14. FEATURE DISTRIBUTION SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("14. FEATURE DISTRIBUTION SUMMARY")
print("=" * 80)

distribution_rows = []

for feature in FEATURES:

    values = df[feature].dropna()

    if len(values) == 0:
        continue

    distribution_rows.append({
        "feature": feature,
        "n": len(values),
        "mean": values.mean(),
        "std": values.std(),
        "variance": values.var(),
        "min": values.min(),
        "q01": values.quantile(0.01),
        "q05": values.quantile(0.05),
        "q25": values.quantile(0.25),
        "median": values.median(),
        "q75": values.quantile(0.75),
        "q95": values.quantile(0.95),
        "q99": values.quantile(0.99),
        "max": values.max(),
        "skewness": values.skew()
    })

distribution_df = pd.DataFrame(
    distribution_rows
)

distribution_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "15_feature_distribution_summary.csv"
    ),
    index=False
)

print(
    distribution_df.to_string(index=False)
)

# ============================================================
# 15. CORE FEATURE SET
# ============================================================

CORE_FEATURES = [
    "RR_interval_ms",
    "R_amplitude_mV",
    "P_amplitude_mV",
    "P_duration_ms"
]

# ============================================================
# 16. EXTENDED FEATURE SET
# ============================================================

EXTENDED_FEATURES = [
    "RR_interval_ms",
    "R_amplitude_mV",
    "P_amplitude_mV",
    "P_duration_ms",
    "QRS_duration_ms",
    "PR_interval_ms",
    "QT_interval_ms",
    "T_amplitude_mV",
    "T_duration_ms"
]

# ============================================================
# 17. HEART RATE ALTERNATIVE SET
# ============================================================

HR_ALTERNATIVE_FEATURES = [
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

# ============================================================
# 18. FEATURE SET COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("18. CORE vs EXTENDED FEATURE SETS")
print("=" * 80)

feature_sets = {
    "Core": CORE_FEATURES,
    "Extended": EXTENDED_FEATURES,
    "HR_Alternative": HR_ALTERNATIVE_FEATURES,
    "All_10": FEATURES
}

feature_set_rows = []

for set_name, feature_list in feature_sets.items():

    subset = df[feature_list]

    total_cells = subset.shape[0] * subset.shape[1]

    missing_cells = subset.isna().sum().sum()

    complete_rows = subset.notna().all(axis=1).sum()

    incomplete_rows = len(subset) - complete_rows

    feature_set_rows.append({
        "feature_set": set_name,
        "number_of_features": len(feature_list),
        "features": ", ".join(feature_list),
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "missing_percent":
            (missing_cells / total_cells) * 100,
        "complete_rows": complete_rows,
        "complete_row_percent":
            (complete_rows / len(subset)) * 100,
        "incomplete_rows": incomplete_rows,
        "incomplete_row_percent":
            (incomplete_rows / len(subset)) * 100
    })

feature_set_df = pd.DataFrame(
    feature_set_rows
)

feature_set_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "16_feature_set_comparison.csv"
    ),
    index=False
)

print(
    feature_set_df[
        [
            "feature_set",
            "number_of_features",
            "missing_percent",
            "complete_rows",
            "complete_row_percent"
        ]
    ].to_string(index=False)
)

# ============================================================
# 19. FEATURE DECISION TABLE
# ============================================================

print("\n" + "=" * 80)
print("19. PRELIMINARY FEATURE DECISION TABLE")
print("=" * 80)

decision_rows = []

for feature in FEATURES:

    missing_percent = (
        missing_df.loc[
            missing_df["feature"] == feature,
            "missing_percent"
        ].iloc[0]
    )

    if feature == "Heart_rate_bpm":

        preliminary_decision = (
            "EXCLUDE INITIALLY"
        )

        reason = (
            "Mathematically derived from RR_interval_ms; "
            "redundant information."
        )

    elif feature in CORE_FEATURES:

        preliminary_decision = (
            "CORE / KEEP"
        )

        reason = (
            "Low missingness and suitable for robust "
            "baseline modeling."
        )

    else:

        preliminary_decision = (
            "SPECIAL TREATMENT / TEST"
        )

        reason = (
            "Potentially useful morphology feature, "
            "but missingness/delineation reliability "
            "requires evaluation."
        )

    decision_rows.append({
        "feature": feature,
        "missing_percent": missing_percent,
        "preliminary_decision":
            preliminary_decision,
        "reason": reason
    })

decision_df = pd.DataFrame(
    decision_rows
)

decision_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "17_preliminary_feature_decision.csv"
    ),
    index=False
)

print(
    decision_df.to_string(index=False)
)

# ============================================================
# 20. AUTOMATIC RANKING SCORE
# ============================================================
#
# This is ONLY a screening score.
#
# It combines:
#   - low missingness
#   - class separation
#   - lower redundancy
#
# It must NOT be treated as the final AI feature importance.
# ============================================================

print("\n" + "=" * 80)
print("20. PRELIMINARY FEATURE SCREENING SCORE")
print("=" * 80)

score_rows = []

for feature in FEATURES:

    missing_percent = missing_df.loc[
        missing_df["feature"] == feature,
        "missing_percent"
    ].iloc[0]

    separation_match = (
        feature_separation_summary_df[
            feature_separation_summary_df["feature"] == feature
        ]
    )

    if len(separation_match) > 0:

        separation_score = (
            separation_match[
                "mean_absolute_class_separation"
            ].iloc[0]
        )

    else:

        separation_score = 0

    # Maximum absolute correlation with another feature
    pair_rows = correlation_pairs_df[
        (
            correlation_pairs_df["feature_1"] == feature
        )
        |
        (
            correlation_pairs_df["feature_2"] == feature
        )
    ]

    if len(pair_rows) > 0:

        max_corr = pair_rows[
            "absolute_correlation"
        ].max()

    else:

        max_corr = 0

    # Reliability score:
    # 100 = no missingness
    # decreases with missingness
    reliability_score = max(
        0,
        100 - missing_percent
    )

    # Separation is normalized later
    score_rows.append({
        "feature": feature,
        "missing_percent": missing_percent,
        "reliability_score": reliability_score,
        "mean_absolute_class_separation":
            separation_score,
        "maximum_absolute_correlation":
            max_corr
    })

screening_df = pd.DataFrame(
    score_rows
)

# Normalize class separation
max_sep = screening_df[
    "mean_absolute_class_separation"
].max()

if max_sep > 0:

    screening_df[
        "normalized_class_separation"
    ] = (
        screening_df[
            "mean_absolute_class_separation"
        ] / max_sep
    ) * 100

else:

    screening_df[
        "normalized_class_separation"
    ] = 0

# Redundancy penalty
screening_df[
    "redundancy_penalty"
] = screening_df[
    "maximum_absolute_correlation"
] * 100

# Final screening score
#
# 40% reliability
# 40% class separation
# 20% redundancy penalty
#
# Again: this is NOT the final model feature importance.

screening_df[
    "preliminary_screening_score"
] = (
    0.40 *
    screening_df[
        "reliability_score"
    ]
    +
    0.40 *
    screening_df[
        "normalized_class_separation"
    ]
    -
    0.20 *
    screening_df[
        "redundancy_penalty"
    ]
)

screening_df = screening_df.sort_values(
    "preliminary_screening_score",
    ascending=False
)

screening_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "18_preliminary_feature_screening_score.csv"
    ),
    index=False
)

print(
    screening_df.to_string(index=False)
)

# ============================================================
# 21. FINAL TEXT REPORT
# ============================================================

print("\n" + "=" * 80)
print("21. GENERATING FINAL REPORT")
print("=" * 80)

report_path = os.path.join(
    OUTPUT_DIR,
    "feature_selection_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "MIT-BIH FEATURE SELECTION ANALYSIS REPORT\n"
    )

    report.write(
        "=" * 70 + "\n\n"
    )

    report.write(
        f"Dataset shape: {df.shape}\n"
    )

    report.write(
        f"Number of candidate features: {len(FEATURES)}\n"
    )

    report.write(
        f"Number of beat classes: {df[TARGET].nunique()}\n\n"
    )

    report.write(
        "CANDIDATE FEATURES\n"
    )

    report.write(
        "-" * 70 + "\n"
    )

    for feature in FEATURES:

        report.write(
            f"- {feature}\n"
        )

    report.write("\n")

    report.write(
        "PRELIMINARY FEATURE DECISIONS\n"
    )

    report.write(
        "-" * 70 + "\n"
    )

    for _, row in decision_df.iterrows():

        report.write(
            f"\n{row['feature']}\n"
        )

        report.write(
            f"  Missingness: "
            f"{row['missing_percent']:.4f}%\n"
        )

        report.write(
            f"  Decision: "
            f"{row['preliminary_decision']}\n"
        )

        report.write(
            f"  Reason: "
            f"{row['reason']}\n"
        )

    report.write("\n")

    report.write(
        "CORE FEATURE SET\n"
    )

    report.write(
        "-" * 70 + "\n"
    )

    for feature in CORE_FEATURES:

        report.write(
            f"- {feature}\n"
        )

    report.write("\n")

    report.write(
        "EXTENDED FEATURE SET\n"
    )

    report.write(
        "-" * 70 + "\n"
    )

    for feature in EXTENDED_FEATURES:

        report.write(
            f"- {feature}\n"
        )

    report.write("\n")

    report.write(
        "HR ALTERNATIVE FEATURE SET\n"
    )

    report.write(
        "-" * 70 + "\n"
    )

    for feature in HR_ALTERNATIVE_FEATURES:

        report.write(
            f"- {feature}\n"
        )

    report.write("\n")

    report.write(
        "IMPORTANT INTERPRETATION\n"
    )

    report.write(
        "-" * 70 + "\n"
    )

    report.write(
        "The preliminary decisions are screening decisions only.\n"
    )

    report.write(
        "They are NOT the final AI feature selection.\n"
    )

    report.write(
        "Final selection should be confirmed using the same "
        "leakage-safe train/test split across the candidate "
        "AI models and feature ablation experiments.\n"
    )

    report.write("\n")

    report.write(
        "RR vs HR\n"
    )

    report.write(
        "-" * 70 + "\n"
    )

    if len(rr_hr_summary) > 0:

        report.write(
            f"RR/HR correlation: "
            f"{rr_hr_corr:.6f}\n"
        )

        report.write(
            "Recommendation: use RR initially and "
            "exclude HR from the primary feature set.\n"
        )

    report.write("\n")

    report.write(
        "CORE vs EXTENDED\n"
    )

    report.write(
        "-" * 70 + "\n"
    )

    for _, row in feature_set_df.iterrows():

        report.write(
            f"\n{row['feature_set']}:\n"
        )

        report.write(
            f"  Number of features: "
            f"{row['number_of_features']}\n"
        )

        report.write(
            f"  Missing data: "
            f"{row['missing_percent']:.4f}%\n"
        )

        report.write(
            f"  Complete rows: "
            f"{row['complete_row_percent']:.4f}%\n"
        )

    report.write("\n")

    report.write(
        "NEXT STEP\n"
    )

    report.write(
        "-" * 70 + "\n"
    )

    report.write(
        "Proceed to leakage-safe dataset preparation and "
        "AI model comparison.\n"
    )

# ============================================================
# 22. FINAL SUMMARY ON SCREEN
# ============================================================

print("\n" + "=" * 80)
print("STEP 4 COMPLETE")
print("=" * 80)

print("\nCandidate features analyzed:")
print(len(FEATURES))

print("\nCore feature set:")
for feature in CORE_FEATURES:
    print("  -", feature)

print("\nExtended feature set:")
for feature in EXTENDED_FEATURES:
    print("  -", feature)

print("\nInitially excluded:")
print("  - Heart_rate_bpm")
print("    Reason: redundant with RR_interval_ms")

print("\nOutput folder:")
print(OUTPUT_DIR)

print("\nImportant output files:")

output_files = [
    "01_feature_basic_statistics.csv",
    "02_feature_missingness.csv",
    "03_missingness_by_beat_class.csv",
    "04_pearson_correlation_matrix.csv",
    "05_spearman_correlation_matrix.csv",
    "06_feature_correlation_pairs.csv",
    "07_RR_vs_HR_redundancy.csv",
    "08_classwise_feature_statistics.csv",
    "09_classwise_feature_separation.csv",
    "10_feature_class_separation_summary.csv",
    "11_classwise_feature_medians.csv",
    "12_suspicious_extreme_value_summary.csv",
    "13_suspicious_extreme_value_rows.csv",
    "14_IQR_outlier_summary.csv",
    "15_feature_distribution_summary.csv",
    "16_feature_set_comparison.csv",
    "17_preliminary_feature_decision.csv",
    "18_preliminary_feature_screening_score.csv",
    "feature_selection_report.txt"
]

for filename in output_files:

    full_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    if os.path.exists(full_path):

        size_kb = (
            os.path.getsize(full_path)
            / 1024
        )

        print(
            f"  ✓ {filename} "
            f"({size_kb:.1f} KB)"
        )

print("\n" + "=" * 80)
print("No changes were made to the master CSV.")
print("=" * 80)