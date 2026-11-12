# ============================================================
# STEP 3 - EXTREME VALUE INVESTIGATION
# MIT-BIH Arrhythmia Database
#
# Purpose:
#   Investigate extreme / suspicious ECG feature values
#   before preparing the dataset for AI/ML.
#
# IMPORTANT:
#   This script DOES NOT delete or modify the master dataset.
# ============================================================

import os
import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATH
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(
    PROJECT_DIR,
    "master_MITBIH_48_records_features.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "extreme_value_investigation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 3 - EXTREME VALUE INVESTIGATION")
print("=" * 70)

print("\nProject folder:")
print(PROJECT_DIR)

print("\nInput file:")
print(INPUT_FILE)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"\nERROR: Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully.")
print("Shape:", df.shape)


# ============================================================
# 3. FEATURE LIST
# ============================================================

features = [
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

available_features = [
    col for col in features
    if col in df.columns
]

print("\nFeatures being investigated:")
for col in available_features:
    print("  -", col)


# ============================================================
# 4. BASIC MIN / MAX / QUANTILES
# ============================================================

print("\n" + "=" * 70)
print("4. BASIC DISTRIBUTION STATISTICS")
print("=" * 70)

quantiles = [
    0.001,
    0.01,
    0.05,
    0.25,
    0.50,
    0.75,
    0.95,
    0.99,
    0.999
]

distribution_rows = []

for feature in available_features:

    values = pd.to_numeric(
        df[feature],
        errors="coerce"
    ).dropna()

    if len(values) == 0:
        continue

    row = {
        "feature": feature,
        "count": len(values),
        "min": values.min(),
        "max": values.max(),
        "mean": values.mean(),
        "std": values.std(),
    }

    for q in quantiles:
        row[f"q{q}"] = values.quantile(q)

    distribution_rows.append(row)


distribution_df = pd.DataFrame(distribution_rows)

print(distribution_df.to_string(index=False))

distribution_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "01_feature_distribution_statistics.csv"
    ),
    index=False
)


# ============================================================
# 5. DOMAIN-BASED SCREENING LIMITS
#
# These are ONLY screening thresholds.
# They are NOT automatic deletion rules.
# ============================================================

screening_limits = {

    # Heart timing
    "RR_interval_ms": {
        "low": 250,
        "high": 3000
    },

    "Heart_rate_bpm": {
        "low": 20,
        "high": 220
    },

    # Wave durations
    "P_duration_ms": {
        "low": 30,
        "high": 250
    },

    "QRS_duration_ms": {
        "low": 40,
        "high": 250
    },

    "PR_interval_ms": {
        "low": 80,
        "high": 350
    },

    "QT_interval_ms": {
        "low": 200,
        "high": 700
    },

    "T_duration_ms": {
        "low": 40,
        "high": 500
    }
}


# ============================================================
# 6. FLAG VALUES OUTSIDE SCREENING LIMITS
# ============================================================

print("\n" + "=" * 70)
print("6. DOMAIN-BASED EXTREME VALUE SCREENING")
print("=" * 70)

screening_summary = []
screened_rows = []

for feature, limits in screening_limits.items():

    if feature not in df.columns:
        continue

    series = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

    low = limits["low"]
    high = limits["high"]

    low_mask = series < low
    high_mask = series > high
    extreme_mask = low_mask | high_mask

    valid_count = series.notna().sum()

    low_count = low_mask.sum()
    high_count = high_mask.sum()
    total_count = extreme_mask.sum()

    print(f"\n{feature}")
    print(f"  Lower than {low}: {low_count}")
    print(f"  Higher than {high}: {high_count}")
    print(f"  Total flagged: {total_count}")

    screening_summary.append({
        "feature": feature,
        "lower_limit": low,
        "upper_limit": high,
        "below_lower_limit": low_count,
        "above_upper_limit": high_count,
        "total_flagged": total_count,
        "valid_values": valid_count,
        "percent_flagged": (
            100 * total_count / valid_count
            if valid_count > 0 else np.nan
        )
    })

    if total_count > 0:

        temp = df.loc[
            extreme_mask,
            [
                "record_id",
                "channel",
                "beat_index",
                "beat_label",
                feature
            ]
        ].copy()

        temp["feature"] = feature
        temp["reason"] = np.where(
            temp[feature] < low,
            "below_lower_limit",
            "above_upper_limit"
        )

        screened_rows.append(temp)


screening_summary_df = pd.DataFrame(screening_summary)

screening_summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "02_domain_screening_summary.csv"
    ),
    index=False
)

if screened_rows:
    screened_values_df = pd.concat(
        screened_rows,
        ignore_index=True
    )

    screened_values_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "03_domain_screened_rows.csv"
        ),
        index=False
    )
else:
    screened_values_df = pd.DataFrame()


# ============================================================
# 7. ROBUST STATISTICAL OUTLIERS USING IQR
#
# Outlier rule:
#   below Q1 - 1.5*IQR
#   above Q3 + 1.5*IQR
#
# Again, this does NOT mean the values are wrong.
# ============================================================

print("\n" + "=" * 70)
print("7. STATISTICAL OUTLIER SCREENING - IQR")
print("=" * 70)

iqr_summary = []
iqr_rows = []

for feature in available_features:

    series = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

    valid = series.dropna()

    if len(valid) == 0:
        continue

    q1 = valid.quantile(0.25)
    q3 = valid.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    mask = (
        (series < lower_bound) |
        (series > upper_bound)
    )

    count = mask.sum()

    print(f"\n{feature}")
    print(f"  Q1: {q1:.4f}")
    print(f"  Q3: {q3:.4f}")
    print(f"  IQR: {iqr:.4f}")
    print(f"  Lower bound: {lower_bound:.4f}")
    print(f"  Upper bound: {upper_bound:.4f}")
    print(f"  IQR outliers: {count}")

    iqr_summary.append({
        "feature": feature,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": count,
        "percent_outlier": (
            100 * count / len(valid)
        )
    })

    if count > 0:

        temp = df.loc[
            mask,
            [
                "record_id",
                "channel",
                "beat_index",
                "beat_label",
                feature
            ]
        ].copy()

        temp["feature"] = feature
        temp["IQR_lower_bound"] = lower_bound
        temp["IQR_upper_bound"] = upper_bound

        iqr_rows.append(temp)


iqr_summary_df = pd.DataFrame(iqr_summary)

iqr_summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "04_IQR_outlier_summary.csv"
    ),
    index=False
)

if iqr_rows:

    iqr_values_df = pd.concat(
        iqr_rows,
        ignore_index=True
    )

    iqr_values_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "05_IQR_outlier_rows.csv"
        ),
        index=False
    )
else:
    iqr_values_df = pd.DataFrame()


# ============================================================
# 8. EXTREME VALUES BY FEATURE
# ============================================================

print("\n" + "=" * 70)
print("8. MOST EXTREME OBSERVATIONS")
print("=" * 70)

extreme_rows = []

for feature in available_features:

    series = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

    valid_indices = series.dropna().index

    if len(valid_indices) == 0:
        continue

    # Lowest 10
    lowest_idx = (
        series.loc[valid_indices]
        .nsmallest(10)
        .index
    )

    # Highest 10
    highest_idx = (
        series.loc[valid_indices]
        .nlargest(10)
        .index
    )

    selected_idx = list(
        dict.fromkeys(
            list(lowest_idx) +
            list(highest_idx)
        )
    )

    temp = df.loc[
        selected_idx,
        [
            "record_id",
            "channel",
            "channel_index",
            "beat_index",
            "beat_label",
            "R_sample",
            "R_time_s",
            feature
        ]
    ].copy()

    temp["feature"] = feature

    extreme_rows.append(temp)


if extreme_rows:

    extreme_values_df = pd.concat(
        extreme_rows,
        ignore_index=True
    )

    extreme_values_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "06_most_extreme_values.csv"
        ),
        index=False
    )

    print(
        extreme_values_df.to_string(index=False)
    )


# ============================================================
# 9. RR / HEART RATE CONSISTENCY
#
# HR should approximately equal:
#
#     HR = 60000 / RR_ms
#
# This checks whether the two extracted features agree.
# ============================================================

print("\n" + "=" * 70)
print("9. RR / HEART RATE CONSISTENCY")
print("=" * 70)

rr_hr = df[
    [
        "record_id",
        "channel",
        "beat_index",
        "beat_label",
        "RR_interval_ms",
        "Heart_rate_bpm"
    ]
].copy()

rr_hr["calculated_HR_bpm"] = (
    60000 /
    rr_hr["RR_interval_ms"]
)

rr_hr["HR_difference_bpm"] = (
    rr_hr["Heart_rate_bpm"] -
    rr_hr["calculated_HR_bpm"]
)

rr_hr["HR_relative_error_percent"] = (
    abs(rr_hr["HR_difference_bpm"]) /
    rr_hr["calculated_HR_bpm"].abs()
) * 100

rr_hr_valid = rr_hr[
    np.isfinite(
        rr_hr["HR_relative_error_percent"]
    )
].copy()

print(
    "\nMaximum HR relative error:",
    rr_hr_valid["HR_relative_error_percent"].max()
)

print(
    "Mean HR relative error:",
    rr_hr_valid["HR_relative_error_percent"].mean()
)

print(
    "Rows with HR relative error > 1%:",
    (
        rr_hr_valid["HR_relative_error_percent"] > 1
    ).sum()
)

rr_hr.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "07_RR_HR_consistency.csv"
    ),
    index=False
)


# ============================================================
# 10. DURATION LOGIC CHECK
#
# Check:
#   duration = offset - onset
#
# Sampling rate of MIT-BIH = 360 Hz
#
# 1 sample = 1000 / 360 ms
# ============================================================

print("\n" + "=" * 70)
print("10. DURATION CONSISTENCY CHECK")
print("=" * 70)

FS = 360.0
MS_PER_SAMPLE = 1000.0 / FS

duration_pairs = [
    (
        "P_duration_ms",
        "P_onset_sample",
        "P_offset_sample"
    ),
    (
        "QRS_duration_ms",
        "QRS_onset_sample",
        "QRS_offset_sample"
    ),
    (
        "T_duration_ms",
        "T_onset_sample",
        "T_offset_sample"
    )
]

duration_results = []

for duration_col, onset_col, offset_col in duration_pairs:

    if not all(
        col in df.columns
        for col in [
            duration_col,
            onset_col,
            offset_col
        ]
    ):
        continue

    calculated_duration = (
        df[offset_col] -
        df[onset_col]
    ) * MS_PER_SAMPLE

    difference = (
        df[duration_col] -
        calculated_duration
    )

    valid = difference.dropna()

    print(f"\n{duration_col}")

    if len(valid) > 0:

        print(
            "  Mean absolute difference:",
            valid.abs().mean()
        )

        print(
            "  Maximum absolute difference:",
            valid.abs().max()
        )

        print(
            "  Difference > 1 ms:",
            (valid.abs() > 1).sum()
        )

        duration_results.append({
            "duration_feature": duration_col,
            "valid_rows": len(valid),
            "mean_abs_difference_ms": valid.abs().mean(),
            "max_abs_difference_ms": valid.abs().max(),
            "difference_over_1ms": (
                valid.abs() > 1
            ).sum()
        })


duration_results_df = pd.DataFrame(
    duration_results
)

duration_results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "08_duration_consistency_summary.csv"
    ),
    index=False
)


# ============================================================
# 11. WAVE ORDERING CHECK
#
# Expected approximate order:
#
# P onset
#   <= P peak
#   <= P offset
#   <= QRS onset
#   <= QRS offset
#   <= T onset
#   <= T peak
#   <= T offset
#
# Missing values are allowed.
# Violations are only FLAGGED.
# ============================================================

print("\n" + "=" * 70)
print("11. ECG WAVE ORDERING CHECK")
print("=" * 70)

ordering_columns = [
    "P_onset_sample",
    "P_peak_sample",
    "P_offset_sample",
    "QRS_onset_sample",
    "QRS_offset_sample",
    "T_onset_sample",
    "T_peak_sample",
    "T_offset_sample"
]

available_ordering_columns = [
    col for col in ordering_columns
    if col in df.columns
]

ordering_df = df[
    [
        "record_id",
        "channel",
        "beat_index",
        "beat_label"
    ] + available_ordering_columns
].copy()


def check_order(row):

    values = []

    for col in available_ordering_columns:

        value = row[col]

        if pd.notna(value):
            values.append(
                (col, value)
            )

    violations = []

    for i in range(len(values) - 1):

        name1, value1 = values[i]
        name2, value2 = values[i + 1]

        if value1 > value2:

            violations.append(
                f"{name1}>{name2}"
            )

    return ";".join(violations)


ordering_df["ordering_violation"] = (
    ordering_df.apply(
        check_order,
        axis=1
    )
)

ordering_violations = ordering_df[
    ordering_df["ordering_violation"] != ""
].copy()

print(
    "\nRows with wave-ordering violations:",
    len(ordering_violations)
)

if len(ordering_violations) > 0:

    print("\nFirst 20 violations:")

    print(
        ordering_violations.head(20).to_string(
            index=False
        )
    )

ordering_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "09_wave_ordering_check.csv"
    ),
    index=False
)


# ============================================================
# 12. EXTREME VALUES BY BEAT LABEL
# ============================================================

print("\n" + "=" * 70)
print("12. EXTREME VALUE COUNTS BY BEAT LABEL")
print("=" * 70)

label_summary_rows = []

for feature, limits in screening_limits.items():

    if feature not in df.columns:
        continue

    series = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

    flag = (
        (series < limits["low"]) |
        (series > limits["high"])
    )

    temp = df[
        [
            "beat_label",
            "record_id",
            "channel"
        ]
    ].copy()

    temp["flag"] = flag

    grouped = (
        temp.groupby("beat_label")
        .agg(
            total_beats=("flag", "size"),
            extreme_count=("flag", "sum")
        )
        .reset_index()
    )

    grouped["feature"] = feature

    grouped["extreme_percent"] = (
        100 *
        grouped["extreme_count"] /
        grouped["total_beats"]
    )

    label_summary_rows.append(
        grouped
    )


if label_summary_rows:

    label_extreme_df = pd.concat(
        label_summary_rows,
        ignore_index=True
    )

    label_extreme_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "10_extreme_values_by_beat_label.csv"
        ),
        index=False
    )

    print(
        label_extreme_df.to_string(index=False)
    )


# ============================================================
# 13. EXTREME VALUES BY RECORD
# ============================================================

print("\n" + "=" * 70)
print("13. EXTREME VALUE COUNTS BY RECORD")
print("=" * 70)

record_summary_rows = []

for feature, limits in screening_limits.items():

    if feature not in df.columns:
        continue

    series = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

    flag = (
        (series < limits["low"]) |
        (series > limits["high"])
    )

    temp = df[
        [
            "record_id",
            "channel"
        ]
    ].copy()

    temp["flag"] = flag

    grouped = (
        temp.groupby("record_id")
        .agg(
            total_beats=("flag", "size"),
            extreme_count=("flag", "sum")
        )
        .reset_index()
    )

    grouped["feature"] = feature

    grouped["extreme_percent"] = (
        100 *
        grouped["extreme_count"] /
        grouped["total_beats"]
    )

    record_summary_rows.append(
        grouped
    )


if record_summary_rows:

    record_extreme_df = pd.concat(
        record_summary_rows,
        ignore_index=True
    )

    record_extreme_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "11_extreme_values_by_record.csv"
        ),
        index=False
    )


# ============================================================
# 14. EXTREME VALUES BY CHANNEL
# ============================================================

print("\n" + "=" * 70)
print("14. EXTREME VALUE COUNTS BY CHANNEL")
print("=" * 70)

channel_summary_rows = []

for feature, limits in screening_limits.items():

    if feature not in df.columns:
        continue

    series = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

    flag = (
        (series < limits["low"]) |
        (series > limits["high"])
    )

    temp = df[
        [
            "channel"
        ]
    ].copy()

    temp["flag"] = flag

    grouped = (
        temp.groupby("channel")
        .agg(
            total_beats=("flag", "size"),
            extreme_count=("flag", "sum")
        )
        .reset_index()
    )

    grouped["feature"] = feature

    grouped["extreme_percent"] = (
        100 *
        grouped["extreme_count"] /
        grouped["total_beats"]
    )

    channel_summary_rows.append(
        grouped
    )


if channel_summary_rows:

    channel_extreme_df = pd.concat(
        channel_summary_rows,
        ignore_index=True
    )

    channel_extreme_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "12_extreme_values_by_channel.csv"
        ),
        index=False
    )


# ============================================================
# 15. MOST EXTREME ROWS - COMBINED VIEW
# ============================================================

print("\n" + "=" * 70)
print("15. MOST IMPORTANT EXTREME ROWS")
print("=" * 70)

important_columns = [
    "record_id",
    "channel",
    "beat_index",
    "beat_label",
    "RR_interval_ms",
    "Heart_rate_bpm",
    "P_duration_ms",
    "QRS_duration_ms",
    "PR_interval_ms",
    "QT_interval_ms",
    "T_duration_ms",
    "R_amplitude_mV",
    "P_amplitude_mV",
    "T_amplitude_mV"
]

important_columns = [
    col for col in important_columns
    if col in df.columns
]

# Print rows with highest values for major timing features

for feature in [
    "RR_interval_ms",
    "QRS_duration_ms",
    "PR_interval_ms",
    "QT_interval_ms",
    "T_duration_ms"
]:

    if feature not in df.columns:
        continue

    print(f"\nTop 10 highest {feature}:")

    top_rows = (
        df[
            important_columns
        ]
        .sort_values(
            feature,
            ascending=False
        )
        .head(10)
    )

    print(
        top_rows.to_string(
            index=False
        )
    )


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("16. FINAL SUMMARY")
print("=" * 70)

print("\nDataset shape:")
print(df.shape)

print("\nTotal rows:")
print(len(df))

print("\nFeatures investigated:")
print(len(available_features))

print("\nDomain screening results:")

if len(screening_summary_df) > 0:

    print(
        screening_summary_df[
            [
                "feature",
                "total_flagged",
                "percent_flagged"
            ]
        ].to_string(
            index=False
        )
    )

print("\nWave-ordering violations:")
print(len(ordering_violations))

print("\nOutput folder:")
print(OUTPUT_DIR)

print("\nFiles created:")

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
            f"  {filename:<55}"
            f"{size_kb:>10.1f} KB"
        )


# ============================================================
# END
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 EXTREME VALUE INVESTIGATION COMPLETED")
print("=" * 70)

print(
    "\nIMPORTANT:"
    "\nNo rows were deleted."
    "\nThe original master CSV was not modified."
    "\nAll suspicious values are only FLAGGED for investigation."
)

print("\n")