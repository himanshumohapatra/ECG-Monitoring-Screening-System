# ============================================================
# STEP 8A - FIVE-CLASS ECG LABEL FEASIBILITY ANALYSIS
# ============================================================
#
# Purpose:
#   Test whether the MIT-BIH Arrhythmia Database can support
#   five preliminary ECG window-level categories:
#
#       1. Normal heartbeat
#       2. Arrhythmia
#       3. Tachycardia
#       4. Bradycardia
#       5. Other abnormal ECG patterns
#
# IMPORTANT:
#   - This is a LABEL FEASIBILITY study.
#   - No machine-learning model is trained here.
#   - final_test.csv is NOT used.
#   - The master CSV is NOT modified.
#   - The rules below are engineering/feasibility rules,
#     NOT clinical diagnostic criteria.
#
# Input:
#   master_MITBIH_48_records_features.csv
#
# Output folder:
#   step8a_label_feasibility
#
# ============================================================

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# 0. PROJECT PATHS
# ============================================================

PROJECT_DIR = r"C:\Users\Himanshu Mohapatra\OneDrive\Desktop\Project_Main"

INPUT_FILE = os.path.join(
    PROJECT_DIR,
    "master_MITBIH_48_records_features.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "step8a_label_feasibility"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Heart-rate quality rules
# ------------------------------------------------------------
#
# These are deliberately broad validity limits.
# They are NOT the five-class clinical decision thresholds.
# ------------------------------------------------------------

MIN_VALID_HR = 30.0
MAX_VALID_HR = 200.0

MIN_VALID_RR_MS = 300.0
MAX_VALID_RR_MS = 2000.0


# ------------------------------------------------------------
# Rate categories
# ------------------------------------------------------------

BRADYCARDIA_HR = 60.0
TACHYCARDIA_HR = 100.0


# ------------------------------------------------------------
# Beat-level evidence labels
# ------------------------------------------------------------

NORMAL_LABELS = {
    "N"
}

ARRHYTHMIA_LABELS = {
    "A",
    "a",
    "J",
    "S",
    "V",
    "F",
    "f",
    "E",
    "e",
    "j"
}

OTHER_ABNORMAL_LABELS = {
    "L",
    "R",
    "/",
    "Q"
}


# ------------------------------------------------------------
# 20-second window configuration
# ------------------------------------------------------------

WINDOW_SECONDS = 20.0

MIN_BEATS_PER_WINDOW = 5

MIN_VALID_RATE_FRACTION = 0.70

TACHYCARDIA_FRACTION_THRESHOLD = 0.50
BRADYCARDIA_FRACTION_THRESHOLD = 0.50

ARRHYTHMIA_FRACTION_THRESHOLD = 0.20
OTHER_ABNORMAL_FRACTION_THRESHOLD = 0.20

NORMAL_FRACTION_THRESHOLD = 0.80


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):
    print()
    print("=" * 75)
    print(title)
    print("=" * 75)


def safe_percentage(count, denominator):
    if denominator == 0:
        return 0.0
    return 100.0 * count / denominator


def classify_window(row):
    """
    Assign one preliminary mutually-exclusive label to a 20-second
    ECG window.

    Priority:
        1. Tachycardia
        2. Bradycardia
        3. Arrhythmia
        4. Other abnormal ECG
        5. Normal heartbeat
        6. Uncertain / mixed
    """

    valid_rate_fraction = row["valid_rate_fraction"]
    tachy_fraction = row["tachy_fraction"]
    brady_fraction = row["brady_fraction"]
    arrhythmia_fraction = row["arrhythmia_fraction"]
    other_abnormal_fraction = row["other_abnormal_fraction"]
    normal_fraction = row["normal_fraction"]

    median_hr = row["median_hr_bpm"]

    # --------------------------------------------------------
    # Tachycardia
    # --------------------------------------------------------
    if (
        pd.notna(valid_rate_fraction)
        and pd.notna(tachy_fraction)
        and valid_rate_fraction >= MIN_VALID_RATE_FRACTION
        and tachy_fraction >= TACHYCARDIA_FRACTION_THRESHOLD
    ):
        return "Tachycardia"

    # --------------------------------------------------------
    # Bradycardia
    # --------------------------------------------------------
    if (
        pd.notna(valid_rate_fraction)
        and pd.notna(brady_fraction)
        and valid_rate_fraction >= MIN_VALID_RATE_FRACTION
        and brady_fraction >= BRADYCARDIA_FRACTION_THRESHOLD
    ):
        return "Bradycardia"

    # --------------------------------------------------------
    # Arrhythmia
    # --------------------------------------------------------
    if (
        pd.notna(arrhythmia_fraction)
        and arrhythmia_fraction >= ARRHYTHMIA_FRACTION_THRESHOLD
    ):
        return "Arrhythmia"

    # --------------------------------------------------------
    # Other abnormal ECG
    # --------------------------------------------------------
    if (
        pd.notna(other_abnormal_fraction)
        and other_abnormal_fraction >= OTHER_ABNORMAL_FRACTION_THRESHOLD
    ):
        return "Other abnormal ECG"

    # --------------------------------------------------------
    # Normal heartbeat
    # --------------------------------------------------------
    #
    # Median HR must also fall in the normal rate range.
    # --------------------------------------------------------

    if (
        pd.notna(median_hr)
        and pd.notna(valid_rate_fraction)
        and pd.notna(normal_fraction)
        and valid_rate_fraction >= MIN_VALID_RATE_FRACTION
        and normal_fraction >= NORMAL_FRACTION_THRESHOLD
        and BRADYCARDIA_HR <= median_hr <= TACHYCARDIA_HR
        and (
            pd.isna(arrhythmia_fraction)
            or arrhythmia_fraction < ARRHYTHMIA_FRACTION_THRESHOLD
        )
        and (
            pd.isna(other_abnormal_fraction)
            or other_abnormal_fraction < OTHER_ABNORMAL_FRACTION_THRESHOLD
        )
    ):
        return "Normal heartbeat"

    # --------------------------------------------------------
    # Anything that does not satisfy one clear class
    # --------------------------------------------------------

    return "Uncertain / mixed"


# ============================================================
# 2. LOAD DATA
# ============================================================

print_section("STEP 8A - FIVE-CLASS ECG LABEL FEASIBILITY ANALYSIS")

print()
print("Project directory:")
print(PROJECT_DIR)

print()
print("Input:")
print(INPUT_FILE)

if not os.path.exists(INPUT_FILE):
    print()
    print("ERROR: Input file not found.")
    print(INPUT_FILE)
    sys.exit(1)


print()
print("Loading master dataset...")

df = pd.read_csv(INPUT_FILE)

print()
print("Dataset loaded successfully.")
print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns):,}")


# ============================================================
# 3. REQUIRED COLUMN CHECK
# ============================================================

required_columns = [
    "record_id",
    "channel",
    "beat_index",
    "beat_label",
    "R_time_s",
    "RR_interval_ms",
    "Heart_rate_bpm"
]

missing_required = [
    col for col in required_columns
    if col not in df.columns
]

print()
print("Required columns check:", end=" ")

if missing_required:
    print("FAILED")
    print()
    print("Missing columns:")
    for col in missing_required:
        print("  -", col)
    sys.exit(1)

print("PASSED")


# ============================================================
# 4. BASIC CLEANING / TYPE STANDARDIZATION
# ============================================================

# Preserve original beat label as string.
df["beat_label"] = df["beat_label"].astype(str).str.strip()

# Convert numerical fields safely.
df["R_time_s"] = pd.to_numeric(
    df["R_time_s"],
    errors="coerce"
)

df["RR_interval_ms"] = pd.to_numeric(
    df["RR_interval_ms"],
    errors="coerce"
)

df["Heart_rate_bpm"] = pd.to_numeric(
    df["Heart_rate_bpm"],
    errors="coerce"
)


# ============================================================
# 5. ORIGINAL MIT-BIH BEAT LABEL DISTRIBUTION
# ============================================================

print_section(
    "1. ORIGINAL MIT-BIH BEAT LABEL DISTRIBUTION"
)

label_counts = (
    df["beat_label"]
    .value_counts()
    .sort_index()
)

label_distribution = pd.DataFrame({
    "beat_label": label_counts.index,
    "count": label_counts.values
})

label_distribution["percentage"] = (
    100.0
    * label_distribution["count"]
    / len(df)
)

print(
    label_distribution.to_string(index=False)
)

label_distribution.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "01_original_beat_label_distribution.csv"
    ),
    index=False
)


# ============================================================
# 6. HEART-RATE / RR QUALITY CHECK
# ============================================================

print_section(
    "2. HEART-RATE / RR QUALITY CHECK"
)

valid_hr = (
    df["Heart_rate_bpm"].notna()
    & np.isfinite(df["Heart_rate_bpm"])
    & (df["Heart_rate_bpm"] >= MIN_VALID_HR)
    & (df["Heart_rate_bpm"] <= MAX_VALID_HR)
)

valid_rr = (
    df["RR_interval_ms"].notna()
    & np.isfinite(df["RR_interval_ms"])
    & (df["RR_interval_ms"] >= MIN_VALID_RR_MS)
    & (df["RR_interval_ms"] <= MAX_VALID_RR_MS)
)

valid_rate = valid_hr & valid_rr

df["valid_rate"] = valid_rate

valid_count = int(valid_rate.sum())
invalid_count = int((~valid_rate).sum())

print()
print(
    f"Valid HR + RR rows : "
    f"{valid_count:,} "
    f"({safe_percentage(valid_count, len(df)):.3f}%)"
)

print(
    f"Invalid/suspicious rate rows : "
    f"{invalid_count:,} "
    f"({safe_percentage(invalid_count, len(df)):.3f}%)"
)


rate_quality = pd.DataFrame({
    "category": [
        "Valid HR + RR",
        "Invalid/suspicious HR or RR"
    ],
    "count": [
        valid_count,
        invalid_count
    ]
})

rate_quality["percentage"] = (
    100.0
    * rate_quality["count"]
    / len(df)
)

rate_quality.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "rate_quality_summary.csv"
    ),
    index=False
)


# ============================================================
# 7. CREATE TARGET EVIDENCE FLAGS
# ============================================================

# ------------------------------------------------------------
# Normal heartbeat evidence
# ------------------------------------------------------------

df["normal_evidence"] = (
    df["beat_label"].isin(NORMAL_LABELS)
)


# ------------------------------------------------------------
# Arrhythmia evidence
# ------------------------------------------------------------

df["arrhythmia_evidence"] = (
    df["beat_label"].isin(ARRHYTHMIA_LABELS)
)


# ------------------------------------------------------------
# Other abnormal ECG evidence
# ------------------------------------------------------------

df["other_abnormal_evidence"] = (
    df["beat_label"].isin(OTHER_ABNORMAL_LABELS)
)


# ------------------------------------------------------------
# Tachycardia evidence
# ------------------------------------------------------------

df["tachycardia_evidence"] = (
    valid_rate
    & (df["Heart_rate_bpm"] > TACHYCARDIA_HR)
)


# ------------------------------------------------------------
# Bradycardia evidence
# ------------------------------------------------------------

df["bradycardia_evidence"] = (
    valid_rate
    & (df["Heart_rate_bpm"] < BRADYCARDIA_HR)
)


# ============================================================
# 8. FIVE-CLASS BEAT-LEVEL FEASIBILITY
# ============================================================

print_section(
    "3. FIVE-CLASS BEAT-LEVEL FEASIBILITY"
)

beat_level_counts = {
    "Normal heartbeat evidence":
        int(df["normal_evidence"].sum()),

    "Arrhythmia evidence":
        int(df["arrhythmia_evidence"].sum()),

    "Tachycardia rate evidence":
        int(df["tachycardia_evidence"].sum()),

    "Bradycardia rate evidence":
        int(df["bradycardia_evidence"].sum()),

    "Other abnormal ECG evidence":
        int(df["other_abnormal_evidence"].sum())
}

beat_level_feasibility = pd.DataFrame({
    "category": list(beat_level_counts.keys()),
    "count": list(beat_level_counts.values())
})

beat_level_feasibility["percentage_of_all_rows"] = (
    100.0
    * beat_level_feasibility["count"]
    / len(df)
)

print(
    beat_level_feasibility.to_string(index=False)
)

beat_level_feasibility.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "02_beat_level_five_class_feasibility.csv"
    ),
    index=False
)


# ============================================================
# 9. CLASS OVERLAP ANALYSIS
# ============================================================

print_section(
    "4. CLASS OVERLAP ANALYSIS"
)

flag_columns = [
    "normal_evidence",
    "arrhythmia_evidence",
    "tachycardia_evidence",
    "bradycardia_evidence",
    "other_abnormal_evidence"
]

df["number_of_flags"] = (
    df[flag_columns]
    .sum(axis=1)
    .astype(int)
)

overlap_counts = (
    df["number_of_flags"]
    .value_counts()
    .sort_index()
)

overlap_distribution = pd.DataFrame({
    "number_of_flags": overlap_counts.index,
    "count": overlap_counts.values
})

overlap_distribution["percentage"] = (
    100.0
    * overlap_distribution["count"]
    / len(df)
)

print(
    overlap_distribution.to_string(index=False)
)

overlap_distribution.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "03_target_class_overlap_distribution.csv"
    ),
    index=False
)


# ============================================================
# 10. TARGET FLAG COMBINATIONS
# ============================================================

print_section(
    "5. MOST COMMON TARGET-FLAG COMBINATIONS"
)

# Flag order:
#
#   1 = Normal
#   2 = Arrhythmia
#   3 = Tachycardia
#   4 = Bradycardia
#   5 = Other abnormal
#

def make_flag_string(row):
    return "".join(
        "1" if bool(row[col]) else "0"
        for col in flag_columns
    )


df["flag_combination"] = df.apply(
    make_flag_string,
    axis=1
)

flag_combination_counts = (
    df["flag_combination"]
    .value_counts()
)

flag_combination_distribution = pd.DataFrame({
    "flag_combination":
        flag_combination_counts.index,
    "count":
        flag_combination_counts.values
})

flag_combination_distribution["percentage"] = (
    100.0
    * flag_combination_distribution["count"]
    / len(df)
)

print(
    flag_combination_distribution
    .head(20)
    .to_string(index=False)
)

flag_combination_distribution.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "04_target_flag_combinations.csv"
    ),
    index=False
)


# ============================================================
# 11. MIT-BIH LABEL -> TARGET EVIDENCE
# ============================================================

print_section(
    "6. MIT-BIH LABEL → TARGET EVIDENCE"
)

label_to_target = (
    df.groupby("beat_label")
    .agg(
        count=("beat_label", "size"),
        normal_evidence=("normal_evidence", "sum"),
        arrhythmia_evidence=("arrhythmia_evidence", "sum"),
        tachycardia_evidence=("tachycardia_evidence", "sum"),
        bradycardia_evidence=("bradycardia_evidence", "sum"),
        other_abnormal_evidence=("other_abnormal_evidence", "sum")
    )
    .reset_index()
)

for col in [
    "normal_evidence",
    "arrhythmia_evidence",
    "tachycardia_evidence",
    "bradycardia_evidence",
    "other_abnormal_evidence"
]:
    label_to_target[col] = (
        label_to_target[col]
        .astype(int)
    )

print(
    label_to_target.to_string(index=False)
)

label_to_target.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "05_mitbih_label_to_target_evidence.csv"
    ),
    index=False
)


# ============================================================
# 12. CONSTRUCT PRELIMINARY 20-SECOND WINDOWS
# ============================================================

print_section(
    "7. CONSTRUCTING PRELIMINARY 20-SECOND WINDOWS"
)

print()
print(
    f"Rows available for window construction: "
    f"{len(df):,}"
)


# ------------------------------------------------------------
# IMPORTANT:
#
# Windows are created separately for each record + channel.
# This prevents mixing measurements from different channels.
# ------------------------------------------------------------

window_df = df.copy()

# Remove rows without usable time.
window_df = window_df[
    window_df["R_time_s"].notna()
    & np.isfinite(window_df["R_time_s"])
].copy()

# Determine 20-second window number within each
# record/channel combination.
window_df["window_index"] = (
    np.floor(
        window_df["R_time_s"] / WINDOW_SECONDS
    )
    .astype(int)
)

window_df["window_start_s"] = (
    window_df["window_index"]
    * WINDOW_SECONDS
)

window_df["window_end_s"] = (
    window_df["window_start_s"]
    + WINDOW_SECONDS
)


# ============================================================
# 13. WINDOW AGGREGATION
# ============================================================

group_columns = [
    "record_id",
    "channel",
    "window_index",
    "window_start_s",
    "window_end_s"
]

window_rows = []

for group_key, group in window_df.groupby(
    group_columns,
    sort=True
):

    record_id = group_key[0]
    channel = group_key[1]
    window_index = group_key[2]
    window_start_s = group_key[3]
    window_end_s = group_key[4]

    total_beats = len(group)

    if total_beats < MIN_BEATS_PER_WINDOW:
        continue

    valid_rate_count = int(
        group["valid_rate"].sum()
    )

    valid_rate_fraction = (
        valid_rate_count / total_beats
        if total_beats > 0
        else np.nan
    )

    # --------------------------------------------------------
    # Fractions
    # --------------------------------------------------------

    normal_fraction = (
        group["normal_evidence"].mean()
    )

    arrhythmia_fraction = (
        group["arrhythmia_evidence"].mean()
    )

    tachy_fraction = (
        group["tachycardia_evidence"].mean()
    )

    brady_fraction = (
        group["bradycardia_evidence"].mean()
    )

    other_abnormal_fraction = (
        group["other_abnormal_evidence"].mean()
    )

    # --------------------------------------------------------
    # Heart rate statistics
    # --------------------------------------------------------

    valid_hr_values = group.loc[
        group["valid_rate"],
        "Heart_rate_bpm"
    ].dropna()

    if len(valid_hr_values) > 0:
        median_hr_bpm = float(
            valid_hr_values.median()
        )

        mean_hr_bpm = float(
            valid_hr_values.mean()
        )

        min_hr_bpm = float(
            valid_hr_values.min()
        )

        max_hr_bpm = float(
            valid_hr_values.max()
        )

    else:
        median_hr_bpm = np.nan
        mean_hr_bpm = np.nan
        min_hr_bpm = np.nan
        max_hr_bpm = np.nan

    # --------------------------------------------------------
    # RR statistics
    # --------------------------------------------------------

    valid_rr_values = group.loc[
        group["valid_rate"],
        "RR_interval_ms"
    ].dropna()

    if len(valid_rr_values) > 0:

        median_rr_ms = float(
            valid_rr_values.median()
        )

        mean_rr_ms = float(
            valid_rr_values.mean()
        )

    else:

        median_rr_ms = np.nan
        mean_rr_ms = np.nan

    # --------------------------------------------------------
    # Beat-label information
    # --------------------------------------------------------

    label_counts = (
        group["beat_label"]
        .value_counts()
    )

    dominant_label = (
        label_counts.index[0]
        if len(label_counts) > 0
        else ""
    )

    dominant_label_fraction = (
        label_counts.iloc[0] / total_beats
        if len(label_counts) > 0
        else np.nan
    )

    # --------------------------------------------------------
    # Save window information
    # --------------------------------------------------------

    window_rows.append({
        "record_id":
            record_id,

        "channel":
            channel,

        "window_index":
            window_index,

        "window_start_s":
            window_start_s,

        "window_end_s":
            window_end_s,

        "total_beats":
            total_beats,

        "valid_rate_count":
            valid_rate_count,

        "valid_rate_fraction":
            valid_rate_fraction,

        "median_hr_bpm":
            median_hr_bpm,

        "mean_hr_bpm":
            mean_hr_bpm,

        "min_hr_bpm":
            min_hr_bpm,

        "max_hr_bpm":
            max_hr_bpm,

        "median_rr_ms":
            median_rr_ms,

        "mean_rr_ms":
            mean_rr_ms,

        "normal_fraction":
            normal_fraction,

        "arrhythmia_fraction":
            arrhythmia_fraction,

        "tachy_fraction":
            tachy_fraction,

        "brady_fraction":
            brady_fraction,

        "other_abnormal_fraction":
            other_abnormal_fraction,

        "dominant_label":
            dominant_label,

        "dominant_label_fraction":
            dominant_label_fraction
    })


windows = pd.DataFrame(window_rows)

print()
print(
    f"Preliminary 20-second windows: "
    f"{len(windows):,}"
)


# ============================================================
# 14. ASSIGN PRELIMINARY FIVE-CLASS WINDOW LABEL
# ============================================================

if len(windows) > 0:

    windows["preliminary_class"] = (
        windows.apply(
            classify_window,
            axis=1
        )
    )

else:

    windows["preliminary_class"] = []


# ============================================================
# 15. SAVE ALL PRELIMINARY WINDOWS
# ============================================================

windows.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "06_preliminary_20s_windows.csv"
    ),
    index=False
)


# ============================================================
# 16. PRELIMINARY 20-SECOND CLASS DISTRIBUTION
# ============================================================

print_section(
    "8. PRELIMINARY 20-SECOND CLASS DISTRIBUTION"
)

if len(windows) == 0:

    print()
    print("No qualifying windows were constructed.")

    window_class_distribution = pd.DataFrame(
        columns=[
            "preliminary_class",
            "count",
            "percentage"
        ]
    )

else:

    class_counts = (
        windows["preliminary_class"]
        .value_counts()
    )

    window_class_distribution = pd.DataFrame({
        "preliminary_class":
            class_counts.index,
        "count":
            class_counts.values
    })

    window_class_distribution["percentage"] = (
        100.0
        * window_class_distribution["count"]
        / len(windows)
    )

    print(
        window_class_distribution
        .to_string(index=False)
    )


window_class_distribution.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "07_preliminary_20s_class_distribution.csv"
    ),
    index=False
)


# ============================================================
# 17. RECORD-LEVEL CLASS COVERAGE
# ============================================================

print_section(
    "9. RECORD-LEVEL CLASS COVERAGE"
)

if len(windows) == 0:

    record_coverage = pd.DataFrame()

else:

    record_coverage = (
        windows
        .groupby("record_id")[
            "preliminary_class"
        ]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )

    desired_classes = [
        "Normal heartbeat",
        "Arrhythmia",
        "Tachycardia",
        "Bradycardia",
        "Other abnormal ECG",
        "Uncertain / mixed"
    ]

    for class_name in desired_classes:

        if class_name not in record_coverage.columns:

            record_coverage[class_name] = 0

    record_coverage = record_coverage[
        ["record_id"] + desired_classes
    ]


    # --------------------------------------------------------
    # Convert counts into binary availability indicators
    # --------------------------------------------------------

    record_class_presence = record_coverage.copy()

    for class_name in desired_classes:

        record_class_presence[class_name] = (
            record_class_presence[class_name] > 0
        ).astype(int)

    record_class_presence.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "08_record_level_class_coverage_binary.csv"
        ),
        index=False
    )

print()
print(
    "Number of records represented in windows:",
    len(record_coverage)
)

if len(record_coverage) > 0:

    print()
    print(
        record_coverage.to_string(index=False)
    )

record_coverage.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "08_record_level_class_coverage.csv"
    ),
    index=False
)


# ============================================================
# 18. CLASS DISTRIBUTION BY RECORD
# ============================================================

print_section(
    "10. CLASS DISTRIBUTION BY RECORD"
)

if len(windows) == 0:

    class_by_record = pd.DataFrame()

else:

    class_by_record = (
        windows
        .groupby(
            [
                "record_id",
                "preliminary_class"
            ]
        )
        .size()
        .reset_index(
            name="window_count"
        )
    )

    class_by_record["percentage_within_record"] = (
        class_by_record
        .groupby("record_id")["window_count"]
        .transform(
            lambda x: 100.0 * x / x.sum()
        )
    )

    class_by_record = class_by_record.sort_values(
        [
            "record_id",
            "window_count"
        ],
        ascending=[
            True,
            False
        ]
    )

    print(
        class_by_record.to_string(index=False)
    )

class_by_record.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "09_class_distribution_by_record.csv"
    ),
    index=False
)


# ============================================================
# 19. CLASS FEASIBILITY SUMMARY
# ============================================================

print_section(
    "11. FIVE-CLASS FEASIBILITY SUMMARY"
)

desired_classes = [
    "Normal heartbeat",
    "Arrhythmia",
    "Tachycardia",
    "Bradycardia",
    "Other abnormal ECG"
]

if len(windows) == 0:

    summary_rows = []

else:

    summary_rows = []

    for class_name in desired_classes:

        class_windows = windows[
            windows["preliminary_class"]
            == class_name
        ]

        count = len(class_windows)

        record_count = (
            class_windows["record_id"]
            .nunique()
        )

        summary_rows.append({
            "target_class":
                class_name,

            "window_count":
                count,

            "percentage_of_all_windows":
                safe_percentage(
                    count,
                    len(windows)
                ),

            "number_of_records":
                record_count
        })

summary_df = pd.DataFrame(
    summary_rows
)

if len(summary_df) > 0:

    print(
        summary_df.to_string(index=False)
    )

summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "10_five_class_window_feasibility_summary.csv"
    ),
    index=False
)


# ============================================================
# 20. SAVE CONFIGURATION / RULES
# ============================================================

rules_df = pd.DataFrame({
    "parameter": [
        "MIN_VALID_HR",
        "MAX_VALID_HR",
        "MIN_VALID_RR_MS",
        "MAX_VALID_RR_MS",
        "BRADYCARDIA_HR",
        "TACHYCARDIA_HR",
        "WINDOW_SECONDS",
        "MIN_BEATS_PER_WINDOW",
        "MIN_VALID_RATE_FRACTION",
        "TACHYCARDIA_FRACTION_THRESHOLD",
        "BRADYCARDIA_FRACTION_THRESHOLD",
        "ARRHYTHMIA_FRACTION_THRESHOLD",
        "OTHER_ABNORMAL_FRACTION_THRESHOLD",
        "NORMAL_FRACTION_THRESHOLD"
    ],
    "value": [
        MIN_VALID_HR,
        MAX_VALID_HR,
        MIN_VALID_RR_MS,
        MAX_VALID_RR_MS,
        BRADYCARDIA_HR,
        TACHYCARDIA_HR,
        WINDOW_SECONDS,
        MIN_BEATS_PER_WINDOW,
        MIN_VALID_RATE_FRACTION,
        TACHYCARDIA_FRACTION_THRESHOLD,
        BRADYCARDIA_FRACTION_THRESHOLD,
        ARRHYTHMIA_FRACTION_THRESHOLD,
        OTHER_ABNORMAL_FRACTION_THRESHOLD,
        NORMAL_FRACTION_THRESHOLD
    ]
})

rules_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "11_step8a_labeling_rules.csv"
    ),
    index=False
)


# ============================================================
# 21. FINAL REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "step8a_label_feasibility_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "STEP 8A - FIVE-CLASS ECG LABEL FEASIBILITY ANALYSIS\n"
    )

    report.write("=" * 75 + "\n\n")

    report.write(
        "IMPORTANT: This is an engineering feasibility analysis.\n"
    )

    report.write(
        "The preliminary labels are NOT clinical diagnoses.\n\n"
    )

    report.write(
        f"Input rows: {len(df):,}\n"
    )

    report.write(
        f"Input columns: {len(df.columns):,}\n\n"
    )

    report.write(
        "RATE QUALITY\n"
    )

    report.write(
        f"Valid HR + RR rows: {valid_count:,} "
        f"({safe_percentage(valid_count, len(df)):.3f}%)\n"
    )

    report.write(
        f"Invalid/suspicious rate rows: {invalid_count:,} "
        f"({safe_percentage(invalid_count, len(df)):.3f}%)\n\n"
    )

    report.write(
        "BEAT-LEVEL EVIDENCE\n"
    )

    for category, count in beat_level_counts.items():

        report.write(
            f"{category}: {count:,} "
            f"({safe_percentage(count, len(df)):.3f}%)\n"
        )

    report.write("\n")

    report.write(
        "20-SECOND WINDOW CONSTRUCTION\n"
    )

    report.write(
        f"Total preliminary windows: "
        f"{len(windows):,}\n"
    )

    report.write(
        f"Minimum beats/window: "
        f"{MIN_BEATS_PER_WINDOW}\n"
    )

    report.write(
        f"Window duration: "
        f"{WINDOW_SECONDS:.1f} seconds\n\n"
    )

    report.write(
        "PRELIMINARY WINDOW CLASS DISTRIBUTION\n"
    )

    if len(window_class_distribution) > 0:

        for _, row in window_class_distribution.iterrows():

            report.write(
                f"{row['preliminary_class']}: "
                f"{int(row['count']):,} "
                f"({row['percentage']:.3f}%)\n"
            )

    else:

        report.write(
            "No qualifying windows.\n"
        )

    report.write("\n")

    report.write(
        "LABELING RULE PRIORITY\n"
    )

    report.write(
        "1. Tachycardia\n"
    )

    report.write(
        "2. Bradycardia\n"
    )

    report.write(
        "3. Arrhythmia\n"
    )

    report.write(
        "4. Other abnormal ECG\n"
    )

    report.write(
        "5. Normal heartbeat\n"
    )

    report.write(
        "6. Uncertain / mixed\n\n"
    )

    report.write(
        "NOTE:\n"
    )

    report.write(
        "The five-class rules are preliminary engineering rules "
        "for dataset feasibility. They should not be described "
        "as medically validated diagnostic criteria.\n"
    )


# ============================================================
# 22. FINAL CONSOLE SUMMARY
# ============================================================

print_section(
    "STEP 8A COMPLETED SUCCESSFULLY"
)

print()
print("Output folder:")
print(OUTPUT_DIR)

print()
print("Generated files:")

output_files = sorted(
    os.listdir(OUTPUT_DIR)
)

for filename in output_files:

    print(
        "  -",
        filename
    )

print()
print("Master CSV was NOT modified.")

print("final_test.csv was NOT loaded or modified.")

print()
print("No machine-learning model was trained.")

print()
print(
    "The next step is to inspect the 20-second class "
    "distribution and record-level coverage before "
    "creating the final five-class training dataset."
)

print()
print("=" * 75)