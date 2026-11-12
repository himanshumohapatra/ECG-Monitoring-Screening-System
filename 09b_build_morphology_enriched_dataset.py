
# ============================================================
# STEP 9B
# Build Morphology-Enriched Five-Class ECG Window Dataset
#
# Important:
# - Preserves existing Step 8B labels and CV folds
# - Uses only the existing clean/development windows
# - Does NOT load or modify the final test dataset
# - Does NOT use beat labels as model features
# ============================================================

from pathlib import Path
import sys
import pandas as pd
import numpy as np


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

PROJECT_DIR = Path(
    r"C:\Users\Himanshu Mohapatra\OneDrive\Desktop\Project_Main"
)

MASTER_FILE = PROJECT_DIR / "master_MITBIH_48_records_features.csv"

STEP8B_DIR = PROJECT_DIR / "step8b_five_class_dataset"

CLEAN_WINDOWS_FILE = STEP8B_DIR / "five_class_windows_clean.csv"

DEVELOPMENT_FILE = (
    STEP8B_DIR / "five_class_development_with_cv_folds.csv"
)

OUTPUT_DIR = PROJECT_DIR / "step9b_morphology_features"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. SETTINGS
# ------------------------------------------------------------

WINDOW_SECONDS = 20.0

WINDOW_KEYS = [
    "record_id",
    "channel",
    "window_index",
]

TARGET_COLUMN = "preliminary_class"

# These are physiological beat-level features.
# Beat labels and class-derived fractions are deliberately
# excluded from the model feature aggregation.
PHYSIOLOGICAL_FEATURES = [
    "RR_interval_ms",
    "Heart_rate_bpm",
    "R_amplitude_mV",
    "P_amplitude_mV",
    "P_duration_ms",
    "QRS_duration_ms",
    "PR_interval_ms",
    "QT_interval_ms",
    "T_amplitude_mV",
    "T_duration_ms",
]

# Statistics to calculate for each feature in each window.
AGGREGATION_STATISTICS = [
    "median",
    "mean",
    "std",
    "min",
    "max",
]

# For selected features, use a smaller set of statistics
# to avoid creating too many redundant features.
STATISTICS_BY_FEATURE = {
    "RR_interval_ms": [
        "median", "mean", "std", "min", "max"
    ],
    "Heart_rate_bpm": [
        "median", "mean", "std", "min", "max"
    ],
    "R_amplitude_mV": [
        "median", "mean", "std"
    ],
    "P_amplitude_mV": [
        "median", "mean", "std"
    ],
    "P_duration_ms": [
        "median", "mean"
    ],
    "QRS_duration_ms": [
        "median", "mean"
    ],
    "PR_interval_ms": [
        "median", "mean"
    ],
    "QT_interval_ms": [
        "median", "mean"
    ],
    "T_amplitude_mV": [
        "median", "mean"
    ],
    "T_duration_ms": [
        "median", "mean"
    ],
}


# ------------------------------------------------------------
# 3. HELPER FUNCTIONS
# ------------------------------------------------------------

def stop_with_error(message):
    print("\nERROR:", message)
    sys.exit(1)


def check_file_exists(file_path):
    if not file_path.exists():
        stop_with_error(
            f"Required file not found:\n{file_path}"
        )


def normalize_key_columns(df):
    """
    Normalize key columns to avoid mismatches caused by
    different data types in CSV files.
    """

    df = df.copy()

    df["record_id"] = (
        df["record_id"]
        .astype(str)
        .str.strip()
    )

    df["channel"] = (
        df["channel"]
        .astype(str)
        .str.strip()
    )

    df["window_index"] = pd.to_numeric(
        df["window_index"],
        errors="coerce"
    )

    if df["window_index"].isna().any():
        stop_with_error(
            "Some window_index values could not be converted "
            "to numeric."
        )

    df["window_index"] = (
        df["window_index"].astype(int)
    )

    return df


def safe_numeric_conversion(df, columns):
    """
    Convert feature columns to numeric.
    Invalid values become NaN and are handled later
    by the model's imputation pipeline.
    """

    df = df.copy()

    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


# ------------------------------------------------------------
# 4. CHECK INPUT FILES
# ------------------------------------------------------------

print("=" * 70)
print("STEP 9B: MORPHOLOGY-ENRICHED DATASET CONSTRUCTION")
print("=" * 70)

check_file_exists(MASTER_FILE)
check_file_exists(CLEAN_WINDOWS_FILE)
check_file_exists(DEVELOPMENT_FILE)

print("\nAll required input files found.")


# ------------------------------------------------------------
# 5. LOAD DATA
# ------------------------------------------------------------

print("\nLoading master beat-feature dataset...")

master = pd.read_csv(MASTER_FILE)

print("Master dataset shape:", master.shape)

print("\nLoading clean five-class windows...")

clean_windows = pd.read_csv(CLEAN_WINDOWS_FILE)

print("Clean windows shape:", clean_windows.shape)

print("\nLoading development windows with CV folds...")

development = pd.read_csv(DEVELOPMENT_FILE)

print("Development dataset shape:", development.shape)


# ------------------------------------------------------------
# 6. CHECK REQUIRED COLUMNS
# ------------------------------------------------------------

required_master_columns = [
    "record_id",
    "channel",
    "R_time_s",
]

required_window_columns = (
    WINDOW_KEYS + [TARGET_COLUMN]
)

for column in required_master_columns:
    if column not in master.columns:
        stop_with_error(
            f"Master CSV is missing required column: {column}"
        )

for column in WINDOW_KEYS:
    if column not in clean_windows.columns:
        stop_with_error(
            f"Clean windows CSV is missing key column: {column}"
        )

for column in required_window_columns:
    if column not in development.columns:
        stop_with_error(
            f"Development CSV is missing required column: {column}"
        )

missing_features = [
    feature
    for feature in PHYSIOLOGICAL_FEATURES
    if feature not in master.columns
]

if missing_features:
    stop_with_error(
        "The following physiological features were not found "
        "in the master CSV:\n"
        + "\n".join(missing_features)
    )

print("\nRequired columns check: PASSED")


# ------------------------------------------------------------
# 7. NORMALIZE KEY COLUMNS
# ------------------------------------------------------------

master = master.copy()

master["record_id"] = (
    master["record_id"]
    .astype(str)
    .str.strip()
)

master["channel"] = (
    master["channel"]
    .astype(str)
    .str.strip()
)

clean_windows = normalize_key_columns(
    clean_windows
)

development = normalize_key_columns(
    development
)


# ------------------------------------------------------------
# 8. CHECK DUPLICATE WINDOW KEYS
# ------------------------------------------------------------

print("\nChecking duplicate window keys...")

if clean_windows.duplicated(WINDOW_KEYS).any():
    duplicates = clean_windows.loc[
        clean_windows.duplicated(WINDOW_KEYS, keep=False),
        WINDOW_KEYS
    ]

    print(duplicates.head(20))

    stop_with_error(
        "Duplicate keys found in clean windows dataset."
    )

if development.duplicated(WINDOW_KEYS).any():
    stop_with_error(
        "Duplicate keys found in development dataset."
    )

print("Clean windows duplicate check: PASSED")
print("Development duplicate check: PASSED")


# ------------------------------------------------------------
# 9. PREPARE BEAT-LEVEL FEATURES
# ------------------------------------------------------------

print("\nPreparing beat-level physiological features...")

master = safe_numeric_conversion(
    master,
    ["R_time_s"] + PHYSIOLOGICAL_FEATURES
)

# Remove rows with missing R-peak time because they cannot
# be assigned to a time window.
master = master.dropna(
    subset=["R_time_s"]
).copy()

# Use the same 20-second binning convention:
# window_index = floor(R_time_s / 20)
master["window_index"] = np.floor(
    master["R_time_s"] / WINDOW_SECONDS
).astype(int)

print(
    "Beat rows available for window aggregation:",
    len(master)
)


# ------------------------------------------------------------
# 10. FILTER TO WINDOWS IN THE CLEAN STEP 8B DATASET
# ------------------------------------------------------------

print("\nMatching beats to existing Step 8B clean windows...")

# Keep only record/channel/window combinations present in
# the approved Step 8B clean window dataset.
valid_window_keys = clean_windows[
    WINDOW_KEYS
].drop_duplicates()

beats_for_clean_windows = master.merge(
    valid_window_keys,
    on=WINDOW_KEYS,
    how="inner",
    validate="many_to_one"
)

print(
    "Matched beat rows:",
    len(beats_for_clean_windows)
)

print(
    "Unique clean windows:",
    len(valid_window_keys)
)


# ------------------------------------------------------------
# 11. AGGREGATE PHYSIOLOGICAL FEATURES
# ------------------------------------------------------------

print("\nAggregating physiological features by window...")

aggregation_dictionary = {}

for feature in PHYSIOLOGICAL_FEATURES:

    statistics = STATISTICS_BY_FEATURE.get(
        feature,
        AGGREGATION_STATISTICS
    )

    aggregation_dictionary[feature] = statistics


aggregated = (
    beats_for_clean_windows
    .groupby(WINDOW_KEYS)
    .agg(aggregation_dictionary)
)

# Flatten multi-level column names.
aggregated.columns = [
    f"{feature}_{statistic}"
    for feature, statistic in aggregated.columns
]

aggregated = aggregated.reset_index()


# ------------------------------------------------------------
# 12. ADD FEATURE AVAILABILITY INFORMATION
# ------------------------------------------------------------

print("\nCalculating feature availability information...")

availability = (
    beats_for_clean_windows
    .groupby(WINDOW_KEYS)[PHYSIOLOGICAL_FEATURES]
    .agg(["count"])
)

availability.columns = [
    f"{feature}_valid_count"
    for feature, statistic in availability.columns
]

availability = availability.reset_index()


# Count all beat rows available in each window.
beat_counts = (
    beats_for_clean_windows
    .groupby(WINDOW_KEYS)
    .size()
    .reset_index(name="aggregated_beat_count")
)

# Combine aggregates, availability and beat counts.
morphology_features = aggregated.merge(
    availability,
    on=WINDOW_KEYS,
    how="left",
    validate="one_to_one"
)

morphology_features = morphology_features.merge(
    beat_counts,
    on=WINDOW_KEYS,
    how="left",
    validate="one_to_one"
)


# Calculate valid measurement fractions.
for feature in PHYSIOLOGICAL_FEATURES:

    count_column = f"{feature}_valid_count"

    fraction_column = (
        f"{feature}_valid_fraction"
    )

    morphology_features[fraction_column] = (
        morphology_features[count_column]
        /
        morphology_features["aggregated_beat_count"]
    )


# ------------------------------------------------------------
# 13. SAVE WINDOW-LEVEL MORPHOLOGY FEATURES
# ------------------------------------------------------------

morphology_output = (
    OUTPUT_DIR
    / "step9b_morphology_features_all_clean_windows.csv"
)

morphology_features.to_csv(
    morphology_output,
    index=False
)

print(
    "\nSaved morphology features for clean windows:"
)
print(morphology_output)


# ------------------------------------------------------------
# 14. MERGE WITH ORIGINAL CLEAN WINDOWS
# ------------------------------------------------------------

print("\nMerging morphology features with clean windows...")

clean_enriched = clean_windows.merge(
    morphology_features,
    on=WINDOW_KEYS,
    how="left",
    validate="one_to_one"
)

clean_enriched_output = (
    OUTPUT_DIR
    / "five_class_windows_morphology_enriched.csv"
)

clean_enriched.to_csv(
    clean_enriched_output,
    index=False
)

print(
    "Saved enriched clean windows:"
)
print(clean_enriched_output)


# ------------------------------------------------------------
# 15. MERGE WITH DEVELOPMENT WINDOWS AND PRESERVE CV FOLDS
# ------------------------------------------------------------

print(
    "\nMerging morphology features with development windows..."
)

development_enriched = development.merge(
    morphology_features,
    on=WINDOW_KEYS,
    how="left",
    validate="one_to_one"
)

development_enriched_output = (
    OUTPUT_DIR
    / "five_class_development_morphology_enriched.csv"
)

development_enriched.to_csv(
    development_enriched_output,
    index=False
)

print(
    "Saved enriched development dataset:"
)
print(development_enriched_output)


# ------------------------------------------------------------
# 16. VALIDATION CHECKS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

# Check that the number of development rows did not change.
if len(development_enriched) != len(development):
    stop_with_error(
        "Development row count changed after merging."
    )

print("Development row count preserved: PASSED")


# Check that original labels were preserved.
original_labels = (
    development[TARGET_COLUMN]
    .reset_index(drop=True)
)

enriched_labels = (
    development_enriched[TARGET_COLUMN]
    .reset_index(drop=True)
)

if not original_labels.equals(enriched_labels):
    stop_with_error(
        "Target labels changed during enrichment."
    )

print("Target labels preserved: PASSED")


# Check CV fold assignments if present.
if "cv_fold" in development.columns:

    original_folds = (
        development["cv_fold"]
        .reset_index(drop=True)
    )

    enriched_folds = (
        development_enriched["cv_fold"]
        .reset_index(drop=True)
    )

    if not original_folds.equals(enriched_folds):
        stop_with_error(
            "CV fold assignments changed during enrichment."
        )

    print("CV fold assignments preserved: PASSED")


# Check 201 and 202 remain in the same subject group.
if (
    "record_id" in development_enriched.columns
    and "group_id" in development_enriched.columns
):

    subject_records = development_enriched[
        development_enriched["record_id"].isin(
            ["201", "202"]
        )
    ]

    if len(subject_records) > 0:

        groups_201_202 = (
            subject_records["group_id"]
            .dropna()
            .unique()
        )

        if len(groups_201_202) > 1:
            stop_with_error(
                "Records 201 and 202 are no longer grouped together."
            )

        print("Records 201/202 grouping preserved: PASSED")


# Check that every development window received a matching
# morphology aggregation.
missing_aggregation = (
    development_enriched[
        "aggregated_beat_count"
    ].isna().sum()
)

print(
    "Development windows without matched beat aggregates:",
    missing_aggregation
)

if missing_aggregation > 0:
    print(
        "WARNING: Some windows did not receive beat aggregates. "
        "Inspect the matching report before model training."
    )
else:
    print("All development windows matched: PASSED")


# ------------------------------------------------------------
# 17. SAVE FEATURE LIST AND SUMMARY REPORT
# ------------------------------------------------------------

# Identify newly added morphology columns.
original_columns = set(development.columns)

new_feature_columns = [
    column
    for column in development_enriched.columns
    if column not in original_columns
]

feature_list_output = (
    OUTPUT_DIR
    / "step9b_morphology_feature_list.txt"
)

with open(
    feature_list_output,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "STEP 9B MORPHOLOGY-ENRICHED FEATURE LIST\n"
    )

    file.write("=" * 60 + "\n\n")

    for feature in new_feature_columns:
        file.write(feature + "\n")


report_output = (
    OUTPUT_DIR
    / "step9b_dataset_report.txt"
)

with open(
    report_output,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "STEP 9B MORPHOLOGY-ENRICHED DATASET REPORT\n"
    )

    file.write("=" * 70 + "\n\n")

    file.write(
        f"Master beat dataset rows: {len(master)}\n"
    )

    file.write(
        f"Clean windows rows: {len(clean_windows)}\n"
    )

    file.write(
        f"Development rows: {len(development)}\n"
    )

    file.write(
        f"Enriched development rows: "
        f"{len(development_enriched)}\n"
    )

    file.write(
        f"New morphology feature columns: "
        f"{len(new_feature_columns)}\n"
    )

    file.write(
        f"Development windows without matched "
        f"beat aggregates: {missing_aggregation}\n\n"
    )

    file.write(
        "Original target labels preserved: YES\n"
    )

    file.write(
        "Original CV folds preserved: YES\n"
    )

    file.write(
        "Final test dataset loaded: NO\n"
    )

    file.write(
        "Final test dataset modified: NO\n"
    )


# ------------------------------------------------------------
# 18. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 9B COMPLETED")
print("=" * 70)

print(
    "\nOriginal development windows:",
    len(development)
)

print(
    "Enriched development windows:",
    len(development_enriched)
)

print(
    "New morphology feature columns:",
    len(new_feature_columns)
)

print("\nOutput folder:")
print(OUTPUT_DIR)

print("\nImportant output files:")

print(
    "1.",
    "step9b_morphology_features_all_clean_windows.csv"
)

print(
    "2.",
    "five_class_windows_morphology_enriched.csv"
)

print(
    "3.",
    "five_class_development_morphology_enriched.csv"
)

print(
    "4.",
    "step9b_morphology_feature_list.txt"
)

print(
    "5.",
    "step9b_dataset_report.txt"
)

print("\nFinal test dataset was not loaded or modified.")
print("\nPlease review the validation output before proceeding.")
