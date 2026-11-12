# ============================================================
# STEP 5: LEAKAGE-SAFE DATASET PREPARATION
# MIT-BIH Arrhythmia Database
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold


# ============================================================
# 1. PATHS
# ============================================================

MASTER_CSV = r"c:\Users\Himanshu Mohapatra\OneDrive\Desktop\Project_Main\master_MITBIH_48_records_features.csv"

OUTPUT_DIR = r"c:\Users\Himanshu Mohapatra\OneDrive\Desktop\Project_Main\ml_dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 5: LEAKAGE-SAFE DATASET PREPARATION")
print("=" * 70)

print("\nLoading master dataset...")

df = pd.read_csv(MASTER_CSV)

print(f"Dataset shape: {df.shape}")
print(f"Number of records: {df['record_id'].nunique()}")
print(f"Number of channels: {df['channel'].nunique()}")
print(f"Number of beat classes: {df['beat_label'].nunique()}")


# ============================================================
# 3. BASIC VALIDATION
# ============================================================

required_columns = [
    "record_id",
    "channel",
    "channel_index",
    "beat_index",
    "beat_label",
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column missing: {col}")


print("\nBeat-class distribution:")
print(df["beat_label"].value_counts())


# ============================================================
# 4. CREATE LEAKAGE-SAFE GROUP
# ============================================================
#
# Normally:
#     each ECG record = one group
#
# Special case:
#     records 201 and 202 belong to the same subject
#     so they must be kept together.
#
# This prevents:
#
#       record 201 -> training
#       record 202 -> testing
#
# which would create subject-level leakage.
# ============================================================

df["group_id"] = df["record_id"].astype(str)

df.loc[
    df["record_id"].isin([201, 202]),
    "group_id"
] = "subject_201_202"


print("\nGrouping information:")
print(f"Number of unique groups: {df['group_id'].nunique()}")

print("\nGroups:")
group_summary = (
    df.groupby("group_id")
    .agg(
        records=("record_id", lambda x: ",".join(sorted(x.astype(str).unique()))),
        rows=("record_id", "size"),
        classes=("beat_label", "nunique")
    )
    .reset_index()
)

print(group_summary.to_string(index=False))


# ============================================================
# 5. DEFINE FEATURES
# ============================================================
#
# Primary Core feature set from Step 4:
#
#   RR_interval_ms
#   R_amplitude_mV
#   P_amplitude_mV
#   P_duration_ms
#
# Extended features will be tested later.
# ============================================================

CORE_FEATURES = [
    "RR_interval_ms",
    "R_amplitude_mV",
    "P_amplitude_mV",
    "P_duration_ms",
]

EXTENDED_FEATURES = CORE_FEATURES + [
    "QRS_duration_ms",
    "PR_interval_ms",
    "QT_interval_ms",
    "T_amplitude_mV",
    "T_duration_ms",
]


# ============================================================
# 6. COLUMNS THAT MUST NOT BE MODEL INPUTS
# ============================================================
#
# These columns contain identifiers, target information,
# positional information, or delineation metadata.
#
# In particular:
#
# record_id
# channel
# channel_index
# beat_index
# R_sample
# R_time_s
#
# should NOT be used as predictive features.
# ============================================================

METADATA_COLUMNS = [
    "record_id",
    "channel",
    "channel_index",
    "beat_index",
    "beat_label",
    "group_id",
]

TIMING_POSITION_COLUMNS = [
    "R_sample",
    "R_time_s",
    "P_onset_sample",
    "P_onset_s",
    "P_peak_sample",
    "P_peak_s",
    "P_offset_sample",
    "P_offset_s",
    "QRS_onset_sample",
    "QRS_onset_s",
    "QRS_offset_sample",
    "QRS_offset_s",
    "T_onset_sample",
    "T_onset_s",
    "T_peak_sample",
    "T_peak_s",
    "T_offset_sample",
    "T_offset_s",
]

LEAKAGE_COLUMNS = (
    METADATA_COLUMNS
    + TIMING_POSITION_COLUMNS
)


# ============================================================
# 7. CHECK FEATURE AVAILABILITY
# ============================================================

for feature in EXTENDED_FEATURES:
    if feature not in df.columns:
        raise ValueError(f"Feature not found: {feature}")


# ============================================================
# 8. PREPARE X / y / GROUPS
# ============================================================

X_core = df[CORE_FEATURES].copy()
X_extended = df[EXTENDED_FEATURES].copy()

y = df["beat_label"].copy()
groups = df["group_id"].copy()


# ============================================================
# 9. CHECK FOR IDENTIFIER LEAKAGE
# ============================================================

print("\nChecking leakage-prone columns...")

for col in LEAKAGE_COLUMNS:
    if col in CORE_FEATURES or col in EXTENDED_FEATURES:
        raise ValueError(
            f"LEAKAGE ERROR: {col} appears in model features."
        )

print("No metadata/position columns included in feature sets.")


# ============================================================
# 10. GROUPED STRATIFIED SPLITTING
# ============================================================
#
# We use StratifiedGroupKFold.
#
# First:
#     5 folds
#
# Approximately:
#     80% train
#     20% temporary
#
# Then split the temporary portion into:
#     validation
#     test
#
# while preserving groups.
# ============================================================

print("\nCreating leakage-safe train/validation/test split...")

random_state = 42

sgkf = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=random_state
)

# ------------------------------------------------------------
# First split:
# choose one fold as temporary test/validation set
# ------------------------------------------------------------

folds = list(
    sgkf.split(
        X_core,
        y,
        groups=groups
    )
)

train_idx, temp_idx = folds[0]

train_groups = set(groups.iloc[train_idx])
temp_groups = set(groups.iloc[temp_idx])

print("\nInitial split:")
print(f"Train groups: {len(train_groups)}")
print(f"Temporary groups: {len(temp_groups)}")


# ============================================================
# 11. SPLIT TEMPORARY SET INTO VALIDATION + TEST
# ============================================================

temp_df = df.iloc[temp_idx].copy()

temp_X = temp_df[CORE_FEATURES]
temp_y = temp_df["beat_label"]
temp_groups = temp_df["group_id"]

sgkf_temp = StratifiedGroupKFold(
    n_splits=2,
    shuffle=True,
    random_state=random_state
)

temp_folds = list(
    sgkf_temp.split(
        temp_X,
        temp_y,
        groups=temp_groups
    )
)

val_relative_idx, test_relative_idx = temp_folds[0]

val_idx = temp_df.index[val_relative_idx].to_numpy()
test_idx = temp_df.index[test_relative_idx].to_numpy()


# ============================================================
# 12. FINAL DATASETS
# ============================================================

train_df = df.iloc[train_idx].copy()
val_df = df.loc[val_idx].copy()
test_df = df.loc[test_idx].copy()


# ============================================================
# 13. SORT DATASETS
# ============================================================

sort_columns = [
    "record_id",
    "channel_index",
    "beat_index"
]

train_df = train_df.sort_values(sort_columns)
val_df = val_df.sort_values(sort_columns)
test_df = test_df.sort_values(sort_columns)


# ============================================================
# 14. LEAKAGE CHECK
# ============================================================

train_groups = set(train_df["group_id"])
val_groups = set(val_df["group_id"])
test_groups = set(test_df["group_id"])

train_val_overlap = train_groups.intersection(val_groups)
train_test_overlap = train_groups.intersection(test_groups)
val_test_overlap = val_groups.intersection(test_groups)

print("\n" + "=" * 70)
print("LEAKAGE CHECK")
print("=" * 70)

print(f"Train ∩ Validation groups: {len(train_val_overlap)}")
print(f"Train ∩ Test groups:       {len(train_test_overlap)}")
print(f"Validation ∩ Test groups:  {len(val_test_overlap)}")

if train_val_overlap:
    print("ERROR: Train/validation leakage detected!")
    print(train_val_overlap)
    raise RuntimeError("Train-validation leakage detected.")

if train_test_overlap:
    print("ERROR: Train/test leakage detected!")
    print(train_test_overlap)
    raise RuntimeError("Train-test leakage detected.")

if val_test_overlap:
    print("ERROR: Validation/test leakage detected!")
    print(val_test_overlap)
    raise RuntimeError("Validation-test leakage detected.")

print("\nPASS: No group appears in more than one split.")


# ============================================================
# 15. CHECK 201 / 202
# ============================================================

print("\nChecking records 201 and 202...")

for split_name, split_df in [
    ("train", train_df),
    ("validation", val_df),
    ("test", test_df)
]:
    records = set(split_df["record_id"].unique())

    has_201 = 201 in records
    has_202 = 202 in records

    if has_201 != has_202:
        raise RuntimeError(
            f"ERROR: records 201 and 202 separated in {split_name}."
        )

    if has_201 and has_202:
        print(f"201 and 202 are together in {split_name}.")

print("PASS: 201/202 subject grouping preserved.")


# ============================================================
# 16. DATASET SUMMARY
# ============================================================

def print_split_summary(name, split_df):

    print("\n" + "-" * 70)
    print(name.upper())

    print(f"Rows:       {len(split_df):,}")
    print(f"Records:    {split_df['record_id'].nunique()}")
    print(f"Groups:     {split_df['group_id'].nunique()}")
    print(f"Channels:   {split_df['channel'].nunique()}")

    print("\nClass distribution:")

    counts = split_df["beat_label"].value_counts()
    percentages = split_df["beat_label"].value_counts(normalize=True) * 100

    summary = pd.DataFrame({
        "count": counts,
        "percentage": percentages.round(3)
    })

    print(summary.to_string())


print_split_summary("Train", train_df)
print_split_summary("Validation", val_df)
print_split_summary("Test", test_df)


# ============================================================
# 17. CHECK WHETHER EVERY CLASS EXISTS IN EACH SPLIT
# ============================================================

all_classes = sorted(df["beat_label"].unique())

print("\n" + "=" * 70)
print("CLASS COVERAGE CHECK")
print("=" * 70)

coverage_rows = []

for label in all_classes:

    train_count = int(
        (train_df["beat_label"] == label).sum()
    )

    val_count = int(
        (val_df["beat_label"] == label).sum()
    )

    test_count = int(
        (test_df["beat_label"] == label).sum()
    )

    coverage_rows.append({
        "beat_label": label,
        "train_count": train_count,
        "validation_count": val_count,
        "test_count": test_count,
        "present_in_all": (
            train_count > 0
            and val_count > 0
            and test_count > 0
        )
    })

coverage_df = pd.DataFrame(coverage_rows)

print(coverage_df.to_string(index=False))


# ============================================================
# 18. SAVE DATASETS
# ============================================================

print("\nSaving datasets...")

train_path = os.path.join(
    OUTPUT_DIR,
    "train.csv"
)

val_path = os.path.join(
    OUTPUT_DIR,
    "validation.csv"
)

test_path = os.path.join(
    OUTPUT_DIR,
    "test.csv"
)

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)


# ============================================================
# 19. SAVE FEATURE DEFINITIONS
# ============================================================

# ============================================================
# SAVE FEATURE DEFINITIONS
# ============================================================

feature_rows = []

for feature in CORE_FEATURES:
    feature_rows.append({
        "feature": feature,
        "set": "Core"
    })

for feature in EXTENDED_FEATURES:
    if feature not in CORE_FEATURES:
        feature_rows.append({
            "feature": feature,
            "set": "Extended-only"
        })

feature_info = pd.DataFrame(feature_rows)

feature_info.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "feature_definitions.csv"
    ),
    index=False
)

# ============================================================
# 21. SAVE SPLIT MANIFEST
# ============================================================

manifest_rows = []

for split_name, split_df in [
    ("train", train_df),
    ("validation", val_df),
    ("test", test_df)
]:

    for group_id in sorted(split_df["group_id"].unique()):

        group_records = sorted(
            split_df.loc[
                split_df["group_id"] == group_id,
                "record_id"
            ].unique()
        )

        manifest_rows.append({
            "split": split_name,
            "group_id": group_id,
            "records": ",".join(
                str(x) for x in group_records
            )
        })

manifest_df = pd.DataFrame(manifest_rows)

manifest_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "split_manifest.csv"
    ),
    index=False
)


# ============================================================
# 22. SAVE MODEL COLUMN INFORMATION
# ============================================================

model_columns = pd.DataFrame({
    "column": (
        CORE_FEATURES
        + [
            "QRS_duration_ms",
            "PR_interval_ms",
            "QT_interval_ms",
            "T_amplitude_mV",
            "T_duration_ms",
        ]
    )
})

model_columns.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_feature_columns.csv"
    ),
    index=False
)


# ============================================================
# 23. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 5 COMPLETE")
print("=" * 70)

print(f"\nOutput directory:")
print(OUTPUT_DIR)

print("\nFiles created:")

for filename in sorted(os.listdir(OUTPUT_DIR)):
    print("  -", filename)

print("\nFinal split sizes:")
print(f"Train:      {len(train_df):,}")
print(f"Validation: {len(val_df):,}")
print(f"Test:       {len(test_df):,}")

print("\nFinal group counts:")
print(f"Train:      {train_df['group_id'].nunique()}")
print(f"Validation: {val_df['group_id'].nunique()}")
print(f"Test:       {test_df['group_id'].nunique()}")

print("\nCore features:")
for feature in CORE_FEATURES:
    print("  -", feature)

print("\nExtended features:")
for feature in EXTENDED_FEATURES:
    print("  -", feature)

print("\nTarget:")
print("  beat_label")

print("\nImportant:")
print("  No record/group is shared between train, validation, and test.")
print("  Records 201 and 202 are forced into the same group.")
print("=" * 70)