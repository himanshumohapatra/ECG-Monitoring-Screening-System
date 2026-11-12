# ============================================================
# STEP 8B
# Five-Class Window Dataset Construction & Leakage-Safe Split
# ============================================================

import random
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\Himanshu Mohapatra\OneDrive\Desktop\Project_Main"
)

INPUT_DIR = PROJECT_DIR / "step8a_label_feasibility"
OUTPUT_DIR = PROJECT_DIR / "step8b_five_class_dataset"

INPUT_FILE = INPUT_DIR / "06_preliminary_20s_windows.csv"

# OLD beat-classification final test.
# NEVER load or modify this file.
OLD_FINAL_TEST = (
    PROJECT_DIR
    / "ml_grouped_cv"
    / "final_test.csv"
)


# ============================================================
# 2. SETTINGS
# ============================================================

RANDOM_STATE = 42
TEST_GROUP_FRACTION = 0.20
N_RANDOM_SEARCH = 30000

TARGET_CLASSES = [
    "Normal heartbeat",
    "Arrhythmia",
    "Tachycardia",
    "Bradycardia",
    "Other abnormal ECG",
]

CHANNEL_PRIORITY = [
    "MLII",
    "V1",
    "V5",
    "V2",
    "V4",
]


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def normalize_record_id(value):

    try:
        number = float(value)

        if number.is_integer():
            return str(int(number))

    except Exception:
        pass

    return str(value)


def make_group_id(record_id):

    record_str = normalize_record_id(record_id)

    if record_str in {"201", "202"}:
        return "subject_201_202"

    return record_str


def choose_primary_channel(group_df):

    available_channels = (
        group_df["channel"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    for preferred in CHANNEL_PRIORITY:

        if preferred in available_channels:
            return preferred

    counts = (
        group_df["channel"]
        .astype(str)
        .value_counts()
    )

    return counts.index[0]


def class_distribution(df):

    return (
        df[CLASS_COLUMN]
        .value_counts()
        .reindex(
            TARGET_CLASSES,
            fill_value=0
        )
    )


def distribution_score(
    test_counts,
    total_counts
):

    test_fraction = (
        test_counts
        / test_counts.sum()
    )

    total_fraction = (
        total_counts
        / total_counts.sum()
    )

    return float(
        np.abs(
            test_fraction
            - total_fraction
        ).sum()
    )


def evaluate_test_group_selection(
    selected_groups,
    group_table,
    total_class_counts,
):

    selected = group_table[
        group_table["group_id"]
        .isin(selected_groups)
    ]

    test_rows = int(
        selected["n_windows"].sum()
    )

    total_rows = int(
        group_table["n_windows"].sum()
    )

    test_fraction = (
        test_rows / total_rows
    )

    test_class_counts = (
        selected[TARGET_CLASSES]
        .sum()
    )

    # Every target class must exist in test
    if (
        test_class_counts <= 0
    ).any():

        return None

    class_score = distribution_score(
        test_class_counts,
        total_class_counts,
    )

    size_penalty = abs(
        test_fraction
        - TEST_GROUP_FRACTION
    )

    score = (
        class_score
        + 2.0 * size_penalty
    )

    return {
        "score": score,
        "test_rows": test_rows,
        "test_fraction": test_fraction,
        "class_score": class_score,
        "size_penalty": size_penalty,
        "test_class_counts": test_class_counts,
    }


# ============================================================
# 4. START
# ============================================================

print("=" * 70)
print("STEP 8B - FIVE-CLASS WINDOW DATASET CONSTRUCTION")
print("=" * 70)

print("\nProject directory:")
print(PROJECT_DIR)

print("\nInput:")
print(INPUT_FILE)

print("\nOutput:")
print(OUTPUT_DIR)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

if OLD_FINAL_TEST.exists():

    print(
        "\nProtected old final_test.csv detected:"
    )

    print(OLD_FINAL_TEST)

    print(
        "This file will NOT be loaded or modified."
    )

else:

    print(
        "\nOld final_test.csv was not found."
    )


# ============================================================
# 5. LOAD STEP 8A DATA
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Step 8A input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE
)

print(
    "\nLoaded Step 8A dataset:"
)

print(
    "Rows:",
    len(df)
)

print(
    "Columns:",
    len(df.columns)
)


# ============================================================
# 6. PRINT ACTUAL COLUMN NAMES
# ============================================================

print(
    "\nActual Step 8A columns:"
)

for i, column in enumerate(
    df.columns,
    start=1
):

    print(
        f"  {i:02d}. {column}"
    )


# ============================================================
# 7. AUTOMATICALLY DETECT CLASS COLUMN
# ============================================================

possible_class_columns = [
    "target_class",
    "window_class",
    "five_class",
    "class",
    "label",
    "window_label",
    "target_label",
    "preliminary_class",
]

CLASS_COLUMN = None

for candidate in possible_class_columns:

    if candidate in df.columns:

        values = set(
            df[candidate]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
        )

        # Detect the column if it contains the expected
        # five target class names.
        overlap = (
            values
            & set(TARGET_CLASSES)
        )

        if len(overlap) >= 3:

            CLASS_COLUMN = candidate
            break


# If automatic detection failed, search every column.
if CLASS_COLUMN is None:

    for column in df.columns:

        values = set(
            df[column]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
        )

        overlap = (
            values
            & set(TARGET_CLASSES)
        )

        if len(overlap) >= 3:

            CLASS_COLUMN = column
            break


if CLASS_COLUMN is None:

    raise RuntimeError(
        "\nCould not automatically identify the five-class "
        "window label column.\n\n"
        "The CSV columns printed above show the actual names.\n"
        "Expected values include:\n"
        + "\n".join(
            f"  - {x}"
            for x in TARGET_CLASSES
        )
    )


print(
    "\nDetected five-class label column:"
)

print(
    "  ",
    CLASS_COLUMN
)


# ============================================================
# 8. REQUIRED COLUMN CHECK
# ============================================================

required_columns = [
    "record_id",
    "channel",
    CLASS_COLUMN,
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    raise RuntimeError(
        "Missing required columns:\n"
        + "\n".join(
            missing_columns
        )
    )

print(
    "\nRequired columns detected:"
)

for col in required_columns:

    print(
        "  OK:",
        col
    )


# ============================================================
# 9. ORIGINAL DISTRIBUTION
# ============================================================

print(
    "\nOriginal Step 8A class distribution:"
)

original_distribution = (
    df[CLASS_COLUMN]
    .astype(str)
    .str.strip()
    .value_counts()
)

for class_name, count in (
    original_distribution.items()
):

    print(
        f"  {class_name:<25}"
        f"{count:>7}"
        f"  "
        f"{count / len(df) * 100:>8.3f}%"
    )


# ============================================================
# 10. REMOVE UNCERTAIN / MIXED
# ============================================================

uncertain_mask = (
    df[CLASS_COLUMN]
    .astype(str)
    .str.strip()
    .eq("Uncertain / mixed")
)

uncertain_count = int(
    uncertain_mask.sum()
)

print(
    "\nRemoving uncertain/mixed windows:"
)

print(
    "  Removed:",
    uncertain_count
)

df = df.loc[
    ~uncertain_mask
].copy()

df.reset_index(
    drop=True,
    inplace=True
)

print(
    "  Remaining:",
    len(df)
)


# ============================================================
# 11. CHECK TARGET CLASSES
# ============================================================

unexpected_classes = sorted(
    set(
        df[CLASS_COLUMN]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )
    - set(TARGET_CLASSES)
)

if unexpected_classes:

    raise RuntimeError(
        "Unexpected classes found:\n"
        + "\n".join(
            unexpected_classes
        )
    )


# ============================================================
# 12. CHANNEL DISTRIBUTION
# ============================================================

print(
    "\nChannel distribution after removing uncertain windows:"
)

channel_distribution = (
    df["channel"]
    .astype(str)
    .value_counts()
)

for channel, count in (
    channel_distribution.items()
):

    print(
        f"  {channel:<10}"
        f"{count:>7}"
        f"  "
        f"{count / len(df) * 100:>8.3f}%"
    )


# ============================================================
# 13. RECORD / GROUP IDs
# ============================================================

df["record_id"] = (
    df["record_id"]
    .apply(normalize_record_id)
)

df["group_id"] = (
    df["record_id"]
    .apply(make_group_id)
)


# ============================================================
# 14. ONE PRIMARY CHANNEL PER RECORD
# ============================================================

print(
    "\nSelecting ONE primary ECG channel per record..."
)

selected_channel_rows = []

channel_selection_records = []

for record_id, record_df in (
    df.groupby(
        "record_id",
        sort=True
    )
):

    selected_channel = (
        choose_primary_channel(
            record_df
        )
    )

    selected = record_df[
        record_df["channel"]
        .astype(str)
        == selected_channel
    ].copy()

    selected_channel_rows.append(
        selected
    )

    channel_selection_records.append({
        "record_id": record_id,
        "selected_channel": selected_channel,
        "available_channels": ",".join(
            sorted(
                record_df["channel"]
                .astype(str)
                .unique()
            )
        ),
        "windows_selected": len(
            selected
        ),
        "windows_before_selection": len(
            record_df
        ),
    })


df_clean = pd.concat(
    selected_channel_rows,
    ignore_index=True
)

channel_selection_df = pd.DataFrame(
    channel_selection_records
)

print(
    "\nPrimary channel selection:"
)

for _, row in (
    channel_selection_df.iterrows()
):

    print(
        f"  Record {row['record_id']:>3} -> "
        f"{row['selected_channel']:<5} "
        f"({row['windows_selected']} windows)"
    )


# ============================================================
# 15. ONE CHANNEL CHECK
# ============================================================

channels_per_record = (
    df_clean
    .groupby("record_id")["channel"]
    .nunique()
)

bad_records = channels_per_record[
    channels_per_record > 1
]

if len(bad_records) > 0:

    raise RuntimeError(
        "More than one selected channel exists:\n"
        + str(bad_records)
    )

print(
    "\nPrimary-channel check: PASS"
)


# ============================================================
# 16. DUPLICATE CHECK
# ============================================================

possible_window_columns = [
    "record_id",
    "channel",
    "window_start_s",
    "window_end_s",
]

existing_window_columns = [
    col
    for col in possible_window_columns
    if col in df_clean.columns
]

if len(existing_window_columns) >= 3:

    duplicate_count = int(
        df_clean
        .duplicated(
            subset=existing_window_columns
        )
        .sum()
    )

else:

    duplicate_count = 0

if duplicate_count > 0:

    raise RuntimeError(
        f"Duplicate windows detected: "
        f"{duplicate_count}"
    )

print(
    "Duplicate-window check: PASS"
)


# ============================================================
# 17. CLEAN CLASS DISTRIBUTION
# ============================================================

print(
    "\nFive-class distribution after "
    "primary-channel selection:"
)

clean_distribution = (
    class_distribution(
        df_clean
    )
)

for class_name in TARGET_CLASSES:

    count = int(
        clean_distribution[
            class_name
        ]
    )

    percentage = (
        count
        / len(df_clean)
        * 100
    )

    print(
        f"  {class_name:<25}"
        f"{count:>7}"
        f"  "
        f"{percentage:>8.3f}%"
    )


# ============================================================
# 18. GROUP TABLE
# ============================================================

group_counts = (
    df_clean
    .groupby("group_id")
    .size()
    .rename("n_windows")
    .reset_index()
)

group_class_counts = (
    pd.crosstab(
        df_clean["group_id"],
        df_clean[CLASS_COLUMN]
    )
    .reindex(
        columns=TARGET_CLASSES,
        fill_value=0
    )
    .reset_index()
)

group_table = group_counts.merge(
    group_class_counts,
    on="group_id",
    how="left"
)

print(
    "\nGroup information:"
)

print(
    "  Independent groups:",
    group_table["group_id"].nunique()
)

print(
    "  Records:",
    df_clean["record_id"].nunique()
)


# ============================================================
# 19. 201 / 202 GROUP CHECK
# ============================================================

subject_records = set(
    df_clean.loc[
        df_clean["group_id"]
        == "subject_201_202",
        "record_id"
    ].unique()
)

print(
    "\n201/202 grouping check:"
)

print(
    "  Records:",
    sorted(subject_records)
)

if not {
    "201",
    "202"
}.issubset(subject_records):

    raise RuntimeError(
        "201 and 202 are not both inside "
        "subject_201_202."
    )

print(
    "  PASS: 201 and 202 are grouped together."
)


# ============================================================
# 20. SAVE CLEAN DATASET
# ============================================================

clean_file = (
    OUTPUT_DIR
    / "five_class_windows_clean.csv"
)

df_clean.to_csv(
    clean_file,
    index=False
)

print(
    "\nSaved clean five-class dataset:"
)

print(clean_file)


# ============================================================
# 21. FIND FINAL TEST GROUPS
# ============================================================

all_groups = sorted(
    group_table["group_id"]
    .unique()
)

n_groups = len(all_groups)

target_test_groups = max(
    1,
    round(
        n_groups
        * TEST_GROUP_FRACTION
    )
)

print(
    "\nFinal test-group search:"
)

print(
    "  Total independent groups:",
    n_groups
)

print(
    "  Target test groups:",
    target_test_groups
)

total_class_counts = (
    group_table[TARGET_CLASSES]
    .sum()
)

rng = random.Random(
    RANDOM_STATE
)

best_result = None
best_groups = None

candidate_groups = list(
    all_groups
)

for _ in range(
    N_RANDOM_SEARCH
):

    selected_groups = rng.sample(
        candidate_groups,
        target_test_groups
    )

    result = (
        evaluate_test_group_selection(
            selected_groups,
            group_table,
            total_class_counts,
        )
    )

    if result is None:
        continue

    if (
        best_result is None
        or result["score"]
        < best_result["score"]
    ):

        best_result = result
        best_groups = selected_groups


if best_groups is None:

    raise RuntimeError(
        "Could not find a valid final-test "
        "group selection containing all five classes."
    )


print(
    "\nBest test-group selection score:",
    round(
        best_result["score"],
        6
    )
)

print(
    "\nFinal test groups:"
)

for group_id in sorted(
    best_groups
):

    print(
        " ",
        group_id
    )


# ============================================================
# 22. DEVELOPMENT / TEST SPLIT
# ============================================================

test_group_set = set(
    best_groups
)

development_group_set = (
    set(all_groups)
    - test_group_set
)

development_df = df_clean[
    df_clean["group_id"]
    .isin(
        development_group_set
    )
].copy()

test_df = df_clean[
    df_clean["group_id"]
    .isin(
        test_group_set
    )
].copy()

development_df.reset_index(
    drop=True,
    inplace=True
)

test_df.reset_index(
    drop=True,
    inplace=True
)


# ============================================================
# 23. DEVELOPMENT / TEST LEAKAGE CHECK
# ============================================================

development_groups = set(
    development_df["group_id"]
)

test_groups = set(
    test_df["group_id"]
)

overlap = (
    development_groups
    & test_groups
)

print(
    "\nDevelopment/test leakage check:"
)

print(
    "  Development groups:",
    len(development_groups)
)

print(
    "  Test groups:",
    len(test_groups)
)

print(
    "  Group overlap:",
    len(overlap)
)

if overlap:

    raise RuntimeError(
        f"Development/test leakage:\n{overlap}"
    )

print(
    "  PASS: no development/test group overlap."
)


# ============================================================
# 24. 201 / 202 FINAL SPLIT CHECK
# ============================================================

records_dev = set(
    development_df["record_id"]
)

records_test = set(
    test_df["record_id"]
)

location_201 = (
    "development"
    if "201" in records_dev
    else (
        "test"
        if "201" in records_test
        else "missing"
    )
)

location_202 = (
    "development"
    if "202" in records_dev
    else (
        "test"
        if "202" in records_test
        else "missing"
    )
)

print(
    "\n201/202 final split check:"
)

print(
    "  201:",
    location_201
)

print(
    "  202:",
    location_202
)

if location_201 != location_202:

    raise RuntimeError(
        "201 and 202 were split between "
        "development and test."
    )

print(
    "  PASS: 201 and 202 remain together."
)


# ============================================================
# 25. DEVELOPMENT / TEST COUNTS
# ============================================================

development_class_counts = (
    class_distribution(
        development_df
    )
)

test_class_counts = (
    class_distribution(
        test_df
    )
)

print(
    "\nDevelopment class counts:"
)

for class_name in TARGET_CLASSES:

    print(
        f"  {class_name:<25}"
        f"{int(development_class_counts[class_name]):>7}"
    )

print(
    "\nTest class counts:"
)

for class_name in TARGET_CLASSES:

    print(
        f"  {class_name:<25}"
        f"{int(test_class_counts[class_name]):>7}"
    )


# ============================================================
# 26. FIVE-CLASS COVERAGE CHECK
# ============================================================

missing_dev_classes = [
    class_name
    for class_name in TARGET_CLASSES
    if development_class_counts[
        class_name
    ] == 0
]

missing_test_classes = [
    class_name
    for class_name in TARGET_CLASSES
    if test_class_counts[
        class_name
    ] == 0
]

if missing_dev_classes:

    raise RuntimeError(
        "Development missing classes:\n"
        + "\n".join(
            missing_dev_classes
        )
    )

if missing_test_classes:

    raise RuntimeError(
        "Test missing classes:\n"
        + "\n".join(
            missing_test_classes
        )
    )

print(
    "\nFive-class coverage check: PASS"
)

print(
    "All five classes exist in development and test."
)


# ============================================================
# 27. SAVE DEVELOPMENT / TEST
# ============================================================

development_file = (
    OUTPUT_DIR
    / "five_class_development.csv"
)

test_file = (
    OUTPUT_DIR
    / "five_class_final_test.csv"
)

development_df.to_csv(
    development_file,
    index=False
)

test_df.to_csv(
    test_file,
    index=False
)

print(
    "\nSaved development dataset:"
)

print(development_file)

print(
    "\nSaved NEW five-class final test dataset:"
)

print(test_file)


# ============================================================
# 28. CREATE 5 GROUPED CV FOLDS
# ============================================================

print(
    "\nCreating 5 leakage-safe CV folds..."
)

development_df["cv_fold"] = -1

sgkf = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

fold_summary_records = []

for fold_number, (
    train_idx,
    validation_idx
) in enumerate(
    sgkf.split(
        development_df,
        development_df[CLASS_COLUMN],
        groups=development_df["group_id"],
    ),
    start=1,
):

    development_df.loc[
        validation_idx,
        "cv_fold"
    ] = fold_number

    train_part = (
        development_df.iloc[
            train_idx
        ]
    )

    validation_part = (
        development_df.iloc[
            validation_idx
        ]
    )

    train_groups = set(
        train_part["group_id"]
    )

    validation_groups = set(
        validation_part["group_id"]
    )

    fold_overlap = (
        train_groups
        & validation_groups
    )

    if fold_overlap:

        raise RuntimeError(
            f"Fold {fold_number} "
            f"group leakage:\n"
            f"{fold_overlap}"
        )

    train_counts = (
        class_distribution(
            train_part
        )
    )

    validation_counts = (
        class_distribution(
            validation_part
        )
    )

    record = {
        "fold": fold_number,
        "train_rows": len(
            train_part
        ),
        "validation_rows": len(
            validation_part
        ),
        "train_groups": len(
            train_groups
        ),
        "validation_groups": len(
            validation_groups
        ),
    }

    for class_name in TARGET_CLASSES:

        record[
            f"train_{class_name}"
        ] = int(
            train_counts[class_name]
        )

        record[
            f"validation_{class_name}"
        ] = int(
            validation_counts[class_name]
        )

    fold_summary_records.append(
        record
    )

    print(
        f"\nFold {fold_number}:"
    )

    print(
        "  Train rows:",
        len(train_part)
    )

    print(
        "  Validation rows:",
        len(validation_part)
    )

    print(
        "  Train groups:",
        len(train_groups)
    )

    print(
        "  Validation groups:",
        len(validation_groups)
    )

    print(
        "  Group leakage: PASS"
    )


# ============================================================
# 29. DEVELOPMENT GROUP -> EXACTLY ONE CV FOLD
# ============================================================

group_fold_counts = (
    development_df
    .groupby("group_id")["cv_fold"]
    .nunique()
)

bad_cv_groups = group_fold_counts[
    group_fold_counts != 1
]

if len(bad_cv_groups) > 0:

    raise RuntimeError(
        "Some development groups do not belong "
        "to exactly one CV fold:\n"
        + str(bad_cv_groups)
    )

print(
    "\nDevelopment CV group assignment check: PASS"
)

print(
    "Every development group belongs "
    "to exactly one validation fold."
)


# ============================================================
# 30. CORRECT 201 / 202 CV CHECK
# ============================================================

subject_group = (
    "subject_201_202"
)

subject_in_test = (
    subject_group
    in test_groups
)

subject_in_development = (
    subject_group
    in development_groups
)

if (
    subject_in_test
    and subject_in_development
):

    raise RuntimeError(
        "subject_201_202 appears in both "
        "development and test."
    )


if subject_in_test:

    # This is CORRECT.
    # Final-test groups do not receive CV folds.

    subject_cv_folds = []

    print(
        "\n201/202 CV-fold assignment:"
    )

    print(
        "  subject_201_202 is in FINAL TEST."
    )

    print(
        "  CV folds: []"
    )

    print(
        "  PASS: no CV fold is expected "
        "for a final-test group."
    )


elif subject_in_development:

    subject_cv_folds = sorted(
        development_df.loc[
            development_df["group_id"]
            == subject_group,
            "cv_fold"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    print(
        "\n201/202 CV-fold assignment:"
    )

    print(
        " ",
        subject_cv_folds
    )

    if len(subject_cv_folds) != 1:

        raise RuntimeError(
            "201/202 development group "
            "was split across CV folds."
        )

    print(
        "  PASS: 201/202 are together "
        "in one CV fold."
    )


else:

    raise RuntimeError(
        "subject_201_202 is neither "
        "in development nor final test."
    )


# ============================================================
# 31. SAVE DEVELOPMENT WITH CV
# ============================================================

development_cv_file = (
    OUTPUT_DIR
    / "five_class_development_with_cv_folds.csv"
)

development_df.to_csv(
    development_cv_file,
    index=False
)

print(
    "\nSaved development dataset with CV folds:"
)

print(
    development_cv_file
)


# ============================================================
# 32. SAVE CV SUMMARY
# ============================================================

fold_summary_df = pd.DataFrame(
    fold_summary_records
)

fold_summary_file = (
    OUTPUT_DIR
    / "five_class_cv_fold_summary.csv"
)

fold_summary_df.to_csv(
    fold_summary_file,
    index=False
)

print(
    "\nSaved CV fold summary:"
)

print(
    fold_summary_file
)


# ============================================================
# 33. SAVE GROUP ASSIGNMENT
# ============================================================

group_assignment_records = []

for group_id in all_groups:

    if group_id in test_groups:

        split = "test"
        cv_fold = ""

    else:

        split = "development"

        matching = (
            development_df.loc[
                development_df["group_id"]
                == group_id,
                "cv_fold"
            ]
            .unique()
        )

        if len(matching) != 1:

            raise RuntimeError(
                f"Could not determine "
                f"unique CV fold for {group_id}"
            )

        cv_fold = int(
            matching[0]
        )

    matching_rows = df_clean[
        df_clean["group_id"]
        == group_id
    ]

    records = sorted(
        matching_rows["record_id"]
        .unique()
    )

    group_assignment_records.append({
        "group_id": group_id,
        "records": ",".join(records),
        "split": split,
        "cv_fold": cv_fold,
        "n_windows": len(
            matching_rows
        ),
    })

group_assignment_df = pd.DataFrame(
    group_assignment_records
)

group_assignment_file = (
    OUTPUT_DIR
    / "five_class_group_assignment.csv"
)

group_assignment_df.to_csv(
    group_assignment_file,
    index=False
)

print(
    "\nSaved group assignment:"
)

print(
    group_assignment_file
)


# ============================================================
# 34. SAVE CHANNEL SELECTION
# ============================================================

channel_selection_file = (
    OUTPUT_DIR
    / "primary_channel_selection.csv"
)

channel_selection_df.to_csv(
    channel_selection_file,
    index=False
)

print(
    "\nSaved primary-channel selection:"
)

print(
    channel_selection_file
)


# ============================================================
# 35. SAVE CLASS DISTRIBUTION
# ============================================================

distribution_records = []

for split_name, split_df in [
    ("clean_all", df_clean),
    ("development", development_df),
    ("test", test_df),
]:

    counts = class_distribution(
        split_df
    )

    total = len(split_df)

    for class_name in TARGET_CLASSES:

        count = int(
            counts[class_name]
        )

        distribution_records.append({
            "split": split_name,
            "target_class": class_name,
            "count": count,
            "percentage": (
                count / total * 100
                if total > 0
                else 0
            ),
        })

distribution_df = pd.DataFrame(
    distribution_records
)

distribution_file = (
    OUTPUT_DIR
    / "five_class_distribution.csv"
)

distribution_df.to_csv(
    distribution_file,
    index=False
)

print(
    "\nSaved class distribution:"
)

print(
    distribution_file
)


# ============================================================
# 36. SAVE REPORT
# ============================================================

report_file = (
    OUTPUT_DIR
    / "step8b_dataset_report.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 8B - FIVE-CLASS WINDOW DATASET REPORT\n"
    )

    f.write(
        "=" * 60
        + "\n\n"
    )

    f.write(
        f"Input file:\n{INPUT_FILE}\n\n"
    )

    f.write(
        f"Detected class column:\n"
        f"{CLASS_COLUMN}\n\n"
    )

    f.write(
        f"Clean dataset:\n"
        f"{clean_file}\n\n"
    )

    f.write(
        f"Development dataset:\n"
        f"{development_file}\n\n"
    )

    f.write(
        f"Final test dataset:\n"
        f"{test_file}\n\n"
    )

    f.write(
        "IMPORTANT:\n"
    )

    f.write(
        "The old beat-classification final_test.csv "
        "was NOT used.\n"
    )

    f.write(
        "A separate five-class final test dataset "
        "was created.\n\n"
    )

    f.write(
        "FINAL STATUS\n"
    )

    f.write(
        "-" * 60
        + "\n"
    )

    f.write(
        "PASS - uncertain/mixed windows removed.\n"
    )

    f.write(
        "PASS - one primary channel per record.\n"
    )

    f.write(
        "PASS - 201/202 kept together.\n"
    )

    f.write(
        "PASS - no development/test group leakage.\n"
    )

    f.write(
        "PASS - all five classes in development.\n"
    )

    f.write(
        "PASS - all five classes in final test.\n"
    )

    f.write(
        "PASS - every development group has one CV fold.\n"
    )

    f.write(
        "PASS - old final_test.csv untouched.\n"
    )


# ============================================================
# 37. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("STEP 8B COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    "\nDetected class column:",
    CLASS_COLUMN
)

print(
    "Clean five-class windows:",
    len(df_clean)
)

print(
    "Development windows:",
    len(development_df)
)

print(
    "Final test windows:",
    len(test_df)
)

print(
    "Independent groups:",
    len(all_groups)
)

print(
    "Development groups:",
    len(development_groups)
)

print(
    "Final test groups:",
    len(test_groups)
)

print(
    "\n201/202 status:"
)

if subject_in_test:

    print(
        "  subject_201_202 correctly held "
        "together in FINAL TEST."
    )

else:

    print(
        "  subject_201_202 correctly held "
        "together in DEVELOPMENT."
    )

print(
    "\nOld final_test.csv:"
)

print(
    "  NOT USED"
)

print(
    "  NOT MODIFIED"
)

print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)

print(
    "\nReport:"
)

print(
    report_file
)

print(
    "\n" + "=" * 70
)
print(
    "READY FOR THE NEXT STEP"
)
print(
    "=" * 70
)