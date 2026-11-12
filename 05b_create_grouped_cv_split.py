# ============================================================
# STEP 5B: GROUPED CROSS-VALIDATION + FINAL TEST SPLIT
# MIT-BIH Arrhythmia Database
#
# Purpose:
#   1. Group beats by ECG recording / subject
#   2. Keep records 201 and 202 together
#   3. Create a leakage-safe final test set
#   4. Use grouped 5-fold cross-validation on development data
#   5. Verify that no recording/subject crosses splits
#
# IMPORTANT:
#   Individual beats are NEVER randomly split.
# ============================================================

import os
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold


# ============================================================
# 1. PATHS
# ============================================================

MASTER_CSV = (
    r"c:\Users\Himanshu Mohapatra\OneDrive\Desktop"
    r"\Project_Main\master_MITBIH_48_records_features.csv"
)

OUTPUT_DIR = (
    r"c:\Users\Himanshu Mohapatra\OneDrive\Desktop"
    r"\Project_Main\ml_grouped_cv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. SETTINGS
# ============================================================

RANDOM_STATE = 42

# Approximately 20% of the 47 independent groups.
TEST_GROUP_FRACTION = 0.20

# Number of grouped CV folds.
N_CV_FOLDS = 5

# Number of random candidate test-group selections.
N_RANDOM_CANDIDATES = 20000


# ============================================================
# 3. LOAD MASTER DATASET
# ============================================================

print("=" * 75)
print("STEP 5B: GROUPED CROSS-VALIDATION + FINAL TEST SPLIT")
print("=" * 75)

print("\nLoading master dataset...")

df = pd.read_csv(MASTER_CSV)

print(f"Dataset shape: {df.shape}")
print(f"Records:       {df['record_id'].nunique()}")
print(f"Beat classes:  {df['beat_label'].nunique()}")


# ============================================================
# 4. CREATE SUBJECT / RECORD GROUPS
# ============================================================

# Each recording is its own group.
df["group_id"] = df["record_id"].astype(str)

# 201 and 202 are from the same subject.
df.loc[
    df["record_id"].isin([201, 202]),
    "group_id"
] = "subject_201_202"


print("\nGrouping:")

n_groups = df["group_id"].nunique()

print(f"Independent groups: {n_groups}")

if n_groups != 47:
    print(
        "WARNING: Expected 47 groups based on the current dataset."
    )


# ============================================================
# 5. CREATE GROUP-LEVEL SUMMARY
# ============================================================

classes = sorted(df["beat_label"].unique())

group_class_counts = pd.crosstab(
    df["group_id"],
    df["beat_label"]
)

# Make sure every class is represented as a column.
group_class_counts = group_class_counts.reindex(
    columns=classes,
    fill_value=0
)

group_row_counts = (
    df.groupby("group_id")
    .size()
    .rename("total_rows")
)

group_summary = group_class_counts.copy()
group_summary["total_rows"] = group_row_counts

group_summary = group_summary.reset_index()


# ============================================================
# 6. CHECK CLASS DISTRIBUTION ACROSS GROUPS
# ============================================================

print("\n" + "=" * 75)
print("CLASS DISTRIBUTION ACROSS GROUPS")
print("=" * 75)

class_group_counts = {}

for label in classes:

    n_groups_with_class = int(
        (group_class_counts[label] > 0).sum()
    )

    total_beats = int(
        group_class_counts[label].sum()
    )

    class_group_counts[label] = n_groups_with_class

    print(
        f"{str(label):>3} : "
        f"{total_beats:>7,} beats "
        f"in {n_groups_with_class:>2} groups"
    )


# ============================================================
# 7. IDENTIFY CLASSES THAT CANNOT BE HELD OUT
# ============================================================
#
# If a class exists in only ONE independent recording group,
# putting that group into the final test set would leave ZERO
# examples of that class for training.
#
# Therefore such groups are protected from the test set.
# ============================================================

protected_classes = [
    label
    for label, count in class_group_counts.items()
    if count < 2
]

print("\nClasses occurring in only one group:")

if protected_classes:
    for label in protected_classes:
        print(
            f"  {label} -> protected from final test split"
        )
else:
    print("  None")


# ============================================================
# 8. GROUP LIST
# ============================================================

all_groups = sorted(
    df["group_id"].unique()
)

protected_groups = set()

for label in protected_classes:

    groups_for_class = set(
        group_class_counts.index[
            group_class_counts[label] > 0
        ]
    )

    protected_groups.update(groups_for_class)


candidate_test_groups = [
    g for g in all_groups
    if g not in protected_groups
]

print("\nGroup selection:")
print(f"Total groups:             {len(all_groups)}")
print(f"Protected groups:         {len(protected_groups)}")
print(f"Candidate test groups:    {len(candidate_test_groups)}")


# ============================================================
# 9. TARGET TEST SIZE
# ============================================================

target_test_groups = max(
    1,
    round(len(all_groups) * TEST_GROUP_FRACTION)
)

print(
    f"\nTarget final test groups: "
    f"{target_test_groups}"
)


# ============================================================
# 10. TOTAL DATASET CLASS DISTRIBUTION
# ============================================================

total_class_counts = (
    df["beat_label"]
    .value_counts()
    .reindex(classes, fill_value=0)
)

total_rows = len(df)


# ============================================================
# 11. TEST SPLIT SCORING FUNCTION
# ============================================================
#
# We do NOT simply choose random groups.
#
# Candidate test sets are scored according to:
#
#   1. Test size close to 20%
#   2. Class distribution close to the full dataset
#   3. Prefer coverage of classes occurring in multiple groups
#   4. Avoid putting all examples of a class into test
#
# This is a practical group-level optimization.
# ============================================================

def score_test_groups(test_groups):

    test_groups = set(test_groups)

    test_df_mask = df["group_id"].isin(test_groups)

    test_counts = (
        df.loc[test_df_mask, "beat_label"]
        .value_counts()
        .reindex(classes, fill_value=0)
    )

    test_rows = int(test_counts.sum())

    # --------------------------------------------------------
    # 1. Test row fraction error
    # --------------------------------------------------------

    target_fraction = TEST_GROUP_FRACTION

    actual_fraction = test_rows / total_rows

    size_error = abs(
        actual_fraction - target_fraction
    )

    # --------------------------------------------------------
    # 2. Class-distribution error
    # --------------------------------------------------------

    total_proportions = (
        total_class_counts / total_rows
    )

    if test_rows > 0:
        test_proportions = (
            test_counts / test_rows
        )
    else:
        test_proportions = total_proportions * 0

    distribution_error = np.mean(
        np.abs(
            test_proportions.values
            - total_proportions.values
        )
    )

    # --------------------------------------------------------
    # 3. Rare-class coverage
    # --------------------------------------------------------
    #
    # Classes with more than one group are desirable in both
    # development and test.
    # --------------------------------------------------------

    coverage_penalty = 0.0

    for label in classes:

        groups_available = class_group_counts[label]

        if groups_available >= 2:

            in_test = (
                test_counts[label] > 0
            )

            in_development = (
                total_class_counts[label]
                - test_counts[label]
            ) > 0

            # Penalize if class appears in only one side.
            if not in_test:
                coverage_penalty += 0.002

            if not in_development:
                coverage_penalty += 0.020

    # --------------------------------------------------------
    # 4. Group-count error
    # --------------------------------------------------------

    group_count_error = abs(
        len(test_groups)
        - target_test_groups
    ) / len(all_groups)

    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------

    score = (
        3.0 * size_error
        + 5.0 * distribution_error
        + 10.0 * coverage_penalty
        + 1.0 * group_count_error
    )

    return score


# ============================================================
# 12. RANDOMIZED GROUP SEARCH
# ============================================================

print("\n" + "=" * 75)
print("SEARCHING FOR FINAL TEST GROUPS")
print("=" * 75)

rng = random.Random(RANDOM_STATE)

best_score = float("inf")
best_test_groups = None

for iteration in range(N_RANDOM_CANDIDATES):

    if len(candidate_test_groups) < target_test_groups:
        raise RuntimeError(
            "Not enough candidate groups to create test set."
        )

    selected = rng.sample(
        candidate_test_groups,
        target_test_groups
    )

    score = score_test_groups(selected)

    if score < best_score:

        best_score = score
        best_test_groups = sorted(selected)


print("\nBest test-group selection found.")

print(f"Best score: {best_score:.6f}")

print("\nFINAL TEST GROUPS:")

for group in best_test_groups:
    print(f"  {group}")


# ============================================================
# 13. DEVELOPMENT GROUPS
# ============================================================

test_groups = set(best_test_groups)

development_groups = [
    g for g in all_groups
    if g not in test_groups
]

development_groups = sorted(
    development_groups
)

print("\nDEVELOPMENT GROUPS:")

for group in development_groups:
    print(f"  {group}")


# ============================================================
# 14. CREATE FINAL TEST / DEVELOPMENT DATASETS
# ============================================================

test_mask = df["group_id"].isin(test_groups)
development_mask = df["group_id"].isin(
    development_groups
)

test_df = df.loc[test_mask].copy()
development_df = df.loc[development_mask].copy()


# ============================================================
# 15. VERIFY GROUP LEAKAGE
# ============================================================

test_group_set = set(
    test_df["group_id"].unique()
)

development_group_set = set(
    development_df["group_id"].unique()
)

overlap = (
    test_group_set
    .intersection(development_group_set)
)

print("\n" + "=" * 75)
print("DEVELOPMENT / TEST LEAKAGE CHECK")
print("=" * 75)

print(
    f"Development groups: {len(development_group_set)}"
)

print(
    f"Test groups:        {len(test_group_set)}"
)

print(
    f"Overlapping groups: {len(overlap)}"
)

if overlap:
    print("ERROR: Leakage detected!")
    print(overlap)
    raise RuntimeError(
        "Development/test group leakage detected."
    )

print("PASS: No development/test group overlap.")


# ============================================================
# 16. CHECK 201 / 202
# ============================================================

print("\nChecking records 201 and 202...")

test_records = set(
    test_df["record_id"].unique()
)

development_records = set(
    development_df["record_id"].unique()
)

if (
    (201 in test_records)
    != (202 in test_records)
):
    raise RuntimeError(
        "201 and 202 were separated in test."
    )

if (
    (201 in development_records)
    != (202 in development_records)
):
    raise RuntimeError(
        "201 and 202 were separated in development."
    )

if 201 in test_records:
    print("201 + 202 -> TEST")
else:
    print("201 + 202 -> DEVELOPMENT")

print("PASS: 201/202 subject grouping preserved.")


# ============================================================
# 17. PRINT FINAL TEST DISTRIBUTION
# ============================================================

def print_distribution(
    name,
    data
):

    print("\n" + "-" * 75)
    print(name)

    print(
        f"Rows:   {len(data):,}"
    )

    print(
        f"Groups: {data['group_id'].nunique()}"
    )

    print(
        f"Records:{data['record_id'].nunique()}"
    )

    counts = (
        data["beat_label"]
        .value_counts()
        .reindex(classes, fill_value=0)
    )

    percentages = (
        counts / len(data) * 100
    )

    summary = pd.DataFrame({
        "count": counts,
        "percentage": percentages.round(3)
    })

    print(
        summary.to_string()
    )

    return summary


development_distribution = print_distribution(
    "DEVELOPMENT DATASET",
    development_df
)

test_distribution = print_distribution(
    "FINAL TEST DATASET",
    test_df
)


# ============================================================
# 18. CREATE GROUPED STRATIFIED CV
# ============================================================
#
# IMPORTANT:
#
# Cross-validation is performed ONLY on the development
# groups.
#
# The final test groups are never included.
# ============================================================

print("\n" + "=" * 75)
print("CREATING GROUPED CROSS-VALIDATION FOLDS")
print("=" * 75)

development_X = development_df[
    ["RR_interval_ms"]
]

development_y = development_df[
    "beat_label"
]

development_groups_series = development_df[
    "group_id"
]

cv = StratifiedGroupKFold(
    n_splits=N_CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)

# Store fold assignment for every development row.
development_with_fold = development_df.copy()

development_with_fold["cv_fold"] = -1

fold_summaries = []

for fold_number, (
    train_relative_idx,
    validation_relative_idx
) in enumerate(
    cv.split(
        development_X,
        development_y,
        groups=development_groups_series
    ),
    start=1
):

    train_indices = (
        development_df.index[
            train_relative_idx
        ]
    )

    validation_indices = (
        development_df.index[
            validation_relative_idx
        ]
    )

    development_with_fold.loc[
        validation_indices,
        "cv_fold"
    ] = fold_number

    fold_train = development_df.loc[
        train_indices
    ]

    fold_validation = development_df.loc[
        validation_indices
    ]

    train_groups_fold = set(
        fold_train["group_id"]
    )

    validation_groups_fold = set(
        fold_validation["group_id"]
    )

    fold_overlap = (
        train_groups_fold
        .intersection(validation_groups_fold)
    )

    if fold_overlap:
        raise RuntimeError(
            f"Leakage detected in CV fold {fold_number}: "
            f"{fold_overlap}"
        )

    # --------------------------------------------------------
    # Fold summary
    # --------------------------------------------------------

    row = {
        "fold": fold_number,
        "train_rows": len(fold_train),
        "validation_rows": len(fold_validation),
        "train_groups": len(train_groups_fold),
        "validation_groups": len(
            validation_groups_fold
        )
    }

    for label in classes:

        row[
            f"train_{label}"
        ] = int(
            (
                fold_train["beat_label"]
                == label
            ).sum()
        )

        row[
            f"validation_{label}"
        ] = int(
            (
                fold_validation["beat_label"]
                == label
            ).sum()
        )

    fold_summaries.append(row)

    print(
        f"\nFold {fold_number}"
    )

    print(
        f"  Train groups:      "
        f"{len(train_groups_fold)}"
    )

    print(
        f"  Validation groups: "
        f"{len(validation_groups_fold)}"
    )

    print(
        f"  Train rows:        "
        f"{len(fold_train):,}"
    )

    print(
        f"  Validation rows:   "
        f"{len(fold_validation):,}"
    )

    print(
        f"  Group overlap:     "
        f"{len(fold_overlap)}"
    )


# ============================================================
# 19. VERIFY EVERY DEVELOPMENT ROW HAS A FOLD
# ============================================================

unassigned = (
    development_with_fold["cv_fold"] == -1
).sum()

if unassigned > 0:

    raise RuntimeError(
        f"{unassigned} development rows "
        f"were not assigned a CV fold."
    )

print(
    "\nPASS: Every development row assigned "
    "to a validation fold."
)


# ============================================================
# 20. CHECK THAT EACH GROUP HAS ONLY ONE FOLD
# ============================================================

group_fold_counts = (
    development_with_fold
    .groupby("group_id")["cv_fold"]
    .nunique()
)

bad_groups = group_fold_counts[
    group_fold_counts != 1
]

if len(bad_groups) > 0:

    print(
        "ERROR: A group appears in multiple "
        "validation folds."
    )

    print(bad_groups)

    raise RuntimeError(
        "Group appears in multiple CV folds."
    )

print(
    "PASS: Every development group belongs "
    "to exactly one validation fold."
)


# ============================================================
# 21. SAVE FOLD SUMMARY
# ============================================================

fold_summary_df = pd.DataFrame(
    fold_summaries
)

fold_summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "cv_fold_summary.csv"
    ),
    index=False
)


# ============================================================
# 22. SAVE TEST / DEVELOPMENT DATA
# ============================================================

test_df = test_df.sort_values(
    [
        "record_id",
        "channel_index",
        "beat_index"
    ]
)

development_with_fold = (
    development_with_fold
    .sort_values(
        [
            "cv_fold",
            "record_id",
            "channel_index",
            "beat_index"
        ]
    )
)

test_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_test.csv"
    ),
    index=False
)

development_with_fold.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "development_with_cv_folds.csv"
    ),
    index=False
)


# ============================================================
# 23. SAVE GROUP ASSIGNMENT MANIFEST
# ============================================================

group_assignment_rows = []

for group in all_groups:

    if group in test_groups:
        split = "FINAL_TEST"
    else:
        split = "DEVELOPMENT"

    group_assignment_rows.append({
        "group_id": group,
        "split": split,
        "records": ",".join(
            sorted(
                df.loc[
                    df["group_id"] == group,
                    "record_id"
                ]
                .astype(str)
                .unique()
            )
        ),
        "rows": int(
            (
                df["group_id"] == group
            ).sum()
        )
    })

group_assignment_df = pd.DataFrame(
    group_assignment_rows
)

group_assignment_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "group_assignment.csv"
    ),
    index=False
)


# ============================================================
# 24. SAVE CV GROUP MANIFEST
# ============================================================

cv_group_rows = []

for fold in range(
    1,
    N_CV_FOLDS + 1
):

    validation_groups = sorted(
        development_with_fold.loc[
            development_with_fold["cv_fold"]
            == fold,
            "group_id"
        ].unique()
    )

    training_groups = sorted(
        set(development_groups)
        - set(validation_groups)
    )

    cv_group_rows.append({
        "fold": fold,
        "training_groups": ",".join(
            training_groups
        ),
        "validation_groups": ",".join(
            validation_groups
        )
    })

cv_group_manifest_df = pd.DataFrame(
    cv_group_rows
)

cv_group_manifest_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "cv_group_manifest.csv"
    ),
    index=False
)


# ============================================================
# 25. FINAL LEAKAGE REPORT
# ============================================================

print("\n" + "=" * 75)
print("FINAL LEAKAGE REPORT")
print("=" * 75)

print(
    "FINAL TEST vs DEVELOPMENT:"
)

print(
    f"  Group overlap = "
    f"{len(test_group_set.intersection(development_group_set))}"
)

for fold in range(
    1,
    N_CV_FOLDS + 1
):

    validation_groups = set(
        development_with_fold.loc[
            development_with_fold["cv_fold"]
            == fold,
            "group_id"
        ]
    )

    training_groups = (
        set(development_groups)
        - validation_groups
    )

    overlap = (
        training_groups
        .intersection(validation_groups)
    )

    print(
        f"  CV Fold {fold}: "
        f"group overlap = {len(overlap)}"
    )

print(
    "\nAll leakage checks completed."
)


# ============================================================
# 26. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 75)
print("STEP 5B COMPLETE")
print("=" * 75)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nFiles created:")

for filename in sorted(
    os.listdir(OUTPUT_DIR)
):
    print(f"  - {filename}")

print("\nFinal test groups:")
for group in sorted(test_groups):
    print(f"  {group}")

print(
    f"\nFinal test rows: "
    f"{len(test_df):,}"
)

print(
    f"Development rows: "
    f"{len(development_df):,}"
)

print(
    f"\nDevelopment CV folds: "
    f"{N_CV_FOLDS}"
)

print(
    "\nIMPORTANT:"
)

print(
    "  - Entire recordings remain together."
)

print(
    "  - Records 201 and 202 remain together."
)

print(
    "  - Final test groups are never used "
    "in development CV."
)

print(
    "  - Every development group belongs "
    "to exactly one validation fold."
)

print(
    "  - No individual beat is randomly "
    "split across datasets."
)

print("=" * 75)