from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(
    r"C:\Users\Himanshu Mohapatra\OneDrive\Desktop\Project_Main"
)

print("\n" + "=" * 75)
print("STEP 10A — LOCATE AND INSPECT FINAL TEST DATA")
print("=" * 75)

# Search project folder and subfolders for likely Step 8B/9B CSV files
csv_files = sorted(PROJECT_DIR.rglob("*.csv"))

keywords = [
    "test",
    "window",
    "morphology",
    "development",
    "clean",
    "split",
]

relevant_files = [
    path for path in csv_files
    if any(word in path.name.lower() for word in keywords)
]

if not relevant_files:
    print("\nNo likely CSV files found.")
    raise SystemExit

print(f"\nFound {len(relevant_files)} relevant CSV files:\n")

for path in relevant_files:
    print("-" * 75)
    print(f"FILE: {path}")

    try:
        df = pd.read_csv(path, nrows=5)

        print(f"Columns: {list(df.columns)}")

        # Read full file only to report its row count
        full_df = pd.read_csv(path)
        print(f"Shape: {full_df.shape}")

        possible_split_columns = [
            col for col in full_df.columns
            if any(term in col.lower() for term in [
                "split",
                "set",
                "partition",
                "test",
                "development",
                "fold",
            ])
        ]

        if possible_split_columns:
            print("\nPossible split/grouping columns:")
            for col in possible_split_columns:
                print(f"  {col}: {full_df[col].value_counts(dropna=False).to_dict()}")

    except Exception as e:
        print(f"Could not inspect file: {e}")

print("\n" + "=" * 75)
print("INSPECTION COMPLETE — NO MODEL WAS FIT")
print("=" * 75)