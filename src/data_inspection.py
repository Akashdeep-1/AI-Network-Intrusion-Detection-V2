from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CICIoT2023 DATA INSPECTION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ciciot2023"
    / "Merged_CSV"
)

OUTPUT_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def inspect_first_file():
    """Inspect the structure of the first CSV file."""

    first_file = DATA_DIR / "Merged01.csv"

    print("\n" + "=" * 70)
    print("CICIoT2023 DATASET INSPECTION")
    print("=" * 70)

    print(f"\nDataset directory:")
    print(DATA_DIR)

    print(f"\nInspecting:")
    print(first_file)

    if not first_file.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {first_file}"
        )

    # Read only a small sample
    df = pd.read_csv(first_file, nrows=10000)

    print("\n" + "-" * 70)
    print("BASIC INFORMATION")
    print("-" * 70)

    print(f"Rows sampled       : {len(df):,}")
    print(f"Columns            : {len(df.columns)}")
    print(f"Memory usage       : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print("\nColumn names:")
    for i, column in enumerate(df.columns, start=1):
        print(f"{i:3}. {column}")

    print("\n" + "-" * 70)
    print("DATA TYPES")
    print("-" * 70)

    print(df.dtypes.to_string())

    print("\n" + "-" * 70)
    print("FIRST 5 ROWS")
    print("-" * 70)

    print(df.head().to_string())

    print("\n" + "-" * 70)
    print("MISSING VALUES")
    print("-" * 70)

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing) == 0:
        print("No missing values found in the 10,000-row sample.")
    else:
        print(missing.to_string())

    print("\n" + "-" * 70)
    print("INFINITE VALUES")
    print("-" * 70)

    numeric_df = df.select_dtypes(include=[np.number])

    infinite_count = np.isinf(numeric_df).sum()
    infinite_count = infinite_count[infinite_count > 0].sort_values(
        ascending=False
    )

    if len(infinite_count) == 0:
        print("No infinite values found in the 10,000-row sample.")
    else:
        print(infinite_count.to_string())

    print("\n" + "-" * 70)
    print("DUPLICATES")
    print("-" * 70)

    duplicate_count = df.duplicated().sum()

    print(f"Duplicate rows in sample: {duplicate_count:,}")

    print("\n" + "-" * 70)
    print("POSSIBLE LABEL COLUMNS")
    print("-" * 70)

    for column in df.columns:
        if (
            "label" in column.lower()
            or "attack" in column.lower()
            or "class" in column.lower()
        ):
            print(f"{column}")

    print("\n" + "-" * 70)
    print("UNIQUE VALUES OF CANDIDATE LABEL COLUMNS")
    print("-" * 70)

    for column in df.columns:
        if (
            "label" in column.lower()
            or "attack" in column.lower()
            or "class" in column.lower()
        ):
            print(f"\n[{column}]")
            print(df[column].value_counts(dropna=False).head(50).to_string())

    return df


def inspect_all_file_headers():
    """Verify all 63 CSV files have the same structure."""

    print("\n" + "=" * 70)
    print("CHECKING ALL CSV FILE HEADERS")
    print("=" * 70)

    files = sorted(DATA_DIR.glob("Merged*.csv"))

    print(f"\nCSV files found: {len(files)}")

    if len(files) != 63:
        print(
            f"WARNING: Expected 63 files, but found {len(files)}."
        )

    reference_columns = None
    inconsistent_files = []

    for file in files:

        header = pd.read_csv(file, nrows=0)

        columns = list(header.columns)

        if reference_columns is None:
            reference_columns = columns
        elif columns != reference_columns:
            inconsistent_files.append(file.name)

    if not inconsistent_files:
        print("\nAll files have identical column structures. ✓")
    else:
        print("\nFiles with different column structures:")
        for file in inconsistent_files:
            print(f"  - {file}")

    print(f"\nNumber of columns: {len(reference_columns)}")

    return reference_columns
def scan_complete_label_distribution():
    """
    Scan all 63 CSV files and calculate the complete
    individual attack-label distribution.
    """

    print("\n" + "=" * 70)
    print("FULL DATASET LABEL DISTRIBUTION")
    print("=" * 70)

    files = sorted(DATA_DIR.glob("Merged*.csv"))

    label_counts = {}
    total_rows = 0

    for index, file in enumerate(files, start=1):

        print(
            f"\rProcessing file {index}/{len(files)}: {file.name}",
            end="",
            flush=True
        )

        # Only load Label column
        for chunk in pd.read_csv(
            file,
            usecols=["Label"],
            chunksize=100_000
        ):

            chunk["Label"] = chunk["Label"].astype(str).str.strip()

            counts = chunk["Label"].value_counts()

            for label, count in counts.items():
                label_counts[label] = (
                    label_counts.get(label, 0) + int(count)
                )

            total_rows += len(chunk)

    print("\n")

    distribution = (
        pd.Series(label_counts, name="Count")
        .sort_values(ascending=False)
    )

    distribution_df = distribution.reset_index()
    distribution_df.columns = ["Label", "Count"]

    distribution_df["Percentage"] = (
        distribution_df["Count"] / total_rows * 100
    )

    print("-" * 70)
    print(f"TOTAL ROWS: {total_rows:,}")
    print(f"UNIQUE LABELS: {len(distribution_df)}")
    print("-" * 70)

    print(
        distribution_df.to_string(
            index=False,
            formatters={
                "Percentage": "{:.4f}%".format
            }
        )
    )

    # Save complete distribution
    output_file = OUTPUT_DIR / "ciciot2023_label_distribution.csv"

    distribution_df.to_csv(
        output_file,
        index=False
    )

    print("\nSaved distribution to:")
    print(output_file)

    return distribution_df

def main():

    # Inspect one file in detail
    inspect_first_file()

    # Verify all files
    inspect_all_file_headers()

    # Scan the complete dataset
    scan_complete_label_distribution()

    print("\n" + "=" * 70)
    print("FULL DATASET INSPECTION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()