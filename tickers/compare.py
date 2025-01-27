from tqdm import tqdm
import os
import pandas as pd
import json
import argparse
from typing import Dict, List, Set


def list_csv_files(folder_path: str) -> Set[str]:
    """
    Lists all CSV files in the given folder.

    Args:
        folder_path (str): Path to the folder.

    Returns:
        Set[str]: A set of CSV file names.
    """
    return set([f for f in os.listdir(folder_path) if f.lower().endswith(".csv")])


def read_csv_with_date(file_path: str) -> pd.DataFrame:
    """
    Reads a CSV file into a DataFrame, parsing the 'date' or 'datetime' column as datetime.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame with 'date' as a standardized column.
    """
    possible_date_columns = ["date", "datetime"]
    try:
        # Read the CSV without parsing dates initially
        df = pd.read_csv(file_path)

        # Identify which date column exists
        date_column = None
        for col in possible_date_columns:
            if col in df.columns:
                date_column = col
                break

        if date_column is None:
            raise ValueError(f"Missing 'date' or 'datetime' column in {file_path}.")

        # Parse the identified date column
        df[date_column] = pd.to_datetime(df[date_column], errors="raise")

        # Rename the column to 'date' for consistency
        if date_column != "date":
            df = df.rename(columns={date_column: "date"})

        return df
    except Exception as e:
        raise ValueError(f"Error reading {file_path}: {e}")


def compare_columns_aligned(
    raw_df: pd.DataFrame, preprocessed_df: pd.DataFrame
) -> Dict:
    """
    Compares columns with the same names in two aligned DataFrames based on 'date'.

    Args:
        raw_df (pd.DataFrame): DataFrame from raw data.
        preprocessed_df (pd.DataFrame): DataFrame from preprocessed data.

    Returns:
        Dict: A dictionary containing differences per column and NaN counts.
    """
    differences = {}

    # Identify common columns excluding 'date'
    common_columns = raw_df.columns.intersection(preprocessed_df.columns).difference(
        ["date"]
    )

    # Merge DataFrames on 'date'
    merged_df = pd.merge(
        raw_df,
        preprocessed_df,
        on="date",
        how="inner",
        suffixes=("_raw", "_preprocessed"),
    )

    for column in common_columns:
        raw_col = merged_df[f"{column}_raw"]
        preprocessed_col = merged_df[f"{column}_preprocessed"]

        # Count NaNs
        raw_nans = raw_col.isna().sum()
        preprocessed_nans = preprocessed_col.isna().sum()

        # Compare values where neither is NaN
        comparison_mask = raw_col.notna() & preprocessed_col.notna()
        unequal_mask = comparison_mask & (raw_col != preprocessed_col)

        differing_dates = (
            merged_df.loc[unequal_mask, "date"]
            .dt.strftime("%Y-%m-%d %H:%M:%S")
            .tolist()
        )
        raw_values = raw_col.loc[unequal_mask].tolist()
        preprocessed_values = preprocessed_col.loc[unequal_mask].tolist()

        if unequal_mask.any() or raw_nans > 0 or preprocessed_nans > 0:
            differences[column] = {}
            if raw_nans > 0 or preprocessed_nans > 0:
                differences[column]["nan_counts"] = {
                    "raw": int(raw_nans),
                    "preprocessed": int(preprocessed_nans),
                }
            if unequal_mask.any():
                differences[column]["differences"] = []
                for date, raw_val, pre_val in zip(
                    differing_dates, raw_values, preprocessed_values
                ):
                    differences[column]["differences"].append(
                        {"date": date, "raw": raw_val, "preprocessed": pre_val}
                    )

    return differences


def compare_csv_files_aligned(raw_file_path: str, preprocessed_file_path: str) -> Dict:
    """
    Compares two CSV files aligned on the 'date' column and identifies differences in common columns.

    Args:
        raw_file_path (str): Path to the raw CSV file.
        preprocessed_file_path (str): Path to the preprocessed CSV file.

    Returns:
        Dict: A dictionary containing differences per column and NaN counts.
    """
    try:
        raw_df = read_csv_with_date(raw_file_path)
    except ValueError as e:
        return {"error": str(e)}

    try:
        preprocessed_df = read_csv_with_date(preprocessed_file_path)
    except ValueError as e:
        return {"error": str(e)}

    differences = compare_columns_aligned(raw_df, preprocessed_df)
    return differences


def generate_report_aligned(
    raw_folder: str,
    preprocessed_folder: str,
    output_file: str = "comparison_report_aligned.json",
) -> None:
    """
    Generates a JSON report comparing CSV files in two folders aligned on the 'date' column.

    Args:
        raw_folder (str): Path to the raw data folder.
        preprocessed_folder (str): Path to the preprocessed data folder.
        output_file (str, optional): Name of the output JSON file. Defaults to 'comparison_report_aligned.json'.
    """
    # List all CSV files in both folders
    raw_files = list_csv_files(raw_folder)
    preprocessed_files = list_csv_files(preprocessed_folder)

    # Identify paired and unpaired files
    paired_files = raw_files.intersection(preprocessed_files)
    unpaired_raw = raw_files - preprocessed_files
    unpaired_preprocessed = preprocessed_files - raw_files

    # Initialize report dictionary
    report = {
        "unpaired_files": {
            "raw_data_folder": sorted(list(unpaired_raw)),
            "preprocessed_data_folder": sorted(list(unpaired_preprocessed)),
        },
        "differences_in_paired_files": {},
    }

    # Compare each paired file
    for file_name in tqdm(sorted(paired_files)):
        raw_file_path = os.path.join(raw_folder, file_name)
        preprocessed_file_path = os.path.join(preprocessed_folder, file_name)

        diffs = compare_csv_files_aligned(raw_file_path, preprocessed_file_path)
        if diffs:
            report["differences_in_paired_files"][file_name] = diffs

    # Save the report to a JSON file
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        print(f"Comparison report has been generated: {output_file}")
    except Exception as e:
        print(f"Failed to write the report to {output_file}: {e}")


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Compare CSV files in two folders aligned on 'date', identify unpaired files, and find differences in paired files."
    )
    parser.add_argument(
        "raw_folder", type=str, help="Path to the folder containing raw CSV files."
    )
    parser.add_argument(
        "preprocessed_folder",
        type=str,
        help="Path to the folder containing preprocessed CSV files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="comparison_report_aligned.json",
        help='Name of the output JSON report file. Default is "comparison_report_aligned.json".',
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    raw_folder = args.raw_folder
    preprocessed_folder = args.preprocessed_folder
    output_file = args.output

    # Validate folder paths
    if not os.path.isdir(raw_folder):
        print(f"Error: Raw data folder does not exist: {raw_folder}")
        return
    if not os.path.isdir(preprocessed_folder):
        print(f"Error: Preprocessed data folder does not exist: {preprocessed_folder}")
        return

    generate_report_aligned(raw_folder, preprocessed_folder, output_file)


if __name__ == "__main__":
    main()
