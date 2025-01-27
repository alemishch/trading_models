import os
import pandas as pd


def get_missing_intervals(file_path):
    """
    Returns a list of (start_gap, end_gap) tuples indicating gaps > 1 minute.
    Looks for a column named 'datetime' or 'date'.
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

    # Determine which column to use for datetime
    if "datetime" in df.columns:
        dt_col = "datetime"
    elif "date" in df.columns:
        dt_col = "date"
    else:
        # No valid column, so no intervals to report
        return []

    # Convert to datetime
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    # Drop rows with invalid timestamps
    df = df.dropna(subset=[dt_col])

    if df.empty:
        return []

    # Sort by datetime
    df.sort_values(by=dt_col, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Calculate diffs between consecutive timestamps
    df["diff"] = df[dt_col].diff()

    # Find rows where diff > 1 minute
    missing_threshold = pd.Timedelta(minutes=1)
    gap_indices = df.index[df["diff"] > missing_threshold]

    # Build a list of gap intervals
    missing_intervals = []
    for idx in gap_indices:
        # Gap "start" is the timestamp in the previous row
        prev_idx = idx - 1
        if prev_idx >= 0:
            start_gap = df.loc[prev_idx, dt_col]
            end_gap = df.loc[idx, dt_col]
            missing_intervals.append((start_gap, end_gap))

    return missing_intervals


def compare_folders(folder1, folder2):
    """
    Compares two folders of CSV files. For each common file name, checks whether
    the missing 1-minute intervals are identical. Prints a summary of findings.
    """
    # Collect CSV files in each folder
    files1 = {f for f in os.listdir(folder1) if f.lower().endswith(".csv")}
    files2 = {f for f in os.listdir(folder2) if f.lower().endswith(".csv")}

    # Identify files in both folders
    common_files = files1 & files2
    only_in_1 = files1 - files2
    only_in_2 = files2 - files1

    # Report files that are unpaired
    if only_in_1:
        print("Files only in folder1:")
        for f in sorted(only_in_1):
            print(f"  {f}")
    if only_in_2:
        print("Files only in folder2:")
        for f in sorted(only_in_2):
            print(f"  {f}")

    if not common_files:
        print("\nNo files in common between the two folders.")
        return

    # Compare missing intervals for each common file
    print("\nComparing common files...")
    for filename in sorted(common_files):
        file_path_1 = os.path.join(folder1, filename)
        file_path_2 = os.path.join(folder2, filename)

        intervals1 = get_missing_intervals(file_path_1)
        intervals2 = get_missing_intervals(file_path_2)

        # You could compare directly, but note that direct equality means
        # exact matching in order and number of intervals. If you just want
        # to see if the sets of intervals are identical, convert them to sets:
        # However, tuples might differ if there's a microsecond difference, etc.
        # For a strict check, let's do direct list comparison:
        if intervals1 == intervals2:
            print(f"  {filename}: MISSING INTERVALS ARE IDENTICAL.")
        else:
            print(f"  {filename}: DIFFERENCES FOUND in missing intervals.")
            set1 = set(intervals1)
            set2 = set(intervals2)

            # Find intervals unique to each file
            unique_to_1 = set1 - set2
            unique_to_2 = set2 - set1

            # Calculate total difference duration
            total_diff_duration = pd.Timedelta(0)

            for gap in unique_to_1.union(unique_to_2):
                start_gap, end_gap = gap
                total_diff_duration += end_gap - start_gap

            # Convert total duration to a readable format (e.g., seconds)
            total_diff_seconds = total_diff_duration.total_seconds()

            print(
                f"    Total difference in missing intervals duration: {total_diff_seconds} seconds"
            )

            # # (Optional) Print unique gaps for more detail
            # if unique_to_1:
            #     print(f"    Missing in folder1 but not in folder2:")
            #     for gap in unique_to_1:
            #         print(f"      Gap from {gap[0]} to {gap[1]} ({gap[1] - gap[0]})")
            # if unique_to_2:
            #     print(f"    Missing in folder2 but not in folder1:")
            #     for gap in unique_to_2:
            #         print(f"      Gap from {gap[0]} to {gap[1]} ({gap[1] - gap[0]})")


if __name__ == "__main__":
    folder1 = "tmp"  # change to your actual folder1 path
    folder2 = "mr_regimes_download"  # change to your actual folder2 path

    compare_folders(folder1, folder2)
