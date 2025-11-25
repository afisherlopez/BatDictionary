import pandas as pd

# Read the CSV file with FileID
input_file = "original_lake2_with_FileID.csv" #moved outside repo for space reasons
output_file = "Files_above_length_8.csv"

print(f"Reading {input_file}...")
df = pd.read_csv(input_file)

print(f"Original shape: {df.shape}")
print(f"Original number of rows: {len(df)}")
print(f"Original number of unique FileIDs: {df['FileID'].nunique()}")

# Count rows per FileID
rows_per_file = df.groupby('FileID').size()

print(f"\nRows per FileID statistics (before filtering):")
print(f"  Mean: {rows_per_file.mean():.2f}")
print(f"  Median: {rows_per_file.median():.2f}")
print(f"  Min: {rows_per_file.min()}")
print(f"  Max: {rows_per_file.max()}")

# Find FileIDs with at least 8 rows
file_ids_to_keep = rows_per_file[rows_per_file >= 8].index

print(f"\nFileIDs with >= 8 rows: {len(file_ids_to_keep)}")
print(f"FileIDs with < 8 rows (to be removed): {len(rows_per_file) - len(file_ids_to_keep)}")

# Filter dataframe to keep only FileIDs with at least 8 rows
df_filtered = df[df['FileID'].isin(file_ids_to_keep)]

print(f"\nFiltered shape: {df_filtered.shape}")
print(f"Filtered number of rows: {len(df_filtered)}")
print(f"Filtered number of unique FileIDs: {df_filtered['FileID'].nunique()}")
print(f"Rows removed: {len(df) - len(df_filtered)}")

# Verify minimum rows per FileID in filtered data
filtered_rows_per_file = df_filtered.groupby('FileID').size()
print(f"\nRows per FileID statistics (after filtering):")
print(f"  Mean: {filtered_rows_per_file.mean():.2f}")
print(f"  Median: {filtered_rows_per_file.median():.2f}")
print(f"  Min: {filtered_rows_per_file.min()}")
print(f"  Max: {filtered_rows_per_file.max()}")

# Save filtered data
print(f"\nSaving to {output_file}...")
df_filtered.to_csv(output_file, index=False)

print("Done!")
print(f"\nSummary:")
print(f"  Input file: {input_file}")
print(f"  Output file: {output_file}")
print(f"  Original rows: {len(df):,}")
print(f"  Filtered rows: {len(df_filtered):,}")
print(f"  Rows removed: {len(df) - len(df_filtered):,}")
print(f"  Original FileIDs: {df['FileID'].nunique()}")
print(f"  Filtered FileIDs: {df_filtered['FileID'].nunique()}")
print(f"  FileIDs removed: {df['FileID'].nunique() - df_filtered['FileID'].nunique()}")

