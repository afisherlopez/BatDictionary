import pandas as pd

# Read the CSV file
input_file = "data/Files_above_length_8.csv"
#output_file = "~/BatDictionary/data/Files_above_length_8.csv"

print(f"Reading {input_file}...")
df = pd.read_csv(input_file)

print(f"Original shape: {df.shape}")
print(f"Number of rows: {len(df)}")
print(f"Number of unique FileIDs: {df['FileID'].nunique()}")

print("\nBefore sorting - Sample of first 10 rows:")
print(df[['FileID', 'TimeInFile', 'Filename']].head(10))

# Sort by FileID first, then by TimeInFile within each FileID
print("\nSorting by FileID (primary) and TimeInFile (secondary)...")
df = df.sort_values(by=['FileID', 'TimeInFile'], ascending=[True, True])

# Reset index after sorting
df = df.reset_index(drop=True)

print("\nAfter sorting - Sample of first 10 rows:")
print(df[['FileID', 'TimeInFile', 'Filename']].head(10))

# Verify sorting worked correctly
print("\nVerification - checking a few FileIDs:")
for file_id in df['FileID'].unique()[:3]:
    file_data = df[df['FileID'] == file_id]
    times = file_data['TimeInFile'].values
    is_sorted = all(times[i] <= times[i+1] for i in range(len(times)-1))
    print(f"  FileID {file_id}: {len(file_data)} rows, TimeInFile sorted: {is_sorted}")
    print(f"    TimeInFile range: {times[0]} to {times[-1]}")

# Save sorted data
#print(f"\nSaving to {output_file}...")
#df.to_csv(output_file, index=False)

print("\nDone!")
print(f"\nSummary:")
#print(f"  Input file: {input_file}")
#print(f"  Output file: {output_file}")
print(f"  Total rows: {len(df):,}")
print(f"  Total FileIDs: {df['FileID'].nunique()}")
print(f"  Sorting: FileID (ascending) → TimeInFile (ascending)")

