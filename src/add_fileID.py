import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

input_file = "original_lake2.csv" #moved outside repo for space reasons
output_file = "original_lake2_with_FileID.csv" #moved outside repo for space reasons

def modify_csv(input_file, output_file):

    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)

    print(f"Original shape: {df.shape}")
    print(f"Number of rows: {len(df)}")

    # Get unique filenames in the order they appear
    unique_filenames = df['Filename'].unique()
    print(f"Number of unique filenames: {len(unique_filenames)}")

    # Create a mapping from filename to FileID (starting at 1)
    filename_to_id = {filename: idx + 1 for idx, filename in enumerate(unique_filenames)}

    # Add FileID column at the end
    df['FileID'] = df['Filename'].map(filename_to_id)

    print(f"New shape: {df.shape}")
    print(f"\nFirst few rows with FileID:")
    print(df[['Filename', 'FileID']].head(10))

    print(f"\nVerifying FileID assignment:")
    print(f"  Rows with FileID=1: {len(df[df['FileID'] == 1])} (Filename: {df[df['FileID'] == 1]['Filename'].iloc[0]})")
    print(f"  Rows with FileID=2: {len(df[df['FileID'] == 2])} (Filename: {df[df['FileID'] == 2]['Filename'].iloc[0]})")
    print(f"  Rows with FileID=3: {len(df[df['FileID'] == 3])} (Filename: {df[df['FileID'] == 3]['Filename'].iloc[0]})")

    # Save to new CSV file
    print(f"\nSaving to {output_file}...")
    df.to_csv(output_file, index=False)

    print("Done!")
    print(f"\nSummary:")
    print(f"  Total rows: {len(df)}")
    print(f"  Total unique files: {len(unique_filenames)}")
    print(f"  FileID range: 1 to {df['FileID'].max()}")
    print(f"  Output saved to: {output_file}")
    return df

def create_histogram():
    df = modify_csv(input_file, output_file)
    print("\nCreating histogram...")
    rows_per_file = df.groupby('FileID').size()

    plt.figure(figsize=(12, 6))
    # Create bins at integer boundaries for proper alignment
    bins = range(rows_per_file.min(), rows_per_file.max() + 2)  # +2 to include max value
    plt.hist(rows_per_file, bins=bins, edgecolor='black', alpha=0.7, color='steelblue', align='left')
    plt.xlabel('Number of Rows per File', fontsize=12)
    plt.ylabel('Frequency (Number of Files)', fontsize=12)
    plt.title('Distribution of Rows per FileID', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')

    # Add statistics to the plot
    mean_rows = rows_per_file.mean()
    median_rows = rows_per_file.median()
    plt.axvline(mean_rows, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_rows:.1f}')
    plt.axvline(median_rows, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_rows:.1f}')
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.legend()

    plt.tight_layout()
    plt.savefig('rows_per_fileID_histogram.png', dpi=300, bbox_inches='tight')
    print("Histogram saved as: rows_per_fileID_histogram.png")
    plt.show()

    print(f"\nHistogram Statistics:")
    print(f"  Mean rows per file: {mean_rows:.2f}")
    print(f"  Median rows per file: {median_rows:.2f}")
    print(f"  Min rows per file: {rows_per_file.min()}")
    print(f"  Max rows per file: {rows_per_file.max()}")
    print(f"  Std deviation: {rows_per_file.std():.2f}")

create_histogram()