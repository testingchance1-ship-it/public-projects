import pandas as pd
import re

def convert_date_column_name(col_name):
    """Convert date format from M_D_YYYY to date_YYYY_D_M"""
    # Check if it matches the date pattern M_D_YYYY or MM_DD_YYYY
    date_pattern = r'^(\d{1,2})_(\d{1,2})_(\d{4})$'
    match = re.match(date_pattern, col_name)
    
    if match:
        month = match.group(1)
        day = match.group(2)
        year = match.group(3)
        return f"date_{year}_{day}_{month}"
    else:
        return col_name

# Read the CSV file
input_file = 'cleaned_test production data for ingestion.csv'
df = pd.read_csv(input_file)

print(f"Original columns (first 10): {df.columns[:10].tolist()}")

# Rename columns and convert to lowercase
new_columns = [convert_date_column_name(col).lower() for col in df.columns]
df.columns = new_columns

print(f"\nNew columns (first 10): {df.columns[:10].tolist()}")

# Save with the same filename
df.to_csv(input_file, index=False)

print(f"\nSuccessfully updated and saved: {input_file}")
print(f"Total columns renamed: {sum(1 for old, new in zip(df.columns, new_columns) if old != new)}")
