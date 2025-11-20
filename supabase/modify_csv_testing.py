#this will scrape the column headers from a cleaned csv and export them to a text file for easy import to supabase table creation

import pandas as pd
import numpy as np
import sys
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Extract CSV headers for SQL table creation')
parser.add_argument('--csv', type=str, help='Path to the cleaned CSV file')
parser.add_argument('--table', type=str, help='Supabase table name')
args = parser.parse_args()

# Use command-line args if provided, otherwise prompt user
csv_file = args.csv if args.csv else input("Enter the path to the cleaned CSV file: ")
table_name = args.table if args.table else input("Enter the table name: ")

df = pd.read_csv(csv_file)
print(df)

#select column headers
columns = list(df.columns)
print(columns)

# Use the full column names (not split)
headers = pd.Series(columns)
print(headers)

with open('headers_output.txt', 'w') as f:
    # Write the SQL table creation header
    f.write(f"create table {table_name}(\n")
    f.write("    id serial primary key,\n")
    
    # Write each column header
    for header in headers:
        f.write(f"    {header},\n")
    
    # Close the SQL statement
    f.write(")\n")

print("\nHeaders exported to headers_output.txt with SQL table creation")

