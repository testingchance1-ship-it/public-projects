# this is a generic csv cleaner that can handle various csv formats and clean them for vector dbs

import pandas as pd
import numpy as np
import re
import argparse
from pathlib import Path
from typing import Optional, List, Dict

class GenericCSVCleaner:
    """
    A flexible CSV cleaner that can handle various CSV formats and prepare them 
    for vector databases or machine learning applications.
    """
    
    def __init__(self, input_file: str, config: Optional[Dict] = None):
        self.input_file = input_file
        self.config = config or {}
        self.df = None
        self.original_shape = None
        self.cleaning_report = []
        
    def load_csv(self, encoding: str = 'utf-8', delimiter: str = ','):
        """Load CSV file with error handling."""
        try:
            self.df = pd.read_csv(self.input_file, encoding=encoding, delimiter=delimiter)
            self.original_shape = self.df.shape
            self.log(f"✓ Loaded CSV: {self.original_shape[0]} rows × {self.original_shape[1]} columns")
            return True
        except Exception as e:
            self.log(f"✗ Error loading CSV: {e}")
            # Try alternative encodings
            for enc in ['latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    self.df = pd.read_csv(self.input_file, encoding=enc, delimiter=delimiter)
                    self.original_shape = self.df.shape
                    self.log(f"✓ Loaded CSV with {enc} encoding: {self.original_shape[0]} rows × {self.original_shape[1]} columns")
                    return True
                except:
                    continue
            return False
    
    def log(self, message: str):
        """Log cleaning operations."""
        self.cleaning_report.append(message)
        print(message)
    
    def detect_and_remove_header_rows(self, threshold: float = 0.7):
        """
        Auto-detect and remove header rows that aren't properly parsed.
        Looks for rows where most values are strings when rest of column is numeric.
        """
        if self.df is None:
            return
        
        rows_to_drop = []
        
        for idx in range(min(10, len(self.df))):  # Check first 10 rows
            row = self.df.iloc[idx]
            # Check if row contains mostly string values that look like headers
            string_count = sum(isinstance(val, str) and not val.replace('.', '').replace('-', '').isdigit() 
                             for val in row)
            if string_count / len(row) > threshold:
                rows_to_drop.append(idx)
        
        if rows_to_drop:
            self.df = self.df.drop(rows_to_drop).reset_index(drop=True)
            self.log(f"✓ Removed {len(rows_to_drop)} header row(s)")
    
    def clean_column_names(self):
        """Clean and standardize column names."""
        if self.df is None:
            return
        
        original_cols = self.df.columns.tolist()
        new_cols = []
        
        for idx, col in enumerate(original_cols):
            col_str = str(col)
            
            # Remove or replace unnamed columns
            if col_str.startswith('Unnamed:') or col_str.strip() == '':
                new_cols.append(f'column_{idx}')
            else:
                # Clean the column name
                clean_col = col_str.strip()
                # Replace special characters with underscore
                clean_col = re.sub(r'[^\w\s-]', '_', clean_col)
                # Replace spaces with underscore
                clean_col = re.sub(r'\s+', '_', clean_col)
                # Remove consecutive underscores
                clean_col = re.sub(r'_+', '_', clean_col)
                # Remove leading/trailing underscores
                clean_col = clean_col.strip('_')
                new_cols.append(clean_col)
        
        # Handle duplicate column names
        seen = {}
        final_cols = []
        for col in new_cols:
            if col in seen:
                seen[col] += 1
                final_cols.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                final_cols.append(col)
        
        self.df.columns = final_cols
        
        changed = sum(1 for o, n in zip(original_cols, final_cols) if str(o) != n)
        if changed > 0:
            self.log(f"✓ Cleaned {changed} column name(s)")
    
    def remove_empty_columns(self, threshold: float = 0.95):
        """Remove columns with mostly missing values."""
        if self.df is None:
            return
        
        missing_ratio = self.df.isnull().sum() / len(self.df)
        cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
        
        if cols_to_drop:
            self.df = self.df.drop(columns=cols_to_drop)
            self.log(f"✓ Removed {len(cols_to_drop)} empty column(s) (>{threshold*100}% missing)")
    
    def remove_empty_rows(self, threshold: float = 0.95):
        """Remove rows with mostly missing values."""
        if self.df is None:
            return
        
        initial_rows = len(self.df)
        missing_ratio = self.df.isnull().sum(axis=1) / len(self.df.columns)
        self.df = self.df[missing_ratio <= threshold].reset_index(drop=True)
        
        removed = initial_rows - len(self.df)
        if removed > 0:
            self.log(f"✓ Removed {removed} empty row(s) (>{threshold*100}% missing)")
    
    def clean_numeric_columns(self):
        """Clean and convert numeric columns."""
        if self.df is None:
            return
        
        cleaned_count = 0
        
        for col in self.df.columns:
            # Skip if already numeric
            if pd.api.types.is_numeric_dtype(self.df[col]):
                continue
            
            # Try to clean and convert to numeric
            original = self.df[col].copy()
            
            # Remove common numeric formatting
            cleaned = self.df[col].astype(str)
            cleaned = cleaned.str.replace(',', '')  # Remove commas
            cleaned = cleaned.str.replace('$', '')  # Remove dollar signs
            cleaned = cleaned.str.replace('€', '')  # Remove euro signs
            cleaned = cleaned.str.replace('%', '')  # Remove percent signs
            cleaned = cleaned.str.strip()  # Remove whitespace
            
            # Replace empty strings with NaN
            cleaned = cleaned.replace(['', 'nan', 'NaN', 'None', 'null'], np.nan)
            
            # Try to convert to numeric
            numeric_col = pd.to_numeric(cleaned, errors='coerce')
            
            # If more than 50% of non-null values can be converted, use numeric
            non_null_original = original.notna().sum()
            non_null_numeric = numeric_col.notna().sum()
            
            if non_null_original > 0 and (non_null_numeric / non_null_original) > 0.5:
                self.df[col] = numeric_col
                cleaned_count += 1
        
        if cleaned_count > 0:
            self.log(f"✓ Cleaned and converted {cleaned_count} numeric column(s)")
    
    def handle_missing_values(self, strategy: str = 'auto'):
        """
        Handle missing values with various strategies.
        
        Strategies:
        - 'auto': Smart fill based on column type
        - 'drop': Drop rows with any missing values
        - 'zero': Fill with 0
        - 'mean': Fill numeric with mean
        - 'median': Fill numeric with median
        - 'mode': Fill with most common value
        - 'forward': Forward fill
        """
        if self.df is None:
            return
        
        initial_missing = self.df.isnull().sum().sum()
        
        if initial_missing == 0:
            self.log("✓ No missing values found")
            return
        
        if strategy == 'drop':
            self.df = self.df.dropna().reset_index(drop=True)
        
        elif strategy == 'zero':
            self.df = self.df.fillna(0)
        
        elif strategy == 'mean':
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
            # Fill non-numeric with mode or empty string
            for col in self.df.select_dtypes(exclude=[np.number]).columns:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else '')
        
        elif strategy == 'median':
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
            for col in self.df.select_dtypes(exclude=[np.number]).columns:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else '')
        
        elif strategy == 'mode':
            for col in self.df.columns:
                if not self.df[col].mode().empty:
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        
        elif strategy == 'forward':
            self.df = self.df.fillna(method='ffill').fillna(method='bfill')
        
        elif strategy == 'auto':
            # Smart fill based on data type and context
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            
            # For numeric columns, use median
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
            
            # For categorical/string columns, use mode or 'unknown'
            for col in self.df.select_dtypes(exclude=[np.number]).columns:
                if not self.df[col].mode().empty:
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
                else:
                    self.df[col] = self.df[col].fillna('unknown')
        
        final_missing = self.df.isnull().sum().sum()
        self.log(f"✓ Handled missing values: {initial_missing} → {final_missing} (strategy: {strategy})")
    
    def remove_duplicate_rows(self):
        """Remove duplicate rows."""
        if self.df is None:
            return
        
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        removed = initial_rows - len(self.df)
        
        if removed > 0:
            self.log(f"✓ Removed {removed} duplicate row(s)")
    
    def detect_and_convert_dates(self):
        """Attempt to detect and convert date columns."""
        if self.df is None:
            return
        
        converted = 0
        
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                continue
            
            # Try to convert to datetime
            try:
                date_col = pd.to_datetime(self.df[col], errors='coerce')
                # If more than 50% can be converted, it's likely a date column
                if date_col.notna().sum() / len(self.df) > 0.5:
                    self.df[col] = date_col
                    converted += 1
            except:
                continue
        
        if converted > 0:
            self.log(f"✓ Detected and converted {converted} date column(s)")
    
    def normalize_numeric_columns(self, method: str = 'none'):
        """
        Normalize numeric columns.
        
        Methods:
        - 'none': No normalization
        - 'minmax': Scale to 0-1 range
        - 'standard': Z-score normalization (mean=0, std=1)
        """
        if self.df is None or method == 'none':
            return
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if method == 'minmax':
            for col in numeric_cols:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val != min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
            self.log(f"✓ Applied min-max normalization to {len(numeric_cols)} column(s)")
        
        elif method == 'standard':
            for col in numeric_cols:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val != 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
            self.log(f"✓ Applied standard normalization to {len(numeric_cols)} column(s)")
    
    def generate_report(self):
        """Generate a cleaning report."""
        print("\n" + "="*70)
        print("CLEANING REPORT")
        print("="*70)
        
        for item in self.cleaning_report:
            print(item)
        
        print("\n" + "-"*70)
        print("FINAL DATA SUMMARY")
        print("-"*70)
        print(f"Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"\nData types:")
        print(self.df.dtypes.value_counts())
        print(f"\nMissing values: {self.df.isnull().sum().sum()}")
        
        print("\n" + "-"*70)
        print("FIRST FEW ROWS")
        print("-"*70)
        print(self.df.head())
        
        if len(self.df.select_dtypes(include=[np.number]).columns) > 0:
            print("\n" + "-"*70)
            print("NUMERIC COLUMN STATISTICS")
            print("-"*70)
            print(self.df.describe())
    
    def save(self, output_file: str, index: bool = False):
        """Save cleaned data to CSV."""
        if self.df is None:
            print("No data to save!")
            return False
        
        try:
            self.df.to_csv(output_file, index=index)
            self.log(f"\n✓ Saved cleaned data to: {output_file}")
            return True
        except Exception as e:
            self.log(f"\n✗ Error saving file: {e}")
            return False
    
    def clean(self, 
              remove_empty_cols: bool = True,
              remove_empty_rows: bool = True,
              clean_names: bool = True,
              clean_numerics: bool = True,
              handle_missing: str = 'auto',
              remove_duplicates: bool = True,
              detect_dates: bool = False,
              normalize: str = 'none'):
        """
        Execute full cleaning pipeline.
        
        Args:
            remove_empty_cols: Remove columns with >95% missing values
            remove_empty_rows: Remove rows with >95% missing values
            clean_names: Clean column names
            clean_numerics: Clean and convert numeric columns
            handle_missing: Strategy for missing values
            remove_duplicates: Remove duplicate rows
            detect_dates: Try to detect and convert date columns
            normalize: Normalization method ('none', 'minmax', 'standard')
        """
        print("\n" + "="*70)
        print("STARTING CSV CLEANING PROCESS")
        print("="*70 + "\n")
        
        if clean_names:
            self.clean_column_names()
        
        if remove_empty_cols:
            self.remove_empty_columns()
        
        if remove_empty_rows:
            self.remove_empty_rows()
        
        if clean_numerics:
            self.clean_numeric_columns()
        
        if detect_dates:
            self.detect_and_convert_dates()
        
        if handle_missing:
            self.handle_missing_values(strategy=handle_missing)
        
        if remove_duplicates:
            self.remove_duplicate_rows()
        
        if normalize != 'none':
            self.normalize_numeric_columns(method=normalize)
        
        self.generate_report()
        
        return self.df


def main():
    # Interactive mode - ask for input and output files
    print("="*70)
    print("GENERIC CSV CLEANER FOR VECTOR DATABASES")
    print("="*70)
    
    input_file = input("\nEnter the input CSV file path: ").strip()
    output_file = input("Enter the output CSV file path (or press Enter for default): ").strip()
    
    # Generate output filename if not provided
    if not output_file:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"cleaned_{input_path.name}")
        print(f"Using default output file: {output_file}")
    
    # Ask for additional options
    print("\nCleaning Options:")
    print("1. Basic cleaning (default)")
    print("2. Aggressive cleaning with normalization")
    print("3. Custom options")
    
    choice = input("\nSelect cleaning mode (1-3) [default: 1]: ").strip() or "1"
    
    # Set default options
    encoding = 'utf-8'
    delimiter = ','
    clean_names = True
    remove_empty = True
    clean_numerics = True
    missing_strategy = 'auto'
    remove_duplicates = True
    detect_dates = False
    normalize = 'none'
    
    if choice == "2":
        normalize = 'minmax'
        detect_dates = True
        print("Using aggressive cleaning with min-max normalization and date detection")
    elif choice == "3":
        # Custom options
        missing_strategy = input("Missing value strategy (auto/drop/zero/mean/median/mode/forward) [auto]: ").strip() or 'auto'
        normalize = input("Normalization method (none/minmax/standard) [none]: ").strip() or 'none'
        detect_dates_input = input("Detect and convert date columns? (y/n) [n]: ").strip().lower()
        detect_dates = detect_dates_input == 'y'
    
    # Create cleaner instance
    cleaner = GenericCSVCleaner(input_file)
    
    # Load CSV
    if not cleaner.load_csv(encoding=encoding, delimiter=delimiter):
        print("Failed to load CSV file!")
        return
    
    # Clean the data
    cleaner.clean(
        remove_empty_cols=remove_empty,
        remove_empty_rows=remove_empty,
        clean_names=clean_names,
        clean_numerics=clean_numerics,
        handle_missing=missing_strategy,
        remove_duplicates=remove_duplicates,
        detect_dates=detect_dates,
        normalize=normalize
    )
    
    # Save cleaned data
    cleaner.save(output_file)


if __name__ == "__main__":
    main()
