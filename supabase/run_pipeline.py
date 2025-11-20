#!/usr/bin/env python3
"""
Master script to execute the complete CSV to Supabase pipeline.
Runs all steps sequentially from CSV cleaning to RPC function generation.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_script(script_name, description, skip_prompt=False, args=None):
    """Run a Python script and handle its execution."""
    print("\n" + "="*80)
    print(f"STEP: {description}")
    print("="*80)
    print(f"Running: {script_name}\n")
    
    if not skip_prompt:
        proceed = input(f"Press Enter to continue or 'skip' to skip this step: ").strip().lower()
        if proceed == 'skip':
            print(f"Skipping {script_name}...")
            return True
    
    try:
        # Build command with arguments
        cmd = [sys.executable, script_name]
        if args:
            cmd.extend(args)
        
        # Run the script
        result = subprocess.run(
            cmd,
            cwd=os.getcwd(),
            check=False
        )
        
        if result.returncode != 0:
            print(f"\n⚠️  Warning: {script_name} exited with code {result.returncode}")
            retry = input("Continue anyway? (yes/no): ").strip().lower()
            if retry not in ['yes', 'y']:
                return False
        else:
            print(f"\n✓ {script_name} completed successfully")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error running {script_name}: {e}")
        retry = input("Continue anyway? (yes/no): ").strip().lower()
        return retry in ['yes', 'y']

def main():
    """Execute the complete pipeline."""
    print("="*80)
    print("CSV TO SUPABASE PIPELINE")
    print("="*80)
    print("\nThis script will guide you through all steps to:")
    print("1. Clean your CSV")
    print("2. Generate SQL schema")
    print("3. Format column names")
    print("4. Import data to Supabase")
    print("5. Generate RPC function for querying")
    print("\nYou'll be prompted for inputs at each step.")
    
    input("\nPress Enter to begin...")
    
    # Get CSV filename and table name from user (only once)
    print("\n" + "="*80)
    print("CONFIGURATION")
    print("="*80)
    csv_filename = input("Enter the cleaned CSV filename (will be used throughout pipeline): ").strip()
    table_name = input("Enter the Supabase table name (will be used throughout pipeline): ").strip()
    print(f"\n✓ Using CSV: {csv_filename}")
    print(f"✓ Using table: {table_name}")
    
    # Step 1: Clean CSV
    if not run_script(
        "generic_csv_cleaner.py",
        "Step 1: Clean CSV file (may lose column headers depending on format)"
    ):
        print("\n❌ Pipeline stopped by user")
        return
    
    # Step 2: Extract headers
    if not run_script(
        "modify_csv_testing.py",
        "Step 2: Extract headers and create headers_output.txt",
        args=["--csv", csv_filename, "--table", table_name]
    ):
        print("\n❌ Pipeline stopped by user")
        return
    
    # Step 3: Modify headers for SQL
    if not run_script(
        "header_modify_for_sql.py",
        "Step 3: Convert headers to SQL-friendly format"
    ):
        print("\n❌ Pipeline stopped by user")
        return
    
    print("\n" + "="*80)
    print("⚠️  MANUAL STEP REQUIRED")
    print("="*80)
    print("Please verify that 'embedding vector(1536)' column is in header_modified_for_sql.txt")
    print("The script should have added it automatically, but please double-check.")
    input("\nPress Enter once you've verified...")
    
    # Step 4: Rearrange date columns
    if not run_script(
        "rearrange_date.py",
        "Step 4: Rearrange date column names in CSV"
    ):
        print("\n❌ Pipeline stopped by user")
        return
    
    print("\n" + "="*80)
    print("⚠️  MANUAL STEP REQUIRED")
    print("="*80)
    print("Before continuing:")
    print("1. Copy contents of 'header_modified_for_sql.txt'")
    print("2. Paste into Supabase SQL Editor")
    print("3. Run the SQL to create the table")
    input("\nPress Enter once table is created in Supabase...")
    
    # Step 5: Import to Supabase
    if not run_script(
        "csv_import_to _supabase.py",
        "Step 5: Import CSV data to Supabase with embeddings",
        args=["--csv", csv_filename, "--table", table_name]
    ):
        print("\n❌ Pipeline stopped by user")
        return
    
    # Step 6: Generate RPC function
    if not run_script(
        "generate_rpc_function.py",
        "Step 6: Generate RPC function for vector similarity search"
    ):
        print("\n❌ Pipeline stopped by user")
        return
    
    print("\n" + "="*80)
    print("⚠️  MANUAL STEP REQUIRED")
    print("="*80)
    print("Before you can query:")
    print("1. Copy contents of 'rpc_function_production.sql'")
    print("2. Paste into Supabase SQL Editor")
    print("3. Run the SQL to create the RPC function")
    input("\nPress Enter to continue...")
    
    # Step 7: Query option
    print("\n" + "="*80)
    print("✓ PIPELINE COMPLETE")
    print("="*80)
    print("\nYour data is now in Supabase and ready to query!")
    print("\nYou can now use 'query_supabase_generic.py' to search your database.")
    
    query_now = input("\nWould you like to query the database now? (yes/no): ").strip().lower()
    
    if query_now in ['yes', 'y']:
        run_script(
            "query_supabase_generic.py",
            "Step 7: Query Supabase database",
            skip_prompt=True,
            args=["--table", table_name]
        )
    
    print("\n" + "="*80)
    print("Pipeline execution finished!")
    print("="*80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Pipeline interrupted by user")
        sys.exit(1)
