import re

def extract_table_name_and_columns(sql_content):
    """Extract table name and column definitions from CREATE TABLE statement."""
    # Extract table name
    table_match = re.search(r'create table (\w+)\s*\(', sql_content, re.IGNORECASE)
    if not table_match:
        raise ValueError("Could not find table name in SQL")
    
    table_name = table_match.group(1)
    
    # Extract column definitions (between CREATE TABLE and closing parenthesis)
    columns_match = re.search(r'create table \w+\s*\((.*)\)', sql_content, re.IGNORECASE | re.DOTALL)
    if not columns_match:
        raise ValueError("Could not extract column definitions")
    
    columns_text = columns_match.group(1)
    
    # Parse individual columns
    columns = []
    for line in columns_text.strip().split('\n'):
        line = line.strip().rstrip(',')
        if not line or line.lower().startswith('--'):
            continue
        
        # Split by whitespace to get column name and type
        parts = line.split()
        if len(parts) >= 2:
            col_name = parts[0]
            col_type = parts[1]
            
            # Skip if it's a constraint like PRIMARY KEY
            if col_name.lower() not in ['primary', 'foreign', 'unique', 'check', 'constraint']:
                columns.append({
                    'name': col_name,
                    'type': col_type,
                    'sql_type': ' '.join(parts[1:])  # Full type definition
                })
    
    return table_name, columns

def generate_rpc_function(table_name, columns):
    """Generate SQL RPC function for vector similarity search."""
    
    # Start building the SQL
    sql_parts = []
    
    # Add index creation
    sql_parts.append(f"""-- Create an index on the embedding column for faster similarity search
create index on {table_name} using ivfflat (embedding vector_cosine_ops)
with (lists = 100);
""")
    
    # Start RPC function
    sql_parts.append(f"""-- Create RPC function for vector similarity search
create or replace function match_{table_name} (
  query_embedding vector(1536),
  match_threshold float,
  match_count int
)
returns table (""")
    
    # Add return columns
    return_cols = []
    for col in columns:
        return_cols.append(f"  {col['name']} {col['sql_type']}")
    
    # Add similarity column
    return_cols.append("  similarity float")
    
    sql_parts.append(",\n".join(return_cols))
    sql_parts.append("\n)\nlanguage sql stable\nas $$")
    
    # Add SELECT statement
    select_cols = [f"    {table_name}.{col['name']}" for col in columns]
    select_cols.append(f"    1 - ({table_name}.embedding <=> query_embedding) as similarity")
    
    sql_parts.append("\n  select")
    sql_parts.append(",\n".join(select_cols))
    
    # Add FROM and WHERE clauses
    sql_parts.append(f"  from {table_name}")
    sql_parts.append(f"  where 1 - ({table_name}.embedding <=> query_embedding) > match_threshold")
    sql_parts.append(f"  order by {table_name}.embedding <=> query_embedding")
    sql_parts.append("  limit match_count;")
    sql_parts.append("$$;")
    
    return "\n".join(sql_parts)

def main():
    # Read the SQL file
    input_file = 'header_modified_for_sql.txt'
    
    print(f"Reading {input_file}...\n")
    
    with open(input_file, 'r') as f:
        sql_content = f.read()
    
    # Extract table info
    table_name, columns = extract_table_name_and_columns(sql_content)
    
    print(f"Found table: {table_name}")
    print(f"Found {len(columns)} columns\n")
    
    # Generate RPC function
    rpc_sql = generate_rpc_function(table_name, columns)
    
    # Create output filename
    output_file = f'rpc_function_{table_name}.sql'
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(rpc_sql)
    
    print(f"✓ RPC function generated and saved to: {output_file}")
    print(f"\nYou can now:")
    print(f"1. Copy the contents of {output_file}")
    print(f"2. Paste into Supabase SQL Editor")
    print(f"3. Run the SQL to create the RPC function 'match_{table_name}'")
    print(f"4. Use query_supabase_generic.py to query from terminal")
    
    # Print preview
    print("\n" + "="*80)
    print("PREVIEW OF GENERATED SQL:")
    print("="*80)
    print(rpc_sql[:500] + "..." if len(rpc_sql) > 500 else rpc_sql)

if __name__ == "__main__":
    main()
