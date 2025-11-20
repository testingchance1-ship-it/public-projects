import re

def is_date_format(text):
    """Check if text matches a date format like 1/31/2025, 1_31_2025, 12/31/2025, etc."""
    # Pattern to match date formats: M/D/YYYY or M_D_YYYY or MM/DD/YYYY or MM_DD_YYYY
    date_pattern = r'^\d{1,2}[/_]\d{1,2}[/_]\d{4}$'
    return bool(re.match(date_pattern, text))

def convert_date_to_column_name(date_str):
    """Convert date format like 1/31/2025 or 1_31_2025 to date_2025_31_1 (date_year_day_month)"""
    # Split the date string by either / or _
    if '/' in date_str:
        parts = date_str.split('/')
    else:
        parts = date_str.split('_')
    
    month = parts[0]
    day = parts[1]
    year = parts[2]
    
    # Return in format: date_year_day_month
    return f"date_{year}_{day}_{month}"

# Read the headers_output.txt file
with open('headers_output.txt', 'r') as f:
    content = f.read()

# Split into lines
lines = content.strip().split('\n')

# Process each line
modified_lines = []
for line in lines:
    # Skip the first two lines (create table and id serial primary key)
    if 'create table' in line.lower() or 'id serial primary key' in line:
        modified_lines.append(line)
    # Handle embedding vector line specially
    elif 'embedding vector' in line.lower():
        modified_lines.append(line)
    # Skip the closing parenthesis
    elif line.strip() == ')':
        modified_lines.append(line)
    # Process column lines
    elif line.strip() and line.strip() != ',':
        # Remove trailing comma if present
        column_name = line.strip().rstrip(',')
        
        # Check if it's a date format
        if is_date_format(column_name):
            # Convert date format to SQL-friendly column name
            new_column_name = convert_date_to_column_name(column_name)
            modified_lines.append(f'    {new_column_name} numeric,')
        else:
            modified_lines.append(f'    {column_name} text,')
    else:
        modified_lines.append(line)

# Join back together
modified_content = '\n'.join(modified_lines)

# Write to output file
with open('header_modified_for_sql.txt', 'w') as f:
    f.write(modified_content)

print("Modified headers saved to header_modified_for_sql.txt")
print("\nPreview of first 15 lines:")
print('\n'.join(modified_lines[:15]))