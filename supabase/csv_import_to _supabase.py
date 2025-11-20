#to use the below python script, there are some manual setups to do first:
#be warned this script and instructions uses the provided files in the github, you will need to modify the below instructions if you modify or want ot input your own csv
#login to supabase and create a project
#open the sql editor and run the below command to create a table named 'customers'
#create table customers (
#    id serial primary key,
#    index text,
#    customer_id text,
#    first_name text,
#    last_name text,
#    company text,
#    city text,
#    country text,
#    phone_1 text,
#    phone_2 text,
#    email text,
#    subscription_date text,
#    website text,
#    embedding vector(1536)
#);
# this script uses local environment variables so make sure to change that poriton, if you use something else
# once the table is setup in supabase from the sql editor, you can run the script
#there is a bunch of logging to show you where you are in the process and where something fails
#good luck!

import csv
import os
from supabase import create_client, Client
from openai import OpenAI

import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Import CSV to Supabase with embeddings')
parser.add_argument('--csv', type=str, help='CSV filename')
parser.add_argument('--table', type=str, help='Supabase table name')
args = parser.parse_args()

# Configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_API_KEY")  # Changed to match your env variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Use command-line args if provided, otherwise prompt user
CSV_FILE_PATH = args.csv if args.csv else input("Enter the name of the csv file: ")
TABLE_NAME = args.table if args.table else input("Enter the Supabase table name: ")

# Initialize clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def generate_embedding(text: str) -> list[float]:
    """Generate embedding vector for given text using OpenAI."""
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def process_csv_to_supabase(csv_path: str):
    """Read CSV, generate embeddings, and insert into Supabase."""
    
    with open(csv_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        
        for idx, row in enumerate(csv_reader):
            try:
                # Combine relevant fields into text for embedding
                # Adjust this based on your CSV structure
                text_for_embedding = ' '.join([
                    str(value) for value in row.values() if value
                ])
                
                # Generate embedding
                embedding = generate_embedding(text_for_embedding)
                
                # Prepare data for insertion
                # Convert embedding to the format Supabase expects: "[0.1,0.2,0.3,...]"
                embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                
                data = {
                    **row,  # Include all original CSV fields
                    'embedding': embedding_str  # Add the embedding vector as formatted string
                }
                
                # Insert into Supabase
                result = supabase.table(TABLE_NAME).insert(data).execute()
                
                print(f"Row {idx + 1} inserted successfully")
                
            except Exception as e:
                print(f"Error processing row {idx + 1}: {str(e)}")
                continue

if __name__ == "__main__":
    # Verify environment variables are set
    if not all([SUPABASE_URL, SUPABASE_KEY, OPENAI_API_KEY]):
        raise ValueError("Please set SUPABASE_URL, SUPABASE_KEY, and OPENAI_API_KEY environment variables")
    
    print(f"Processing CSV file: {CSV_FILE_PATH}")
    process_csv_to_supabase(CSV_FILE_PATH)
    print("Processing complete!")

