import os
import argparse
from supabase import create_client, Client
from openai import OpenAI

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Query Supabase with vector similarity search')
parser.add_argument('--table', type=str, help='Supabase table name')
args = parser.parse_args()

# Initialize clients
supabase: Client = create_client(
    os.environ.get("SUPABASE_URL"),
    os.environ.get("SUPABASE_API_KEY")
)
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def generate_embedding(text: str) -> list[float]:
    """Generate embedding for text using OpenAI."""
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def main():
    # Get user inputs
    print("\n=== Supabase Vector Search ===\n")
    
    # Use command-line arg if provided, otherwise prompt user
    table_name = args.table if args.table else input("Enter the table name to search: ").strip()
    question = input("Enter your search question: ").strip()
    
    # Optional parameters with defaults
    match_threshold_input = input("Enter match threshold (0.0-1.0, default 0.50): ").strip()
    match_threshold = float(match_threshold_input) if match_threshold_input else 0.50
    
    match_count_input = input("Enter number of results (default 10): ").strip()
    match_count = int(match_count_input) if match_count_input else 10
    
    print(f"\nGenerating embedding for: '{question}'...")
    
    # Generate embedding
    embedding = generate_embedding(question)
    embedding_str = '[' + ','.join(map(str, embedding)) + ']'
    
    # The RPC function name should be match_{table_name}
    rpc_function = f"match_{table_name}"
    
    print(f"Searching table '{table_name}' using RPC function '{rpc_function}'...\n")
    
    try:
        # Execute vector similarity search
        response = supabase.rpc(
            rpc_function,
            {
                'query_embedding': embedding_str,
                'match_threshold': match_threshold,
                'match_count': match_count
            }
        ).execute()
        
        # Display results
        if response.data:
            print(f"Found {len(response.data)} matches:\n")
            print("=" * 80)
            
            for i, record in enumerate(response.data, 1):
                print(f"\nResult {i}:")
                print("-" * 80)
                
                # Display all fields in the record
                for key, value in record.items():
                    if key != 'embedding':  # Skip embedding field (too long)
                        print(f"  {key}: {value}")
                
                # If similarity score exists, display it
                if 'similarity' in record:
                    print(f"  Similarity Score: {record['similarity']:.4f}")
                
            print("=" * 80)
        else:
            print("No matches found.")
            
    except Exception as e:
        print(f"\nError querying Supabase: {e}")
        print(f"\nMake sure:")
        print(f"  1. The RPC function '{rpc_function}' exists in your Supabase database")
        print(f"  2. The table '{table_name}' has an 'embedding' column")
        print(f"  3. Your SUPABASE_URL and SUPABASE_API_KEY environment variables are set")

if __name__ == "__main__":
    main()
