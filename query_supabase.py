import os
from supabase import create_client, Client
from openai import OpenAI

# Initialize clients
supabase: Client = create_client(
    os.environ.get("SUPABASE_URL"),
    os.environ.get("SUPABASE_API_KEY")
)
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def generate_embedding(text: str) -> list[float]:
    """Generate embedding for text."""
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

# Your search query
query = input("What do you want to search up ")

# Generate embedding and format for Supabase
embedding = generate_embedding(query)
embedding_str = '[' + ','.join(map(str, embedding)) + ']'

# Search using your RPC function, ive replaced the match_(table name) with an input so you can call any table you have, makes it more generic and reusable
# leave the query embedding alone but you can modify the match threshold and/or match count to fit your needs or turn them into inputs if you need
response = supabase.rpc(
    input("enter the database you want to search: starts with match_(table name) "),
    {
        'query_embedding': embedding_str,
        'match_threshold': 0.50,
        'match_count': 10
    }
).execute()

# Print results
print(f"Found {len(response.data)} matches:\n")
for i, customer in enumerate(response.data, 1):
    print(f"{i}. {customer.get('first_name')} {customer.get('last_name')}")
    print(f"   Company: {customer.get('company')}")
    print(f"   City: {customer.get('city')}, {customer.get('country')}\n")