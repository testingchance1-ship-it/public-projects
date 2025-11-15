#this will fetch the data of an existing supabase table
#all you have to do is fill in the table name when prompted

from dotenv import load_dotenv
from supabase import create_client, Client
import os

load_dotenv()

supabase_url=os.environ.get("SUPABASE_URL")
supabase_key= os.environ.get("SUPABASE_API_KEY")
supabase: Client = create_client(supabase_url=supabase_url, supabase_key=supabase_key)

response = (
    supabase.table(input("Enter table_name "))
    .select("*")
    .execute()
)
print(response.data)